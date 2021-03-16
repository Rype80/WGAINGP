# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 9:11:01 2021

@author: De Stille Fotograaf
"""
# Imports needed for this function to work
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # FATAL
import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm

# imports from files that are contained in map
from DeStilleFotograaf import PlotHists, PlotLoss, PlotAccs, PRD_Calculator, PRD_Plot, Progress_Plot
from SupportGAIN import CreateBatches, Hint_Matrix, MissingColumns, AccuracyNames, Correct_Imputations, AddAccuracy
from utils import normalization, renormalization, rounding


# GAIN Function. We are trying to impute data from some tabular data form. In this function we give the option to use
# Embeddings. If this is selected, the embedding layers are trained in the genrrator part of the model. 
# Which columns should be embedded should be selected beforehand. For this select the column numbers of columns
# you want to embed from. 

# The steps of the GAIN are as follows:
    # 1. Split data into embedding part and non-embedding part. For embedding part, we create an embedding layer
    #    for every variable. For the non-embedding part, we split the data between the part we are imputing from and 
    #    the part we are usingas features. This implies that we generate noise which we only add to the imputation part. 
    # 2. Define Generator: The Generator function will create the generator network. This network consists of an embedding 
    #    part and the actual generator part. IF embeddings are not selected, the generator function will only return the generator model.
    # 3. Define Discriminator/Critic: The discriminator is used to impute real from imputed data. The discriminator assumes that the embedded variables
    #    are embedded, and therefore expects a higher dimension of the data-matrix than the original data. If no embeddings are selected,
    #    the discriminator expects other input size. It automatically knows if the data is embedded or not. 
    # 4. Losses & Optimizers: Define Loss functions for generator and discrimator as well as the used optimizers. 
    # 5. Training process: Define train process for generator and discriminator, as well as an overal training scheme. This is based on
    #    back-propagation. In case embeddings are selected, it automtatically will update the embedded layers as well. 
    # 6. Run model: Initialise all parts 1-5 and start running the model.
# Additionally to these steps, we will also track the progress of the GAIN by means of progression plots. The user can decide whether 
# he/she/it wants to have these plots avaiable beforehand. Note that a output_path should be provided to store pictures for a moment. 
# For this some additional functions have been written. 
# Besides this, some other functions have been written in order to let the main functions work. 
# Examples are the Batch function, the hint creator function

def WGAINGP(actual_data, data_x,gain_parameters, ceiling,
         Embed_Cols=[], Var_Names=[], embeddings=True, Plot=False, out_path=None, generator_path=None, seed=123):
    # Initialisation of all parameters
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set everything to float 64
    tf.keras.backend.set_floatx('float64')
    
    # Parameters 
    # Model parameters
    N_Critic= 5
    
     # System parameters
    batch_size= gain_parameters['batch_size']
    hint_rate= gain_parameters['hint_rate']
    Alpha=gain_parameters['alpha']
    lambda_gp = gain_parameters['lambda_gp']
    Epochs = gain_parameters['epochs']
    Condintioned= gain_parameters['conditioned_hints']
    conditioned_MSE = gain_parameters['conditioned_MSE']
    conditioned_critic = gain_parameters['conditioned_critic']
    
    # Adam Parameters
    Learning_Rate= 10e-5
    Beta_1 = 0
    Beta_2 = 0.9
    
    # Embedding ceiling
    Ceiling= ceiling
    # Plot parameters
    margins=[0,0]
    plotcols=1
    Plot_Epoch = 1
    if(Plot==True):
        # Path for placing images You need to select this
        assert out_path!= None, "We cannot plot without specifying a path"
        assert len(Var_Names)!=0, "Need variable names"
    
    
    # Break training treshold
    Epoch_Tresh=100
    # Epsilon value for break criterion
    Break_Epsilon= 10e-6
    
    # Step 1:
    # Functions for embeddings
    # Create embedding for categorical variable
    def Entity_Embedding(data, cat, ceil):
    
        Name= str("Cat-Var_" + str(cat))
        Embed_Name= str("Embed_Col_" + str(cat))
        
        Elements= len(np.unique(data[:,cat]))
        Embed_Dim= (int) (min((np.ceil(Elements)/2),ceil))
        Input = keras.Input(shape=1, name=Name)
        Embedding= layers.Embedding(output_dim=Embed_Dim, input_dim=(Elements+2), trainable=True, name=Embed_Name)(Input)
        Flat=layers.Flatten()(Embedding)
        
        return([Input, Flat])
    
    # Combine the embeddings for all categorical variables
    def All_Embedded(data, cats, ceil):
        Embeddings= []
        i=0 
        for cat in cats:
            Embeddings.append(Entity_Embedding(data,cat, ceil[i]))
            i+=1
        return(Embeddings)
    
    
    # Extract Embedding weights which can be used as inputs for other 
    # Machine Learning models
    def Get_Embeddings():
        Embed_vars=len(Embed)
        Embeddings=[]
        for i in range(Embed_vars,Embed_vars*2):
           Embeddings.append([Embed[i-Embed_vars],generator.layers[i].get_weights()[0]])
        return(Embeddings)
    # Function that creates input for a generator
    def GeneratorInput(Input_Data, M, batch):
        Actual_Size= (int) (np.shape(batch)[0])
        
        # Make Embedding ready as input for generator
        Embed_Batch= Input_Data[0][batch]
        Embed_Input = []
        for embedding in range(np.shape(Embed_Batch)[1]):
            Embed_Input.append(np.reshape(Embed_Batch[:,embedding],(Actual_Size,1)))
            
        # Define data of other part
        X_Rest =  Input_Data[1][batch]
        # Define mask matrix for other data
        M_Rest = M[1][batch]
         
        # Generate noise for other part
        Z= np.random.normal(0,1,(Actual_Size, Dim_Rest))
        
        # Create data input of other data for generator
        X_Rest= X_Rest*M_Rest + Z*(1-M_Rest)
        
        # Create overall mask for generator
        M_Big=np.hstack((M[0],M[1]))[batch]
        
        # Concatenate X_Rest and M_Big such that it is suitable as input for generator
        Gen_Other= np.hstack((X_Rest, M_Big))
        
        return([Embed_Input, Gen_Other])
    
    # Step 2: 
    # Create model of data and pass it through generator.
    # Note that we add a boolean for embedding, which can be 
    # set to true or false. In case we return true, we pass through the 
    # embedding layers, if false, we start at the generator part 
    def Generator (data, cats, conts, ceil):
        
            # Create embeddings for all the categorical data
            All_embeddings= All_Embedded(data, cats,ceil)
            
            # Extract contents of embeddings
            Inputs = []
            Outputs= []
            Cat_Cols =0
            for i in range(len(All_embeddings)):
                Inputs.append(All_embeddings[i][0])
                Cat_Cols += np.shape(All_embeddings[i][1])[1]
                Outputs.append(All_embeddings[i][1])
            
            # Input shape for other variables
            other_Inputs = layers.Input(shape=len(conts)*2+Cat_Cols, name="Other_variables")
            # Merge all inputs
            Inputs.append(other_Inputs)
            Outputs.append(other_Inputs)
            
            # Concatenate embeddings
            Concatenate= layers.concatenate(Outputs, name="Concatenate")
            
            # Input dimension
            Dim= (int) (np.shape(Concatenate)[1]/2)
            
            # Generator
            Gen1 = layers.Dense(Dim, use_bias=True, 
                        activation="relu", name="Input_Gen")(Concatenate)
            Gen1 = layers.BatchNormalization()(Gen1)
          
            # Layer 2
            Gen2= layers.Dense(0.5*Dim, use_bias=True, activation='relu', name="Hidden_Gen")(Gen1)
            Gen2 = layers.BatchNormalization()(Gen2)

            # Layer 3
            Gen3= layers.Dense(Dim, use_bias=False,activation='sigmoid', name="Output_Gen")(Gen2)
            
            model= tf.keras.Model(inputs=Inputs,
                                  outputs = [Concatenate, Gen3])
                
                
            return(model)
        
    
    # Step 3: Define Discriminator
    def Critic():
        dim = Dim
        if (conditioned_critic == True):
            dim = len(Embedded_Imputations)
        model = tf.keras.Sequential()
        # Layer 1
        model.add(layers.Dense(dim, use_bias=True, input_shape=(2*dim,),  activation="relu"))
        model.add(tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True))
        
        # Layer 2
        model.add(layers.Dense(0.5*dim, use_bias=True, activation='relu'))
        model.add(tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True))
        
        # Layer 3
        model.add(layers.Dense(1, use_bias=True, activation='tanh'))
        
        return model
    
    
    # Step 4: Define Losses and Optimizers. 
    def Opt_Generator():
        return(tf.keras.optimizers.Adam(learning_rate=Learning_Rate, beta_1=Beta_1, beta_2=Beta_2))
        
    def Opt_Critic():
        return(tf.keras.optimizers.Adam(learning_rate=Learning_Rate, beta_1=Beta_1, beta_2=Beta_2))
        
    # For the generator, we need to define a loss function for its output.
    # Since the generator produces fake samples, we call it fake
    
    def Loss_Generator(Generated, Actual, D_Prob, M):
        # We only compute MSE loss over the columns we impute
        MSE_loss = tf.reduce_mean((M * Actual - M* Generated)**2) / tf.reduce_mean(M)
        Loss_Generator =  -1*tf.reduce_mean(D_Prob) +Alpha*MSE_loss
        return ([Loss_Generator,MSE_loss])
    
    # For the discriminator, we need to have a loss function that is able to 
    # distinguish between real and fake samples. Hence, we create for the two types
    # a different loss function and combine them together. 
   
    # For the discriminator, we need to have a loss function that is able to 
    # distinguish between real and fake samples. Hence, we create for the two types
    # a different loss function and combine them together. 
    def Loss_Critic(Fake, Real):
        Total_Loss = -1*(tf.reduce_mean(Real)- tf.reduce_mean(Fake) )
        return (Total_Loss)  
    
    
    # Gradient Penatly for WGAIN-GP
    def Gradient_Penalty(fake,real, H):
        dim = Dim
        if (conditioned_critic == True):
            dim = len(Embedded_Imputations)
        # Actual size of batch
        Actual_Size= (int) (np.shape(real)[0])
        # Generate epsilon for interpolation of data
        epsilon = tf.cast(tf.random.uniform([Actual_Size,dim], 0., 1.),dtype=tf.float64)
        # Interpolation of real and imputed data
        Interpolation_X = real*epsilon + (1-epsilon)*fake

        Disc_Input = np.hstack((Interpolation_X,H))
        Disc_Input=  tf.convert_to_tensor(Disc_Input)
        # Take gradient over critic in which we vary over
        # the interpolated values
        with tf.GradientTape() as g:
            g.watch(Disc_Input)
            # Critic over interpolation
            Mixed_Critic = critic(Disc_Input, training=True)
        grad = g.gradient(Mixed_Critic, [Disc_Input])[0]
        x_dim = (int) (np.shape(grad)[1]/2)
        grad=np.array(grad)[:,0:Dim]
        grad= tf.convert_to_tensor(grad)
        # L_2-norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad[:,0:x_dim]), axis=None))
        # Gradient Penalty
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp
    
    # Step 5: Training procedure
    # Train generator. In this process the embeddings are also trained
    def Train_Generator(real_data, Generator_Input, H_Batch):
        Dim=np.shape(H_Batch)[1]
        with tf.GradientTape() as Backprop_Generator:
            # Create generator output used for discriminator
            Embedded_data, Generated_Data = generator(inputs=Generator_Input, training=True)
            # Split Data in mask and actual data
            M_Big= Embedded_data[:,Dim:(int)(2*Dim)]
            Embedded_data= Embedded_data[:,0:(int)(Dim)]
            Disc_Input=  tf.convert_to_tensor(np.hstack((Generated_Data, H_Batch)))
            # Only condition on columns we are imputing
            if (conditioned_critic == True):
                M_Big = tf.gather(M_Big,Embedded_Imputations,axis=1)
                real_data = tf.gather(real_data,Embedded_Imputations,axis=1)
                Generated_Impute = tf.gather(Generated_Data,Embedded_Imputations,axis=1)
                H_impute = tf.gather(H_Batch,Embedded_Imputations,axis=1)
                # Convert data such that it is readable for discriminator
                Disc_Input=  tf.convert_to_tensor(np.hstack((Generated_Impute, H_impute)))
            Generated_Prob= critic(Disc_Input, training=False)
            if(conditioned_MSE == True):
                real_data = tf.gather(real_data,Embedded_Imputations,axis=1)
                Generated_Data = tf.gather(Generated_Data,Embedded_Imputations,axis=1)
                H_Batch = tf.gather(H_Batch,Embedded_Imputations,axis=1)
                if(conditioned_critic==False):
                    M_Big = tf.gather(M_Big,Embedded_Imputations,axis=1)
                    real_data = tf.gather(real_data,Embedded_Imputations,axis=1)
            Generator_Loss=Loss_Generator(Generated_Data, real_data, Generated_Prob,M_Big)[0]
            # Compute Gradient
            Grad_Generator = Backprop_Generator.gradient(Generator_Loss, generator.trainable_variables)
            # Update weights
            Opt_Generator().apply_gradients(zip(Grad_Generator, generator.trainable_variables))
    
    # Train Critic
    def Train_Critic(fake, real, H):
        # Condition on the columns which we are imputing
        if (conditioned_critic == True):
            fake = tf.gather(fake, Embedded_Imputations, axis=1)
            real = tf.gather(real, Embedded_Imputations, axis=1)
            H = tf.gather(H, Embedded_Imputations, axis=1)
        F_Input = np.hstack((fake,H ))
        F_Input=  tf.convert_to_tensor(F_Input)
        R_Input = np.hstack((real,H))
        R_Input=  tf.convert_to_tensor(R_Input)
        with tf.GradientTape() as Backprop_Critic:
        
          # Discriminator gets the imputed data from generator as input + hint matrix
          D_Prob_F = critic(F_Input, training=True)
          D_Prob_R = critic(R_Input, training=True)
          
          gp = Gradient_Penalty(real, fake, H)
          # Compute loss for critic. Here we add Gradient Penalty
          Critic_Loss = Loss_Critic(D_Prob_F, D_Prob_R) + lambda_gp*gp
          # Compute gradient
          Grad_Critic = Backprop_Critic.gradient(Critic_Loss, critic.trainable_variables)
          # Update weights for critic
          Opt_Critic().apply_gradients(zip(Grad_Critic, critic.trainable_variables))
    
        # Overall training scheme
    def train(real_data, Input_Data, M, H, Batches):
        # Set seed
        np.random.seed(seed)  
        # Arrays that will be used to track progress
        # Lists which will store the accuracies
        Acc_Names = AccuracyNames(margins, Impute_Cols)
        # Store Discriminator/Critic Loss
        Disc_Loss=[]
        # Store RMSE over imputations
        RMSE=[]
        # Store MSE Loss of Generator
        MSE_Gen= []
        # Store Generator Loss
        Gen_Loss=[]
        # Store AUC 
        AUC=[]
        # Store generators of every epoch
        Generators=[]
        for impute in range(len(Var_Names)):
            AUC.append([])
            Generators.append([])
        # Check if AUC is best
        best_AUC=np.zeros(len(Var_Names))
        # Start time
        start = time.time()
        # Run for number of epochs
        Effective_Epochs = 1
        Break_Count=0
        for epoch in tqdm(range(Epochs)):
            for batch in Batches:
                Generator_Input= GeneratorInput(Input_Data, M, batch)
                
                # Create generator output used for discriminator
                Embed_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
                # Split Data in mask and actual data
                M_Big= Embed_Data[:,Dim:(int)(2*Dim)]
                Embed_Data= Embed_Data[:,0:(int)(Dim)]
                # Assure that we only generate data for imputations
                Generated_Data= M_Big*Embed_Data + (1-M_Big)*Generated_Data
                
                # Create the actual data which we use to train critic
                Real_Batch=real_data[batch]
                # Obtain embeddings from Embed_Data
                Embed_Size= (int) (np.sum(Layer_Size))
                Actual_Embed= Embed_Data[:,0:Embed_Size]
                Actual_Data = tf.convert_to_tensor(np.hstack((Actual_Embed, Real_Batch)))

                # Extract Hint matrix
                H_Batch=H[batch]
                
                for t in range(N_Critic):
                    Train_Critic(Generated_Data, Actual_Data, H_Batch)
                # Train Generator
                Train_Generator(Actual_Data, Generator_Input, H_Batch)
            
            #  Compute losses for discriminator/critic, generator and MSE (generator)
            Generator_Input= GeneratorInput(Input_Data, M, np.arange(0,Obs))
            # Create generator output used for discriminator
            Embed_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
            # Split Data in mask and actual data
            M_Big= Embed_Data[:,Dim:(int)(2*Dim)]
            Embed_Data= Embed_Data[:,0:(int)(Dim)]
            # Assure that we only generate data for imputations
            Generated_Data= M_Big*Embed_Data + (1-M_Big)*Generated_Data
            Disc_Loss, MSE_Gen, Gen_Loss,= All_Lost(Disc_Loss, MSE_Gen, Gen_Loss,
                                                      Generator_Input, H,M_Big) 
            # Obtain imputations and actual values
            Compares=Compare()[0]
            # Calculate AUC
            PRD = PRD_Calculator(Compares,Var_Names)
            for impute in range(len(Var_Names)):
                AUC[impute].append(np.trapz(PRD[impute][1],PRD[impute][0]))
                # Save best generator
                if(best_AUC[impute]<AUC[impute][epoch]):
                    best_AUC[impute]=AUC[impute][epoch]
                    generator.save(generator_path+"Train")
                    Generators[impute]= tf.keras.models.load_model(generator_path +"Train")
            
            # Plot progress    
            if((Plot==True)): 
                # Operations for accuracy plots:
                #  Compute accuracy
                Correct=Correct_Imputations(Compares,Impute_Cols, margins)
                #  Add accuracy to lists
                Acc_Names = AddAccuracy(Acc_Names, Correct, Impute_Cols, margins)
                            
                # Compute RMSE over imputations
                RMSE.append(Compare()[1])  
                
                if((epoch+1)%Plot_Epoch==0):
                    # Plot figures seperately, save as images
                    Hists= PlotHists(Compares,Var_Names, plotcols, out_path)
                    Accs= PlotAccs(Acc_Names, Var_Names, margins, plotcols, out_path)
                    Losses= PlotLoss(Disc_Loss, AUC, MSE_Gen, Gen_Loss,RMSE,out_path)
                    PRD=PRD_Plot(Compares,Var_Names,plotcols, out_path)
                
                    # Open images
                    Hists=Image.open(out_path + "Hists.png")
                    Accs=Image.open(out_path + "Accs.png")
                    PRD = Image.open(out_path + "PRD.png")
                    Losses= Image.open(out_path+ "/Loss.png")
                                
                    # Stack images into list
                    Combined=[Hists, Accs, PRD,  Losses]
                    # Plot all images
                    Complete=Progress_Plot(2,2,Combined,start,epoch, Var_Names, name="Embed WGAINGP")
                    plt.show()
            
            # Check if we should continue training
            if(epoch>0):
                # Obtain previous and current loss values of Generator
                Old_Loss = Gen_Loss[epoch-1] + 10e-8
                New_Loss = Gen_Loss[epoch]
                # Compute %-change
                Rel_Change = (New_Loss - Old_Loss)/Old_Loss
                Effective_Epochs = epoch+1
                # If %-change is too small, tak note of this.
                if(np.abs(Rel_Change)<Break_Epsilon):
                    Break_Count+=1
                    # If we do not change multiple times in a row, we break
                    if(Break_Count==Epoch_Tresh):
                        print("Allez, gij speelt met vuur")
                        if(Plot==True):
                            # Plot figures seperately, save as images
                            Hists= PlotHists(Compares,Var_Names, plotcols, out_path)
                            Accs= PlotAccs(Acc_Names, Var_Names, margins, plotcols, out_path)
                            Losses= PlotLoss(Disc_Loss, AUC, MSE_Gen, Gen_Loss,RMSE,out_path)
                            PRD=PRD_Plot(Compares,Var_Names,plotcols, out_path)
                        
                            # Open images
                            Hists=Image.open(out_path + "Hists.png")
                            Accs=Image.open(out_path + "Accs.png")
                            PRD = Image.open(out_path + "PRD.png")
                            Losses= Image.open(out_path+ "/Loss.png")
                                        
                            # Stack images into list
                            Combined=[Hists, Accs, PRD,  Losses]
                            # Plot all images
                            Complete=Progress_Plot(2,2,Combined,start,epoch, Var_Names, name="Embed WGAIN-GP")
                            plt.show()
                        break
                # In case we do change, reset count. 
                if(np.abs(Rel_Change)>Break_Epsilon):
                    Break_Count=0
        return(Effective_Epochs, best_AUC, Generators)
    
    # OTher functions         
    # Impute function
    def Impute():
        Actual_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
        Reduce_Imputations = np.array(Embedded_Imputations) - (int)(np.sum(Layer_Size))
        Impute_Data=np.array(Generated_Data[:,(int)(np.sum(Layer_Size)):Dim])
        Impute_Data = Impute_Data*(1-M_Rest) + Norm_Data_Rest*M_Rest 
        Impute_Data=renormalization(Impute_Data, Norm_Pars)
        Impute_Data = rounding(Impute_Data, Data_Rest) 
        return([Impute_Data,Reduce_Imputations])
       
    # Function which returns the actual missing values and the imputed values
    def Compare():
        Col_Dim=len(Impute_Cols)
        Actual=actual_data[:,Impute_Cols]
        impute_mask=origin_M[:,Impute_Cols]
        Actual=Actual[impute_mask==0]
        Actual= np.reshape(Actual, ((int)(np.shape(Actual)[0]/Col_Dim),Col_Dim))
        impute_mask=origin_M[:,Impute_Cols]
        Imputed=Impute()
        Imputed=Imputed[0][:,Imputed[1]]
        Imputed=Imputed[impute_mask==0]
        Imputed= np.reshape(Imputed, ((int)(np.shape(Imputed)[0]/Col_Dim),Col_Dim))
        compare= []
        RMSE=0
        for col in range(Col_Dim):
            new=np.vstack((Actual[:,col],Imputed[:,col]))
            RMSE += np.sum((Imputed[:,col]-Actual[:,col])**2)
            compare.append(new)
        T= np.shape(Imputed)[0]*np.shape(Imputed)[1]
        RMSE = (RMSE/T)**(1/2)
        return([compare,RMSE])
        
    # Function that computes both discriminator/critic and generator losses used for plots
    # and adds it to a list
    def All_Lost(Disc_Loss, MSE_Loss, Gen_Loss, Generator_Input, H, M_Big):
        
        # Compute Loss for generator
        Embed_Data, First_Generated_Data = generator(inputs=Generator_Input, training=False)
        # Split Data in mask and actual data
        M_Big= Embed_Data[:,Dim:(int)(2*Dim)]
        Embed_Data= Embed_Data[:,0:(int)(Dim)]
    
        # Assure that we only generate data for imputations
        Generated_Data= M_Big*Embed_Data + (1-M_Big)*First_Generated_Data
        
        # Use actual data
        # Obtain embeddings from Embed_Data
        Embed_Size= (int) (np.sum(Layer_Size))
        Actual_Embed= Embed_Data[:,0:Embed_Size]
        Actual_Data = np.hstack((Actual_Embed, Norm_Actual))
        Disc_Input=  tf.convert_to_tensor(np.hstack((Generated_Data, H)))
        # Condition on columns we impute
        Actual_temp = Actual_Data
        if(conditioned_critic==True):
            Generated_Data = tf.gather(Generated_Data, Embedded_Imputations, axis=1)
            Actual_Data = tf.gather(Actual_Data, Embedded_Imputations, axis=1)
            H = tf.gather(H, Embedded_Imputations, axis=1)
            Disc_Input=  tf.convert_to_tensor(np.hstack((Generated_Data, H)))
        
        F_Input = np.hstack((Generated_Data,H))
        F_Input=  tf.convert_to_tensor(F_Input)
        R_Input = np.hstack((Actual_Data,H))
        R_Input=  tf.convert_to_tensor(R_Input)
        F_Prob = critic(F_Input, training=False)
        R_Prob = critic(R_Input, training=False)         
        
        # Compute Loss discriminator
        Critic_Loss= Loss_Critic(F_Prob,R_Prob) + lambda_gp * Gradient_Penalty(Generated_Data, Actual_Data, H)
        
        # Restore actual data
        Actual_Data = Actual_temp
        Generated_Prob= critic(Disc_Input, training=False)
        if(conditioned_MSE == True):
            First_Generated_Data = tf.gather(First_Generated_Data, Embedded_Imputations, axis=1)
            Actual_Data = tf.gather(Actual_Data, Embedded_Imputations, axis=1)
            M_Big = tf.gather(M_Big, Embedded_Imputations, axis=1)
        Generator_Loss,MSE=Loss_Generator(First_Generated_Data, Actual_Data, Generated_Prob,M_Big) 
                   
        # Compute losses and add to list
        Gen_Loss.append(np.array(Generator_Loss))
        Disc_Loss.append(np.array(Critic_Loss))
        MSE_Loss.append(np.array(MSE))
        return([Disc_Loss, MSE_Loss, Gen_Loss])

    # Step 6: Run model
    # Create mask matrix from data
    origin_M = 1-np.isnan(data_x)
    Impute_Cols=MissingColumns(origin_M)
    if(Plot==True):
        assert len(Impute_Cols) <=2 , "For plotting, we specify two imputations at a time"
    # Deduce dimension
    Obs, Dim=np.shape(data_x)
    
    Embed= Embed_Cols
    # Check that we  have selected embedding variables
    assert len(Embed_Cols)>0 , "Specifcy Embedding variables"
    # Check that we are not imputing any value from the embedding variables
    for col in Impute_Cols:
        assert np.sum(np.array(Embed)==col) == 0 ,"We cannot impute from embeddings"
    # Split data into Embedding part and rest
    Rest=[]
    rest_start=0
    first=0
    for el in Embed_Cols:
        start=el
        add=np.arange(rest_start,start)        
        rest_start=el+1
        if(first==0):
            Rest= np.hstack(add)
        if(first!=0):
            Rest=np.hstack((Rest,add))
        first+=1
        
    if(rest_start!=Dim):
        add=np.arange(rest_start,Dim)
        Rest=np.hstack((Rest,add))

    
    # Define data part from which we will embed
    Data_Embed=data_x[:,Embed]
    
    # Define Layer output sizes for every embedding layer
    Layer_Size=np.zeros(len(Embed))
    
    # Keep track of the indices of the original column;
    # this will be used to create the hint-matrix for the 
    # discriminator
    Original_Column=[]
    first=True
    for embedding in range(len(Embed)):
        Layer_Size[embedding]=len(np.unique(Data_Embed[:,embedding]))
        Layer_Size[embedding]= (int) (min(int(Layer_Size[embedding]/2), Ceiling[embedding]))
        if(first == True):
            Original_Column= np.hstack(np.array([Embed[embedding]]*(int)(Layer_Size[embedding])))
        if(first == False):
            Original_Column= np.hstack((Original_Column,(np.array([Embed[embedding]]*(int)(Layer_Size[embedding])))))
        first=False
    # Append Original column with other data
    Original_Column=np.hstack((Original_Column,Rest))
    
    # Create mask part based on embedding output layers. Note that we assume 
    # that we will not impute on these values. This mask part will be used
    # as input for the generator
    M_Embed=np.ones((np.shape(Data_Embed)[0],(int)(np.sum(Layer_Size))))
    
    # Define Data part from which we will not embed
    Data_Rest=data_x[:,Rest]
    M_Rest= 1-np.isnan(Data_Rest)
                    
    # Normalise other data
    Norm_Data, Norm_Pars= normalization(Data_Rest)
    Norm_Data_Rest = np.nan_to_num(Norm_Data, 0)
    
    # For the actual data (without imputations), also make the embedding and rest split.
    # Here we add the embeddings to the actual data when these are trained from the generator. 
    Actual_Rest = actual_data[:,Rest]
    Norm_Actual, Actual_Pars= normalization(Actual_Rest)
    
    # Define dimensions of other part of data
    Obs_Rest, Dim_Rest= np.shape(Data_Rest)
        
    # Combine the two mask parts into the one big mask
    M= [M_Embed,M_Rest]
    # Also create numpy version of mask
    M_array= np.hstack((M_Embed,M_Rest))
    
    # Create Hint matrix, used for discriminator
    # First extract elements of columns which we want to impute from embedded data
    Embedded_Imputations=[]
    for col in Impute_Cols:
        Embedded_Imputations.append(np.where(Original_Column==col)[0][0]) 
    # Compute Hint matrix based on columns of big mask
    H= Hint_Matrix(hint_rate,M_array, Embedded_Imputations,conditioned=Condintioned)
    
    # Combine embeddings and actual data
    Input_Data=[Data_Embed, Norm_Data]
    
    # Define total dimension of our problem
    Dim=np.shape(M_array)[1]
    
    # Create batches
    Batches= CreateBatches(Obs,batch_size)
    
    # Create generator and discriminator/critic
    generator=Generator(data_x, Embed, Rest, Ceiling)
    critic= Critic()
    
    # Input for generator (embeddings, and mask)
    Generator_Input= GeneratorInput(Input_Data, M, np.arange(0,Obs))
    
    # track running time
    start= time.time()
    
    # train network
    Effective_Epochs, best_AUCs, Best_Generators= train(Norm_Actual, Input_Data, M, H, Batches)

    # Return running time
    Running_Time = round(time.time() - start ,0)
    
    # After training extract the embedded layers
    Actual_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
    
    # After training extract the embedded layers
    Actual_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
    # Get embedding weights
    Embeddings_After= Get_Embeddings()
    

    return([Embeddings_After, Best_Generators,  Effective_Epochs, Running_Time, best_AUCs])
    
        








