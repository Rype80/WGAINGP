# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 9:11:01 2021

@author: De Stille Fotograaf
"""

import numpy as np
from utils import normalization, renormalization, rounding, binary_sampler, rmse_loss 


# Function which extracts column indices for non-embedding variables
def RestIndex(all_data, Embed_Cols):
    N, Dim= np.shape(all_data)
    # Split data in Embed and rest
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
    
    return(Rest)

# Function that makes input for a generator model with embedding layers.
# This function is used to transform the test dataset
def Embedded_GainInput(data_x, Embed_Cols, Impute_Cols, Ceiling, seed=123): 
        Obs, Dim=np.shape(data_x)
        # Split data into Embedding part and rest
        Embed= Embed_Cols
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
            Layer_Size[embedding]= (int) ((min((Layer_Size[embedding]/2), Ceiling[embedding])))
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
        # Combine embeddings and actual data
        Input_Data=[Data_Embed, Norm_Data]
        
        Dim=np.shape(M_array)[1]
        
        Generator_Input=GeneratorInput(Input_Data, M, np.arange(0,Obs))
        
        return([Generator_Input, Embedded_Imputations, Layer_Size, Dim,
               Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest])

def GeneratorInput(Input_Data, M, batch):
    Actual_Size= (int) (np.shape(batch)[0])
    
    # Make Embedding ready as input for generator
    Embed_Batch= Input_Data[0][batch]
    Embed_Input = []
    for embedding in range(np.shape(Embed_Batch)[1]):
        Embed_Input.append(np.reshape(Embed_Batch[:,embedding],(Actual_Size,1)))
        
    # Define data of other part
    X_Rest =  Input_Data[1][batch]
    Dim_Rest= np.shape(X_Rest)[1]
    # Define mask matrix for other data
    M_Rest = M[1][batch]
     
    # Generate noise for other part
    Z= np.random.uniform(0,1,(Actual_Size, Dim_Rest))
    
    # Create data input of other data for generator
    X_Rest= X_Rest*M_Rest + Z*(1-M_Rest)
    
    # Create overall mask for generator
    M_Big=np.hstack((M[0],M[1]))[batch]
        
    # Concatenate X_Rest and M_Big such that it is suitable as input for generator
    Gen_Other= np.hstack((X_Rest, M_Big))
    
    return([Embed_Input, Gen_Other])


# Impute function
def Embed_Impute(Generator_Input, Embedded_Imputations, Layer_Size, Dim,
           Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest,Impute_Cols, generator):
    Actual_Data, Generated_Data = generator(inputs=Generator_Input, training=False)
    Reduce_Imputations = np.array(Embedded_Imputations) - (int)(np.sum(Layer_Size))
    Impute_Data=np.array(Generated_Data[:,(int)(np.sum(Layer_Size)):Dim])
    Impute_Data = Impute_Data*(1-M_Rest) + Norm_Data_Rest*M_Rest 
    Impute_Data=renormalization(Impute_Data, Norm_Pars)
    Impute_Data = rounding(Impute_Data, Data_Rest) 
    return([Impute_Data,Reduce_Imputations])

# Function which returns the actual missing values and the imputed values
def Embed_Compare(Generator_Input, Embedded_Imputations, Layer_Size, Dim,
           Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest, 
           actual_data,origin_M, Impute_Cols,generator):
    Col_Dim=len(Impute_Cols)
    Actual=actual_data[:,Impute_Cols]
    impute_mask=origin_M[:,Impute_Cols]
    Actual=Actual[impute_mask==0]
    Actual= np.reshape(Actual, ((int)(np.shape(Actual)[0]/Col_Dim),Col_Dim))
    impute_mask=origin_M[:,Impute_Cols]
    Imputed=Embed_Impute(Generator_Input, Embedded_Imputations, Layer_Size, Dim,
           Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest,Impute_Cols, generator)
    Imputed=Imputed[0][:,Imputed[1]]
    Imputed=Imputed[impute_mask==0]
    Imputed= np.reshape(Imputed, ((int)(np.shape(Imputed)[0]/Col_Dim),Col_Dim))
    compare= []
    for col in range(Col_Dim):
        new=np.vstack((Actual[:,col],Imputed[:,col]))
        compare.append(new)
    return(compare)

# Function that transforms a categorical variable into a matrix of embeddings
def EmbeddingTransformer(data, column, name, Embedding_weights):
    cat_data= data[:,column]
    Obs=(int) (np.shape(cat_data)[0]) 
    unique_values = (np.sort(np.unique(data[:,column])))
    layer_size=(int) (np.shape(Embedding_weights)[1])
    layers= np.array(list(map(str,np.arange(0,layer_size))),dtype=object)
    # Create variable names for embedded layers
    layer_name= np.add(np.full(layer_size,name),layers)
    
    # Create matix to store embedding values
    Embedding_Transform=np.full((Obs,layer_size),np.reshape(cat_data,(len(cat_data),1)))
    for level in unique_values:
        satisfied=np.where(Embedding_Transform[:,0]== level)[0]
        Embedding_Transform[satisfied,]=Embedding_weights[(int)(level),]

    return([layer_name, Embedding_Transform])

def EmbeddedData(all_data, created_embeddings, Ceiling, Embed_Cols, Embed_Names, Impute_Cols ):
    N, Dim= np.shape(all_data)
    # Split data in Embed and rest
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
    
    Rest_Data=all_data[:,Rest]
    Data_Embed=all_data[:,Embed_Cols]
        
    # Define Layer output sizes for every embedding layer
    Layer_Size=np.zeros(len(Embed_Cols))
    
    Original_Column=[]
    first=True
    for embedding in range(len(Embed_Cols)):
        Layer_Size[embedding]=len(np.unique(Data_Embed[:,embedding]))
        Layer_Size[embedding]= (int) (min(int(Layer_Size[embedding]/2), Ceiling[embedding]))
        if(first == True):
            Original_Column= np.hstack(np.array([Embed_Cols[embedding]]*(int)(Layer_Size[embedding])))
        if(first == False):
            Original_Column= np.hstack((Original_Column,(np.array([Embed_Cols[embedding]]*(int)(Layer_Size[embedding])))))
        first=False
    # Append Original column with other data
    Original_Column=np.hstack((Original_Column,Rest))

    Embed_Transforms=[]
    Embedding_Names=[]
    for i in range(len(Embed_Cols)): 
        data_transform=EmbeddingTransformer(all_data, created_embeddings[i][0], Embed_Names[i], created_embeddings[i][1])
        if(i==0):
            Embed_Transforms=(data_transform[1])
            Embedding_Names=np.reshape(data_transform[0],(1,len(data_transform[0])))
        if (i>0):
            Embed_Transforms=np.hstack((Embed_Transforms, data_transform[1]))
            Embedding_Names=np.hstack((Embedding_Names,np.reshape(data_transform[0],(1,len(data_transform[0])))))
            
    # Embedded data
    Embedded_Data=np.hstack((Embed_Transforms, Rest_Data))
    
    # Create Hint matrix, used for discriminator
    # First extract elements of columns which we want to impute from embedded data
    Embedded_Imputations=[]
    for col in Impute_Cols:
        Embedded_Imputations.append(np.where(Original_Column==col)[0][0]) 
    
    return([Embedded_Data,Embedded_Imputations])


def EmbeddedOrder(all_data, created_embeddings, Ceiling, Embed_Cols, Impute_Cols ):
    N, Dim= np.shape(all_data)
    # Split data in Embed and rest
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
    
    Rest_Data=all_data[:,Rest]
    Rest_Data=np.reshape(Rest_Data,(np.shape(Rest_Data)[1],))
    Data_Embed=all_data[:,Embed_Cols]
        
    # Define Layer output sizes for every embedding layer
    Layer_Size=np.zeros(len(Embed_Cols))
    
    Original_Column=[]
    first=True
    
    
    for embedding in range(len(Embed_Cols)):
        Layer_Size[embedding]=Ceiling[embedding]
        if(first == True):
            Original_Column= np.hstack(np.array([Embed_Cols[embedding]]*(int)(Layer_Size[embedding])))
        if(first == False):
            Original_Column= np.hstack((Original_Column,(np.array([Embed_Cols[embedding]]*(int)(Layer_Size[embedding])))))
        first=False
    # Append Original column with other data
    Original_Column=np.hstack((Original_Column,Rest))
        
    # Embed_Transforms=[]
    for i in range(len(Embed_Cols)): 
        level= (int)(all_data[:,Embed_Cols[i]])
        data_transform=created_embeddings[i][0][level]
        if(i==0):
            Embed_Transforms=(data_transform)
        if (i>0):
            Embed_Transforms=np.hstack((Embed_Transforms, data_transform))
            
    # Embedded data   
    Embedded_Data=np.hstack((Embed_Transforms, Rest_Data))
    
    # Create Hint matrix, used for discriminator
    # First extract elements of columns which we want to impute from embedded data
    Embedded_Imputations=[]
    for col in Impute_Cols:
        Embedded_Imputations.append(np.where(Original_Column==col)[0][0]) 
    
    return([Embedded_Data, Original_Column, Embedded_Imputations])

