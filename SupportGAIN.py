# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 9:11:01 2021

@author: De Stille Fotograaf
"""

import numpy as np
import tensorflow as tf
from utils import normalization, renormalization, rounding, binary_sampler, rmse_loss 


# Additional functions
# Function that creates batches
def CreateBatches(N, batch_size):
    obs = np.arange(0,N)
    np.random.shuffle(obs)
    No_batches=(int)(np.ceil(len(obs)/batch_size))
    batches=[]
    start=0
    end= batch_size
    for i in range(No_batches):
        new= obs[start:end]
        start += batch_size
        end += batch_size
        batches.append(new)
    return(batches)
    
# Hint matrix, we provide two options:
    # 1. Random hint matrix
    # 2. Conditioned hint matrix
def Hint_Matrix(Hint_Rate,M,cols, conditioned=True):
    Obs, Dim= np.shape(M)
    C = len(cols)
    Hint = M.copy()
    if (conditioned == True):
        for col in cols:
            B = np.reshape(np.array(binary_sampler(Hint_Rate, Obs,1)),(Obs))
            Hint[:,col]= B*M[:,col] + 0.5*(1-B)
    if(conditioned==False):
        B= np.reshape(np.array(binary_sampler(Hint_Rate, Obs,Dim)),(Obs, Dim))
        Hint= B*M + 0.5*(1-B)
    return(Hint)

def MissingColumns(M):
    Obs, Dim= np.shape(M)
    cols = []
    for i in range(Dim):
        if(np.sum(M[:,i])!= Obs):
            cols.append(i)
    cols=np.array(cols)
    return(cols)

# Creates lists for accuracy plots
def AccuracyNames(margins, Impute_Cols):
    names=[]
    mc=0
    for i in range(len(Impute_Cols)):
        for j in range(len(margins)):
            Fun_Name = 'C%i' %Impute_Cols[i] + 'M%i' %margins[j] 
            names.append(Fun_Name)
            names[mc] = [names[mc]]
            mc+=1
    return(names)
   
# Define correct accuracy of imputations based on a certain margin
# margin=0 exactly correct, margin=1, 1 deviation in absolute value, etc. 
def Correct_Imputations(compare,Impute_Cols, margins):
  all_correct= []
  for margin in margins:
      correct=[]
      for col in range(len(Impute_Cols)):
          good= np.sum(np.abs(compare[col][0,:]-compare[col][1,:])<=margin)/np.shape(compare[col])[1]
          correct.append(good)
      correct=np.array(correct)
      all_correct.append(correct)
    
  n_correct= all_correct[0]
  for j in range(1,len(margins)):
    n_correct=np.vstack((n_correct,all_correct[j]))
      
  return(n_correct)    
   
# Functions adds accuracies for new iteration      
def AddAccuracy(names, n_correct, cols, margins):
    mc=0        
    for i in range(len(cols)):
        for j in range(len(margins)):
            names[mc].append(n_correct[j,i])
            mc+=1
    return(names)



# Impute function
def Impute(data_x, X, M, generator, Norm_Parameters):
    Gen_Input = np.hstack((X,M ))
    Gen_Input=  tf.convert_to_tensor(Gen_Input)
    Imputed_Data= generator(Gen_Input, training=False)
    Imputed_Data = M * X + (1-M) * Imputed_Data  
    Imputed_Data= np.array(Imputed_Data)
    # Renormalization
    Imputed_Data = renormalization(Imputed_Data, Norm_Parameters)      
    # Rounding
    Imputed_Data = rounding(Imputed_Data, data_x) 
    return(Imputed_Data)

# Function which returns the actual missing values and the imputed values
def Compare(Act_X,X,M, cols, generator, Norm_Parameters):
    Col_Dim=len(cols)
    Actual=Act_X[M==0]
    Actual= np.reshape(Actual, ((int)(np.shape(Actual)[0]/Col_Dim),Col_Dim))
    Imputed = Impute(Act_X,X,M, generator, Norm_Parameters)[M==0]
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

