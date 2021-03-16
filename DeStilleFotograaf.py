# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 9:11:01 2021

@author: WW-uitkering hobbyist 
"""

import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
from prd_score import compute_prd, _prd_to_f_beta, prd_to_max_f_beta_pair, plot 
import time
from itertools import cycle


# Plot histograms for empircal distributions
def PlotHists(compare, Impute_Cols, plotcols=1, out_path=None,test=False):
    title_size=11
    label_size=11
    
    N_cols= len(Impute_Cols)
    plotrows= (int) (np.ceil(N_cols/plotcols))
    if(len(Impute_Cols)==2):
        title_size=11
        label_size=14
        f,a = plt.subplots(2,1, figsize=[4,4])
        f.suptitle("Histograms", size=title_size,y= 0.99, x=0.53)
        f.text(0.53, 0, 'Value', ha='center', size=label_size)
        f.text(-0.003, 0.5, 'Density', va='center', rotation='vertical', size=label_size)
        a = a.ravel()
        for idx,ax in enumerate(a):
            ax.hist(compare[idx][0], bins=len(np.unique(compare[idx][0,])), ec="k", density=True, label="Actual")
            ax.hist(compare[idx][1], bins=len(np.unique(compare[idx][1,])), ec="k", density=True,
                    alpha=0.6, label="Imputed")
            ax.set_title(Impute_Cols[idx], size=title_size)
            plt.tight_layout(h_pad=0.25)
        f.legend(labels=["Actual", "Imputed"] ,loc='upper center', ncol=2, 
                 bbox_to_anchor=[0.53, 0.97],frameon=False, fontsize=9)
    if(len(Impute_Cols)==1):
        f,a = plt.subplots(1,1, figsize=[4,4])
        f.suptitle("Histograms", size=title_size,y= 0.99, x=0.53)
        f.text(0.53, 0.001, 'Value', ha='center', size=label_size)
        f.text(-0.003, 0.5, 'Density', va='center', rotation='vertical', size=label_size)
        a.hist(compare[0][0], bins=len(np.unique(compare[0][0,])), ec="k", density=True,
               label="Actual")
        a.hist(compare[0][1], bins=len(np.unique(compare[0][1,])), ec="k", density=True, 
               alpha=0.6, label="Imputed")
        a.set_title(Impute_Cols[0], size=title_size)
        plt.tight_layout(h_pad=0.25)
        f.legend(labels=["Actual", "Imputed"] ,loc='upper center', ncol=2, 
                 bbox_to_anchor=[0.53, 0.97],frameon=False, fontsize=9)
    plt.close(f)
    if(test==False):
        f.savefig(out_path+ "Hists.png", dpi=310)
    if(test==True):
        f.savefig(out_path+ "Hists_TestSet.png", dpi=310)
    return(f)
    
 # Plot accuracies
def PlotAccs(Names, Impute_Cols, margins, plotcols, out_path=None):
    # Text sizes for plots
    title_size=11
    label_size=11
    
    # Create names for legend
    Margin_Names=[]
    for margin in range(1):
        Name= 'Error Margin = %i' %margins[margin] 
        Margin_Names.append(Name)
    
    count= len(Names[0])-1
    x=np.arange(1,count+1)        
    
    N_cols= len(Impute_Cols)
    N_margins=len(margins)
    plotrows= (int) (np.ceil(N_cols/plotcols))
    
    if(len(Impute_Cols)==2):
        f,a = plt.subplots(2,1, sharex=True, figsize=[4,4])
        f.suptitle("Accuracies", size=title_size, x=0.53)
        f.text(0.53, 0.0, 'Epoch', ha='center', size=label_size)
        f.text(-0.001, 0.5, 'Accuracy', va='center', rotation='vertical', size=label_size)
        plt.tight_layout()
        a = a.ravel()
        row=0
        column=0
        mc=0
        for idx,ax in enumerate(a):
            if(mc<len(Names)):
                for j in range(1):
                    ax.plot(x,Names[(mc+j)][1:(count+1)], label=Margin_Names[j])  
            mc+=3
            ax.set_title(Impute_Cols[idx], size=title_size)
            
        f.legend(labels=Margin_Names ,loc='upper center', ncol=len(margins), 
                 bbox_to_anchor=[0.53, 0.96],frameon=False, fontsize=8)
        plt.tight_layout()
        plt.close(f)
    
    if(len(Impute_Cols)==1):
        f,asq = plt.subplots(plotrows,plotcols, sharex=True, figsize=[4,4])
        f.suptitle("Accuracies", size=title_size, x=0.55)
        f.text(0.53, -0.0001, 'Epoch', ha='center', size=label_size)
        f.text(-0.003, 0.5, 'Accuracy', va='center', rotation='vertical', size=label_size)
        plt.tight_layout()
        row=0
        column=0
        mc=0
        if(mc<len(Names)):
            for j in range(1):
                asq.plot(x,Names[(mc+j)][1:(count+1)], label=Margin_Names[j])  
            mc+=3
            asq.set_title(Impute_Cols[0], size=title_size)
            
        f.legend(labels=Margin_Names ,loc='upper center', ncol=len(margins), 
                 bbox_to_anchor=[0.53, 0.95],frameon=False, fontsize=8)
        plt.tight_layout()
        plt.close(f)     
    
    
    f.savefig(out_path+"Accs.png", dpi=310)
    
    return(f)
    
# Compute PRD --> used for computing AUC
def PRD_Calculator(compare, Impute_Cols):
    # Store all PR_Pairs
        PR_Pairs=[]
        
        # For all columns create PRD-Plots
        for i in range(len(Impute_Cols)):
            # First convert histograms to density arrays
            # Extract bin values for every category
            actual=np.array(compare[i][0],dtype=np.int32)
            Actual_Unique, Actual_Counts = np.unique(actual, return_counts=True)
            Actual_Parttable= np.transpose(np.stack([Actual_Unique, Actual_Counts]))
            
            impute=np.array(compare[i][1],dtype=np.int32)  
            Impute_Unique, Impute_Counts = np.unique(impute, return_counts=True)
            Impute_Parttable= np.transpose(np.stack([Impute_Unique, Impute_Counts]))
            
            # Min and Max values over the two arrays
            min_val= min(min(actual),min(impute))
            max_val= max(max(actual),max(impute))
            
            # Create probabilities
            Actual_Table = np.zeros((max_val-min_val +1 ))
            Impute_Table = np.zeros((max_val-min_val +1 ))
            Actual_Cat=0
            Impute_Cat=0
            count=0
            # Fill aray
            for i in range(min_val, (max_val+1)): 
                Actual_Count=0
                Impute_Count=0
                if(i==Actual_Parttable[Actual_Cat][0]):
                    Actual_Count=Actual_Parttable[Actual_Cat][1]
                    if(Actual_Cat!= (len(Actual_Parttable)-1)):
                        Actual_Cat+=1
                if(i==Impute_Parttable[Impute_Cat][0]):
                    Impute_Count=Impute_Parttable[Impute_Cat][1]
                    if(Impute_Cat!= (len(Impute_Parttable)-1)):
                        Impute_Cat+=1
                Impute_Table[count]= Impute_Count
                Actual_Table[count]=Actual_Count
                count+=1
              
            # Normalise
            Actual_Table= Actual_Table/np.sum(Actual_Table)
            Impute_Table=Impute_Table/np.sum(Impute_Table)
            
            # Compute PRD
            PRD= compute_prd(Impute_Table, Actual_Table)
            PR_Pairs.append(PRD)
            
        return(PR_Pairs)

# Precision Recall Distribution plots
def PRD_Plot(compare, Impute_Cols, plotcols=1, out_path=None, test=False):
        # Create label names
        Column_Names=[]
        for column in range(len(Impute_Cols)):
            Name= Impute_Cols[column]
            Column_Names.append(Name)
        
        
        # Store all PR_Pairs
        PR_Pairs=[]
        
        # For all columns create PRD-Plots
        for i in range(len(Impute_Cols)):
            # First convert histograms to density arrays
            # Extract bin values for every category
            actual=np.array(compare[i][0],dtype=np.int32)
            Actual_Unique, Actual_Counts = np.unique(actual, return_counts=True)
            Actual_Parttable= np.transpose(np.stack([Actual_Unique, Actual_Counts]))
            
            impute=np.array(compare[i][1],dtype=np.int32)  
            Impute_Unique, Impute_Counts = np.unique(impute, return_counts=True)
            Impute_Parttable= np.transpose(np.stack([Impute_Unique, Impute_Counts]))
            
            # Min and Max values over the two arrays
            min_val= min(min(actual),min(impute))
            max_val= max(max(actual),max(impute))
            
            # Create probabilities
            Actual_Table = np.zeros((max_val-min_val +1 ))
            Impute_Table = np.zeros((max_val-min_val +1 ))
            Actual_Cat=0
            Impute_Cat=0
            count=0
            # Fill aray
            for i in range(min_val, (max_val+1)): 
                Actual_Count=0
                Impute_Count=0
                if(i==Actual_Parttable[Actual_Cat][0]):
                    Actual_Count=Actual_Parttable[Actual_Cat][1]
                    if(Actual_Cat!= (len(Actual_Parttable)-1)):
                        Actual_Cat+=1
                if(i==Impute_Parttable[Impute_Cat][0]):
                    Impute_Count=Impute_Parttable[Impute_Cat][1]
                    if(Impute_Cat!= (len(Impute_Parttable)-1)):
                        Impute_Cat+=1
                Impute_Table[count]= Impute_Count
                Actual_Table[count]=Actual_Count
                count+=1
              
            # Normalise
            Actual_Table= Actual_Table/np.sum(Actual_Table)
            Impute_Table=Impute_Table/np.sum(Impute_Table)
            
            # Compute PRD
            PRD= compute_prd(Impute_Table, Actual_Table)
            PR_Pairs.append(PRD)
            
        if(test==False):    
            plot(PR_Pairs, plotcols, Impute_Cols, LabeL=Column_Names, 
                 out_path=out_path+"PRD.png", test=True)
        if(test==True):    
            plot(PR_Pairs, plotcols, Impute_Cols, LabeL=Column_Names, 
                 out_path=out_path+"PRD_TestSet.png", test=True)
               
        
def PlotLoss(Disc_Loss, AUC, Gen_Loss,RMSE, Var_Names, out_path=None):
    # Text sizes for plots
    title_size=14
    label_size=14             
    count= len(Disc_Loss)
    x=np.arange(1,count+1)     
    
    # Number of variables to impute
    dim= (int) (np.shape(AUC)[0])
    
      # Colors 
    colors=cycle(["brown", "brown", "forestgreen","forestgreen","teal", 
                "maroon","navy", "olive"])
    
    f,ax = plt.subplots(2,1, sharex=True, figsize=[5,5])
    f.suptitle("Losses & AUC", size=title_size,y=0.99)
    f.text(0.5, 0.001, 'Epoch', ha='center', size=0.9*label_size)
    # f.text(0.0, 0.5, 'Loss', va='center', rotation='vertical', size=label_size)
    plt.tight_layout()
    ax[0].plot(x,Gen_Loss, label="Generator", color="navy")
    ax[0].set_ylabel("Generator Loss")
    ax2= ax[0].twinx()
    ax2.plot(x, Disc_Loss, label="Discriminator", color="olive")
    ax2.set_ylabel("Discriminator Loss")
    for variable in range(dim):
        # find maximum value
        AUC_array = np.array(AUC[variable])
        best_AUC_el= np.where(AUC_array==max(AUC_array))[0][0]
        best_AUC = AUC_array[best_AUC_el]
        ax[1].scatter((best_AUC_el+1), best_AUC, s=75, color=next(colors))
        ax[1].plot(x,AUC[variable], label= "PRD" + (str) (variable+1), color=next(colors),
                   markevery=best_AUC_el)
    ax[1].set_ylabel("Area under the Curve")
    plt.tight_layout()
    f.legend(loc='upper center', ncol=4, 
         bbox_to_anchor=[0.50, 0.97],frameon=False, fontsize=10)
    
    plt.close(f)     
    
    f.savefig(out_path+"Loss.png", dpi=300)
    
    return(f)    
 
# combine all plots
def Progress_Plot(rows, cols, image, start, epoch, Impute_Cols, name="GAIN", test=False):
    
    # titel font size
    title_size=16
    plt.clf() 
    plt.close("all")
    if(test==False):
        fig=plt.figure(figsize=(8, 9))
        if(len(Impute_Cols)==2):
            fig.suptitle( name+ ': Epoch: ' + str(epoch+1) + 
                     ", Elapsed time: " + str((int)(round(time.time()-start,0))) + " seconds",
                   fontsize=title_size,y=0.94)
        if(len(Impute_Cols)==1):
            fig.suptitle( name+ ': Epoch: ' + str(epoch+1) + 
                     ", Elapsed time: " + str((int)(round(time.time()-start,0))) + " seconds",
                   fontsize=title_size,y=0.96)
            
    if(test==True):
        if(len(Impute_Cols)==1):
            fig=plt.figure(figsize=(8, 9))
            fig.suptitle( name+ ': Fold: ' + str(epoch+1),
                         fontsize=title_size,y=0.73)
        if(len(Impute_Cols)==2):
            fig=plt.figure(figsize=(16, 8))
            fig.suptitle( name+ ': Fold: ' + str(epoch+1),
                         fontsize=title_size,y=1)
    for i in range(1, cols*rows +1):
        img = image[(i-1)]   
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
    
    