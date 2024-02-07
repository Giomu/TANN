#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:05:57 2024

@author: Giorgio
"""


import pandas as pd
import numpy as np



df = pd.read_csv("../Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv",
                 sep=";", index_col=0)

#print(df.head(10), df.tail(10))



def apply_gene_filter(df, A, B):
    
    """ This function returns a filtered df
        Input: 
            df := pandas df of raw counts with samples on rows and genes on cols
            A  := n. of samples in which gene should be > 0
            B  := each gene should have a colSum > B*n.of samples
        Output:
            filtered_df := df containing only genes that satisfied the filter"""
    
    # Calculate the total number of patients
    total_patients = len(df.index)
    
    # Apply the filter to the columns of the DataFrame
    filtered_columns = []
    for column in df.columns:
        # Count the number of rows with values greater than zero for the gene
        num_rows_with_value = (df[column] > 0).sum()
        # Calculate the sum of the column
        column_sum = df[column].sum()
        
        # Check if the gene satisfies both criteria
        if num_rows_with_value >= A and column_sum > B * total_patients:
            filtered_columns.append(column)
    
    # Select only the columns that satisfy the criteria
    filtered_df = df[filtered_columns]
    
    return filtered_df




def normalizeDf(df):
    
    """ This function takes as input a pandas df containing Raw Counts
        and returns log transformed and quantile normalized df
        This function should probably be replaced with a DeSeq one"""
        
    # log2 transform gene values
    df = np.log2(df + 1)
    
    ## Quantile Normalization with qnorm library
    import qnorm
    ds = qnorm.quantile_normalize(df.T)
      
    return (df, ds.T)
        
       
        
    
def check_quantile_normalization(df_original, df_normalized):
    
    """ Function to check if quantile normalization was successfully computed """
    
    # Check 1: Uniform or Normal Distribution
    original_describe = df_original.describe()
    normalized_describe = df_normalized.describe()
    print("Check 1: Distribution before and after normalization:")
    print("Original DataFrame description:")
    print(original_describe)
    print("\nNormalized DataFrame description:")
    print(normalized_describe)
    
    # Check 2: Preservation of Value Ratios
    ratio_check = df_original / df_normalized
    print("\nCheck 2: Preservation of value ratios (should be close to 1):")
    print(ratio_check)
    
    # Check 3: Mean and Standard Deviation
    original_mean_std = original_describe.loc[['mean', 'std']]
    normalized_mean_std = normalized_describe.loc[['mean', 'std']]
    print("\nCheck 3: Mean and standard deviation before and after normalization:")
    print("Mean and standard deviation of Original DataFrame:")
    print(original_mean_std)
    print("\nMean and standard deviation of Normalized DataFrame:")
    print(normalized_mean_std)





def calCorr(df, thr=0.9):
    
    """  This function computes Pearson correlations among pairs of genes and returns a
        list of tuples with corr > thr
    df := pandas df of filtered and normalized gene expressions
            rows: samples
            cols: genes
    thr := threshold of Pearson Correlation
    
    output:
        pairs := list of all correlation pairs
        geneDf := pandas df containing only pairs of genes with corr above thr
    
    
    """
    
    corrDf = df.corr()
    pairs = corrDf.where(np.triu(corrDf, k=1).astype(bool)).stack()
    corrList = list(pairs[pairs.abs().gt(thr)].index)
    #coeff = list(pairs[pairs.abs().gt(thr)])
    geneDf = pd.DataFrame(corrList, columns=['Gene 1', 'Gene 2'])
    #geneDf['PCC'] = coeff
    
    return pairs, geneDf















df_filtered = apply_gene_filter(df, 50, 5)
g, ds = normalizeDf(df_filtered)
#check_quantile_normalization(g, ds)

allPairsCorr, corrDf = calCorr(ds, 0.9)

corrDf.to_csv("../Results/corrPairs.txt", sep='\t', index=False)


























