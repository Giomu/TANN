#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:25:44 2024

@author: Giorgio
"""
# import os

# os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/'
# os.environ['PATH'] += ':/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin'

# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri


# def run_preprocess_data(path_count, path_pheno, corrFilter=True, split_ratio=0.7):
#     # Carica lo script R
#     robjects.r['source']('preprocessData.R')
    
#     # Ottieni la funzione preprocessData dallo script R
#     preprocessData = robjects.globalenv['preprocessData']
    
#     # Esegui la funzione preprocessData con gli argomenti specificati
#     ttsplit = preprocessData(path_count, path_pheno, corrFilter, split_ratio)
    
#     # Converti l'output in un dataframe pandas
#     pandas2ri.activate()
#     df = pandas2ri.ri2py(ttsplit)
    
#     return df

# # Esegui la funzione con i percorsi dei file e altri argomenti necessari
# path_count = "../Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv"
# path_pheno = "../Data/ACC_Adrenocortical_Carcinoma/ACC_Pheno.csv"
# output_df = run_preprocess_data(path_count, path_pheno)
# print(output_df)


import subprocess
#import json

# Definisci il percorso del file Rscript
rscript_path = "/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript"

# Definisci il percorso del file R con la funzione getConsensus.list
r_file_path = "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Codes/preprocessData.R"

# Esegui il comando Rscript e cattura l'output
result = subprocess.run([rscript_path, r_file_path], stdout=subprocess.PIPE)
