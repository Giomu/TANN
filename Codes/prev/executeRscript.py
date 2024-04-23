#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:19:31 2024

@author: Giorgio
"""

import subprocess
import pandas as pd

def esegui_script_R(path_count, path_pheno, corrFilter=True, mincorr=0.4, test_ratio=0.3, seed=123):
    comando = [
        "/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript",
        "--vanilla",
        "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Codes/preprocessData.R",  
        path_count,
        path_pheno,
        str(corrFilter),
        str(mincorr),
        str(test_ratio),
        str(seed)
    ]
    subprocess.run(comando)

# Chiamata alla funzione per eseguire lo script R
esegui_script_R(path_count= '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv',
                path_pheno= '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/ACC_Adrenocortical_Carcinoma/ACC_Pheno.csv',
                mincorr=0.2)


gino = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTrain.csv',
                      sep=";", decimal=",", index_col=0).T