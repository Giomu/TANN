#'@description
#'function that preprocess data for TANN
#'@param path_count
#'@param path_pheno
#'@param corrFilter boolean if to perform corr_filter
#'@param split_ratio split_ratio train/test
preprocessData <- function(path_count,
                           path_pheno,
                           corrFilter = TRUE,
                           mincorr = 0.4,
                           test_ratio= 0.3,
                           seed = 123){
  
  #library(rstudioapi)
  #setwd(dirname(getActiveDocumentContext()$path))
  
  source("/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Codes/preprocessData_Functions.R")
  
  dfsImport <- dfs.import()
  df <- dfsImport[[1]]
  class <- dfsImport[[2]]
  
  print("Performing mini features trimming")
  # Keep only features with at least 10 counts in 1/3 of samples
  keep <- rowSums(df > 10) > round(ncol(df)/3)
  df <- df[keep, ]
  
  # Perform Train and Test split
  ttsplit <- trainTest.split(df = df, class = class, corrFilter = T, mincorr = mincorr,
                             ratio = test_ratio, seed = seed)
  
  write.csv2(ttsplit[[1]], "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTrain.csv")
  write.csv2(ttsplit[[2]], "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTrain.csv")
  write.csv2(ttsplit[[3]], "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTest.csv")
  write.csv2(ttsplit[[4]], "/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTest.csv")
  # return(ttsplit)
 
}

#setwd("Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Codes/")
args <- commandArgs(trailingOnly = TRUE)

# Assegna gli argomenti alle variabili
path_count <- args[1]
path_pheno <- args[2]
corrFilter <- as.logical(args[3])
mincorr <- as.numeric(args[4])
test_ratio <- as.numeric(args[5])
seed <- as.numeric(args[6])
# Esegui il preprocessing dei dati con gli argomenti passati
preprocessData(path_count, path_pheno, corrFilter, mincorr, test_ratio, seed)
