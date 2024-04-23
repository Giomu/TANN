#'@description
#'function that preprocess data for TANN
#'@param path_count
#'@param path_pheno
#'@param corrFilter boolean if to perform corr_filter
#'@param split_ratio split_ratio train/test
preprocessData <- function(path_count,
                           path_pheno,
                           corrFilter = TRUE,
                           mincorr = 0.2,
                           test_ratio= 0.3,
                           seed = 123){
  
  library(rstudioapi)
  setwd(dirname(getActiveDocumentContext()$path))
  
  source("preprocessData_Functions.R")
  
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
  
  write.csv2(ttsplit[[1]], "../Data/countTrain.csv")
  write.csv2(ttsplit[[2]], "../Data/phenoTrain.csv")
  write.csv2(ttsplit[[3]], "../Data/countTest.csv")
  write.csv2(ttsplit[[4]], "../Data/phenoTest.csv")
  # return(ttsplit)
 
}
# cosetto <- preprocessData()
