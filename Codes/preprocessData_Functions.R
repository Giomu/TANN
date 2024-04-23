library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))


#' @description Import dfCount and dfPheno given their paths
#' @param pathdf relative path to dfCount
#' @param pathclin relative path to dfPheno
#' @returns df: dfCount with samples on cols and genes on rows
#' @returns class: S4 DF dfPheno matched to df with only one col of response variable named condition 
dfs.import <- function(pathdf = "../Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv",
                       pathclin = "../Data/ACC_Adrenocortical_Carcinoma/ACC_Pheno.csv"){
  
  # Import first df
  df <- read.csv2(pathdf, row.names = 1)
  df_pheno <- read.csv2(pathclin, row.names = 1)
  
  df <- as.data.frame(t(df))
  # Select from df_pheno the only col we are interested in:
  df_pheno <- df_pheno[,c(1,9)]
  # transform alive status into factor
  # L: alive
  # D: dead
  # match df_count and df_pheno
  m <- match(colnames(df), rownames(df_pheno))
  df_pheno <- df_pheno[m, ]
  
  df_pheno$patient.vital_status <- as.factor(ifelse(df_pheno$patient.vital_status == "alive", "L", "D"))
  #df_pheno <- data.frame(condition = df_pheno$patient.vital_status)
  class <- df_pheno
  colnames(class) <- c("ID", "condition")
  
  return(list(df, class))
}


#import <- dfs.import()


#' Normalize raw count dataframe using DeSeq2 and order clinical dataframe
#'
#' DeSeq_Norm returns the count df normalized and its matched clinical dataframe
#' for both train and test data normalized independently. Variance and dispersion are
#' computed on train set and then applied to the test set
#' @param df_count_train df of raw counts with genes on rows and samples on cols
#' @param df_pheno_train df of clinical observations for each sample of df_count. It
#' should have a column named 'condition' containing dependent variables info
#' @param df_count_test df of raw counts with genes on rows and samples on cols
#' @param df_pheno_test df of clinical observations for each sample of df_count. It
#' should have a column named 'condition' containing dependent variables info
#' @returns df_count with genes on cols and samples on rows normalized with DeSeq2 
#' and the ordered matched df_pheno. A pair for the train set and a pair for the test set
DeSeq_Norm <- function(df_count_train, df_pheno_train, df_count_test, df_pheno_test){
  
  ## Train
  df_count_train <- as.data.frame((df_count_train))
  # match df_count and df_pheno
  m <- match(colnames(df_count_train), rownames(df_pheno_train))
  df_pheno_matched_train <- df_pheno_train[m, ]
  
  ## Test
  df_count_test <- as.data.frame((df_count_test))
  # match df_count and df_pheno
  n <- match(colnames(df_count_test), rownames(df_pheno_test))
  df_pheno_matched_test <- df_pheno_test[n, ]
  
  library(DESeq2)
  
  ## Train
  print("... Starting Train Normalization ...")
  # Build DDS matrix for DeSeq
  ddsTrain <- DESeqDataSetFromMatrix(countData = df_count_train, 
                                     colData = df_pheno_matched_train, 
                                     design =~condition)
  # # Filter (?)
  # keep <- rowSums(counts(ddsTrain)>10) > 10
  # ddsTrain <- ddsTrain[keep, ]
  # Apply DeSeq to DDS matrix
  ddsTrain <- DESeq(ddsTrain)
  # Compute Normalized Count Data
  norm_df_train <- (assay(varianceStabilizingTransformation(ddsTrain, blind = F)))
  
  ## Test
  print("... Starting Test Normalization ...")
  ddsTest <- DESeqDataSetFromMatrix(countData = df_count_test, 
                                    colData = df_pheno_matched_test, 
                                    design = ~condition)
  # Apply filter computed on Train set
  # ddsTest <- ddsTest[keep, ]
  ddsTest <- DESeq(ddsTest)
  dispersionFunction(ddsTest) <- dispersionFunction(ddsTrain)
  norm_df_test <- (assay(varianceStabilizingTransformation(ddsTest, blind = F)))
  
  return(list(norm_df_train, df_pheno_matched_train,
              norm_df_test, df_pheno_matched_test))
  
}




#' @description Split dfCount and Class into train and test according split ratio
#' @param df dfCount as preprocessed from dfs.import
#' @param class S4 dfPheno matched and preprocessed as from dfs.import
#' @param ratio split ratio of test set
#' @param mincorr correlation threshold for soft filter
#' @returns data.train
#' @returns data.test
#' @returns classts: real test labels
trainTest.split <- function(df, class, 
                            ratio = 0.3, 
                            mincorr = 0.2, 
                            seed = 123, 
                            corrFilter = TRUE){
  
  library(feseR)
  set.seed(seed)
  
  data <- df
  nTest <- ceiling(ncol(data) * ratio)
  ind <- sample(ncol(data), nTest, FALSE)
  
  # Minimum count is set to 1 in order to prevent 0 division problem within
  # classification models.
  data.train <- as.matrix(data[ ,-ind] + 1)
  data.test <- as.matrix(data[ ,ind] + 1)
  classtr <- class[-ind, ]
  classts <- class[ind, ]
  
  # Apply DeSeq2 normalization
  # Dispersion and variance are computed on Train and subsequently applied to Test
  normalize.DeSeq <- DeSeq_Norm(df_count_train = data.train,
                                df_pheno_train = classtr,
                                df_count_test = data.test,
                                df_pheno_test = classts)
  
  # Assign Normalization results
  data.train <- as.matrix(normalize.DeSeq[[1]])
  classtr <- normalize.DeSeq[[2]]
  data.test <- as.matrix(normalize.DeSeq[[3]])
  classts <- normalize.DeSeq[[4]]
  
  if (corrFilter == TRUE){ 
    # Apply very basic correlation filter to train df
    classtr.num <- c(1,0)[classtr$condition]
    dtr <- filter.corr(scale(t(data.train), center = T, scale = T), 
                     classtr.num, mincorr = mincorr)
    dtr <- (t(dtr))
    data.train <- data.train[rownames(data.train) %in% rownames(dtr), ]
    # Apply results to test dataset
    data.test <- data.test[rownames(data.test) %in% rownames(data.train), ]
  }
  
  return(list(data.train, classtr, data.test, classts))
  
}





















