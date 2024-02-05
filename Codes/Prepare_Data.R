setwd("Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/GitHub/")

library(dplyr)

########## ---------- df_ACC ---------- ##########
df_acc <- read.delim("Data/ACC_Adrenocortical_Carcinoma/ACC.uncv2.mRNAseq_raw_counts.txt",
                     sep = "\t", dec = ",", row.names = 1)

# Transform as.numeric and round non-integer values
rnames <- rownames(df_acc)
df_acc <- as.data.frame(lapply(df_acc, as.numeric))
df_acc <- as.data.frame(lapply(df_acc, round))
rownames(df_acc) <- rnames
# Transpose df
df_acc <- as.data.frame(t(df_acc))
# Change colnames of df to match official gene symbol
colnames(df_acc) <-  sub("\\|.*", "", colnames(df_acc))
df_acc <- df_acc[, -c(1:29)]
# Change ID rownmaes
rownames(df_acc) <- sub(".01", "", rownames(df_acc), fixed = T)


## df_acc with clinical informations
acc_info <- read.delim("Data/ACC_Adrenocortical_Carcinoma/ACC.clin.merged.txt",
                       sep = "\t", dec = ",", row.names = 1)

# remove rows with more then 5 NA and keep only informative ones
na.count <- rowSums(is.na(acc_info))
acc_info <- acc_info[na.count<=5, ]
acc_info <- as.data.frame(t(acc_info))
acc_info <- acc_info[, c(9,12,13,22,23,28,31,35,41)]
# transform ID to match with df_acc
acc_info$patient.bcr_patient_barcode <- toupper(acc_info$patient.bcr_patient_barcode)
acc_info$patient.bcr_patient_barcode <- gsub("-", ".", acc_info$patient.bcr_patient_barcode, fixed = T)
rownames(acc_info) <- acc_info$patient.bcr_patient_barcode

sum(rownames(acc_info) %in% rownames(df_acc))

write.csv2(df_acc, "Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv")
write.csv2(acc_info, "Data/ACC_Adrenocortical_Carcinoma/ACC_Pheno.csv")
########## ---------- ########## ---------- ##########



########## ---------- df_BRCA ---------- ##########
df_brca <- read.delim("Data/BRCA_Breast_invasive_carcinoma/BRCA.uncv2.mRNAseq_raw_counts.txt",
                     sep = "\t", dec = ",", row.names = 1)

# Transform as.numeric and round non-integer values
rnames <- rownames(df_brca)
df_brca <- as.data.frame(lapply(df_brca, as.numeric))
df_brca <- as.data.frame(lapply(df_brca, round))
rownames(df_brca) <- rnames
# Transpose df
df_brca <- as.data.frame(t(df_brca))
# Change colnames of df to match official gene symbol
colnames(df_brca) <-  sub("\\|.*", "", colnames(df_brca))
df_brca <- df_brca[, -c(1:29)]
# Change ID rownmaes
rownames(df_brca) <- sub(".01", "", rownames(df_brca), fixed = T)


## df_acc with clinical informations
brca_info <- read.delim("Data/BRCA_Breast_invasive_carcinoma/BRCA.clin.merged.txt",
                       sep = "\t", dec = ",", row.names = 1)

# remove rows with more then 5 NA and keep only informative ones
na.count <- rowSums(is.na(brca_info))
brca_info <- brca_info[na.count<=5, ]
brca_info <- as.data.frame(t(brca_info))
brca_info <- brca_info[, c(11,32)]
# transform ID to match with df_acc
brca_info$patient.bcr_patient_barcode <- toupper(brca_info$patient.bcr_patient_barcode)
brca_info$patient.bcr_patient_barcode <- gsub("-", ".", brca_info$patient.bcr_patient_barcode, fixed = T)
rownames(brca_info) <- brca_info$patient.bcr_patient_barcode

sum(rownames(brca_info) %in% rownames(df_brca))

write.csv2(df_brca, "Data/BRCA_Breast_invasive_carcinoma/BRCA_Count.csv")
write.csv2(brca_info, "Data/BRCA_Breast_invasive_carcinoma/BRCA_Pheno.csv")
########## ---------- ########## ---------- ##########



########## ---------- df_COAD ---------- ##########
df_coad <- read.delim("Data/COAD_Colon_adenocarcinoma/COAD.uncv2.mRNAseq_raw_counts.txt",
                      sep = "\t", dec = ",", row.names = 1)

# Transform as.numeric and round non-integer values
rnames <- rownames(df_coad)
df_coad <- as.data.frame(lapply(df_coad, as.numeric))
df_coad <- as.data.frame(lapply(df_coad, round))
rownames(df_coad) <- rnames
# Transpose df
df_coad <- as.data.frame(t(df_coad))
# Change colnames of df to match official gene symbol
colnames(df_coad) <-  sub("\\|.*", "", colnames(df_coad))
df_coad <- df_coad[, -c(1:29)]
# Change ID rownmaes
rownames(df_coad) <- sub(".01", "", rownames(df_coad), fixed = T)


## df_acc with clinical informations
coad_info <- read.delim("Data/COAD_Colon_adenocarcinoma/COAD.clin.merged.txt",
                        sep = "\t", dec = ",", row.names = 1)

# remove rows with more then 5 NA and keep only informative ones
na.count <- rowSums(is.na(coad_info))
coad_info <- coad_info[na.count<=5, ]
coad_info <- as.data.frame(t(coad_info))
coad_info <- coad_info[, c(10,28)]
# transform ID to match with df_acc
coad_info$patient.bcr_patient_barcode <- toupper(coad_info$patient.bcr_patient_barcode)
coad_info$patient.bcr_patient_barcode <- gsub("-", ".", coad_info$patient.bcr_patient_barcode, fixed = T)
rownames(coad_info) <- coad_info$patient.bcr_patient_barcode

sum(rownames(coad_info) %in% rownames(df_coad))

write.csv2(df_coad, "Data/COAD_Colon_adenocarcinoma/COAD_Count.csv")
write.csv2(coad_info, "Data/COAD_Colon_adenocarcinoma/COAD_Pheno.csv")
########## ---------- ########## ---------- ##########



########## ---------- df_KICH ---------- ##########
df_kich <- read.delim("Data/KICH_Kidney_chromophobe/KICH.uncv2.mRNAseq_raw_counts.txt",
                      sep = "\t", dec = ",", row.names = 1)

# Transform as.numeric and round non-integer values
rnames <- rownames(df_kich)
df_kich <- as.data.frame(lapply(df_kich, as.numeric))
df_kich <- as.data.frame(lapply(df_kich, round))
rownames(df_kich) <- rnames
# Transpose df
df_kich <- as.data.frame(t(df_kich))
# Change colnames of df to match official gene symbol
colnames(df_kich) <-  sub("\\|.*", "", colnames(df_kich))
df_kich <- df_kich[, -c(1:29)]
# Change ID rownmaes
rownames(df_kich) <- sub(".01", "", rownames(df_kich), fixed = T)


## df_acc with clinical informations
kich_info <- read.delim("Data/KICH_Kidney_chromophobe/KICH.clin.merged.txt",
                        sep = "\t", dec = ",", row.names = 1)

# remove rows with more then 5 NA and keep only informative ones
na.count <- rowSums(is.na(kich_info))
kich_info <- kich_info[na.count<=5, ]
kich_info <- as.data.frame(t(kich_info))
kich_info <- kich_info[, c(10,27,40)]
# transform ID to match with df_acc
kich_info$patient.bcr_patient_barcode <- toupper(kich_info$patient.bcr_patient_barcode)
kich_info$patient.bcr_patient_barcode <- gsub("-", ".", kich_info$patient.bcr_patient_barcode, fixed = T)
rownames(kich_info) <- kich_info$patient.bcr_patient_barcode

sum(rownames(kich_info) %in% rownames(df_kich))

write.csv2(df_kich, "Data/KICH_Kidney_chromophobe/KICH_Count.csv")
write.csv2(kich_info, "Data/KICH_Kidney_chromophobe/KICH_Pheno.csv")
########## ---------- ########## ---------- ##########



########## ---------- df_KIRC ---------- ##########
df_kirc <- read.delim("Data/KIRC_Kidney_renal_clear_cell_carcinoma/KIRC.uncv2.mRNAseq_raw_counts.txt",
                      sep = "\t", dec = ",", row.names = 1)

# Transform as.numeric and round non-integer values
rnames <- rownames(df_kirc)
df_kirc <- as.data.frame(lapply(df_kirc, as.numeric))
df_kirc <- as.data.frame(lapply(df_kirc, round))
rownames(df_kirc) <- rnames
# Transpose df
df_kirc <- as.data.frame(t(df_kirc))
# Change colnames of df to match official gene symbol
colnames(df_kirc) <-  sub("\\|.*", "", colnames(df_kirc))
df_kirc <- df_kirc[, -c(1:29)]
# Change ID rownmaes
rownames(df_kirc) <- sub(".01", "", rownames(df_kirc), fixed = T)


## df_acc with clinical informations
kirc_info <- read.delim("Data/KIRC_Kidney_renal_clear_cell_carcinoma/KIRC.clin.merged.txt",
                        sep = "\t", dec = ",", row.names = 1)

# remove rows with more then 5 NA and keep only informative ones
na.count <- rowSums(is.na(kirc_info))
kirc_info <- kirc_info[na.count<=5, ]
kirc_info <- as.data.frame(t(kirc_info))
kirc_info <- kirc_info[, c(10,32)]
# transform ID to match with df_acc
kirc_info$patient.bcr_patient_barcode <- toupper(kirc_info$patient.bcr_patient_barcode)
kirc_info$patient.bcr_patient_barcode <- gsub("-", ".", kirc_info$patient.bcr_patient_barcode, fixed = T)
rownames(kirc_info) <- kirc_info$patient.bcr_patient_barcode

sum(rownames(kirc_info) %in% rownames(df_kirc))

write.csv2(df_kirc, "Data/KIRC_Kidney_renal_clear_cell_carcinoma/KIRC_Count.csv")
write.csv2(kirc_info, "Data/KIRC_Kidney_renal_clear_cell_carcinoma/KIRC_Pheno.csv")
########## ---------- ########## ---------- ##########

































