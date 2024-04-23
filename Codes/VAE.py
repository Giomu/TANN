#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:34:05 2024

@author: Giorgio
"""

import os  # Importing the os module to interact with the operating system
import pandas as pd  # Importing pandas library for data manipulation
import numpy as np  # Importing numpy library for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import seaborn as sns  # Importing seaborn for enhanced data visualization
sns.color_palette("tab10")  # Setting color palette for seaborn
sns.set()  # Setting seaborn aesthetic settings
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for feature scaling
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data
from tensorflow.keras.models import Model, Sequential, load_model  # Importing Keras models
from tensorflow.keras.layers import Input, Dense, Lambda  # Importing Keras layers
from tensorflow.keras.losses import mse  # Importing mean squared error loss from Keras
from tensorflow.keras import backend as K  # Importing Keras backend
from tensorflow.keras.callbacks import EarlyStopping  # Importing early stopping callback from Keras
from sklearn.decomposition import PCA  # Importing PCA for dimensionality reduction
import subprocess
 

# # Ensure the model directory exists
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH)

# path = '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Project_Network/Data/df.csv'  # Path to the CSV file

# def preprocess_data(path, scaler= None):
    
#     if scaler is None:
#         scaler = MinMaxScaler()  # Creating MinMaxScaler object if not provided
    
#     data_frame = pd.read_csv(path, sep=";", decimal=",", index_col=0)  # Reading CSV data into a DataFrame
#     #data_frame = data_frame.reset_index()
#     df = data_frame.iloc[: , :-1].apply(pd.to_numeric, errors = 'coerce').dropna()  # Extracting features and handling missing values
#     y  = data_frame.iloc[:, -1:].apply(pd.to_numeric, errors = 'coerce').dropna()  # Extracting target variable and handling missing values
#     y = np.asanyarray(y)  # Converting target variable to numpy array
    
#     X_scaled = scaler.fit_transform(df)  # Scaling features
    
#     return X_scaled, y, scaler, df  # Returning preprocessed data

def execute_Rscript(path_count, path_pheno, corrFilter=True, mincorr=0.4, test_ratio=0.3, seed=123,
                    Rscript="/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin/Rscript",
                    path_script="/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Codes/preprocessData.R"):
    command = [
        Rscript,
        "--vanilla",
        path_script,  
        path_count,
        path_pheno,
        str(corrFilter),
        str(mincorr),
        str(test_ratio),
        str(seed)
    ]
    subprocess.run(command)



def plot_pca(df, scaled = True):
    
    if scaled is False:       
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_tonorm = df[df.columns[-1]]  # Extracting target variable for normalization
        df_normalized = scaler.fit_transform(df_tonorm)  # Normalizing target variable
        df_normalized = pd.DataFrame(df_normalized)
        print(df_tonorm)
        
    else:
        df_normalized = df
    
    pca = PCA()  # Creating PCA object
    pca.fit(df_normalized.iloc[:, :-1])  # Fitting PCA on features

    explained_variance = pca.explained_variance_ratio_  # Explained variance ratio
    cumulative_variance = np.cumsum(explained_variance)  # Cumulative explained variance

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')  # Plotting cumulative explained variance
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Principal Components')
    plt.grid(True)
    plt.show()

    scores_pca = pca.transform(df_normalized.iloc[:, :-1])  # Transforming features to principal components
    
    sns.relplot(x = scores_pca[:,0], y = scores_pca[:,1], 
                hue = df_normalized[df_normalized.columns[-1]],  # Coloring points by target variable
                palette=['r', 'g'])  # Red-green color palette
    plt.grid(True)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')
    plt.show()

    

def plot_metric(history, metric):
    
    """  Plot dei risultati """
    
    train_metrics = history.history[metric]  # Extracting training metric from history
    val_metrics = history.history['val_'+metric]  # Extracting validation metric from history
    epochs = range(1, len(train_metrics) + 1)  # Generating epoch numbers
    plt.plot(epochs, train_metrics, 'bo--')  # Plotting training metric
    plt.plot(epochs, val_metrics, 'ro-')  # Plotting validation metric
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])  # Adding legend
    plt.show()

# def build_lstm_model(input_shape):

#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=input_shape))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     return model


def build_vae(input_dim, latent_dim=31):

    inputs = Input(shape=(input_dim,))  # Defining input layer
    h = Dense(256, activation='relu')(inputs)  # Adding hidden layer with ReLU activation
    h = Dense(128, activation='relu')(h)  # Adding hidden layer with ReLU activation
    h = Dense(64, activation='relu')(h)  # Adding hidden layer with ReLU activation
    z_mean = Dense(latent_dim)(h)  # Adding layer for mean of latent space
    z_log_var = Dense(latent_dim)(h)  # Adding layer for log variance of latent space

   
    def sampling(args):

        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)  # Sampling from standard normal distribution

        return z_mean + K.exp(z_log_var / 2) * epsilon  # Reparameterization trick
  
    z = Lambda(sampling)([z_mean, z_log_var])  # Sampling from latent space

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')  # Defining encoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')  # Defining input layer for decoder
    
    x = Dense(64, activation='relu')(latent_inputs)  # Adding hidden layer with ReLU activation for decoder
    x = Dense(128, activation='relu')(x)  # Adding hidden layer with ReLU activation for decoder
    x = Dense(256, activation='relu')(x)  # Adding hidden layer with ReLU activation for decoder
    
    outputs = Dense(input_dim, activation='sigmoid')(x)  # Output layer with sigmoid activation
    
    decoder = Model(latent_inputs, outputs, name='decoder')  # Defining decoder model
    outputs = decoder(encoder(inputs)[2])  # Connecting encoder and decoder

    vae = Model(inputs, outputs, name='vae_mlp')  # Defining VAE model

    reconstruction_loss = mse(inputs, outputs) * input_dim  # Reconstruction loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)  # KL divergence loss
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5  # Summing KL divergence loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)  # Total VAE loss
    vae.add_loss(vae_loss)  # Adding VAE loss to the model
    vae.compile(optimizer='adam')  # Compiling VAE model
    
    return vae  # Returning VAE model



#def main(data_frame, features, target):
def main():
    
    scaler = MinMaxScaler()  # Initializing MinMaxScaler object
    
    execute_Rscript(path_count= '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/ACC_Adrenocortical_Carcinoma/ACC_Count.csv',
                    path_pheno= '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/ACC_Adrenocortical_Carcinoma/ACC_Pheno.csv',
                    mincorr=0.2)
    
    # X_scaled, y, scaler, df = preprocess_data(path, scaler=None)
    # indices = np.arange(len(y))
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_scaled, y, indices,test_size=0.2, shuffle=True)
    
    X_train = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTrain.csv',
                          sep=";", decimal=",", index_col=0).T  # Reading training data
    X_test = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTest.csv',
                          sep=";", decimal=",", index_col=0).T  # Reading testing data
    y_train = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTrain.csv',
                          sep=";", decimal=",", index_col=0)  # Reading training labels
    y_test = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTest.csv',
                          sep=";", decimal=",", index_col=0)  # Reading testing labels
    
    # Scale X-train and X_test between 0 and 1 with min/max scaler
    X_train = scaler.fit_transform(X_train)  # Scaling training data
    X_test = scaler.fit_transform(X_test)  # Scaling testing data
    
    # Transform factor variables L/D in 1/2 in y_train and y_test
    y_train.loc[y_train["condition"] == "L", "condition"] = 1  # Encoding "L" as 1
    y_train.loc[y_train["condition"] == "D", "condition"] = 2  # Encoding "D" as 2
    y_train["condition"] = pd.to_numeric(y_train["condition"])  # Converting to numeric
    
    y_test.loc[y_test["condition"] == "L", "condition"] = 1  # Encoding "L" as 1
    y_test.loc[y_test["condition"] == "D", "condition"] = 2  # Encoding "D" as 2
    y_test["condition"] = pd.to_numeric(y_test["condition"])  # Converting to numeric
    
    # Remove ID col from y_train and y_test and transform it to numpy array
    y_train = y_train.drop("ID", axis = 1)  # Dropping ID column from training labels
    y_test = y_test.drop("ID", axis = 1)  # Dropping ID column from testing labels
    y_train = y_train.to_numpy()  # Converting training labels to numpy array
    y_test = y_test.to_numpy()  # Converting testing labels to numpy array
    
    #vae = build_vae(len(X_train.columns))
    vae = build_vae(len(X_train[0]))  # Building VAE model with input dimensionality
    vae.summary()  # Printing summary of VAE model
    
    # Train VAE
    EPOCHS = 300  # Number of epochs for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Early stopping callback
    history_vae = vae.fit(X_train, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])  # Fitting VAE model
    
    # # bottleneck model
    # bottleneck = vae.get_layer('encoder').output
    model_bottleneck = Model(inputs = vae.input, outputs=vae.get_layer('encoder').output)  # Creating bottleneck model
    
    train_bneck = model_bottleneck.predict(X_train)[2]  # Predicting bottleneck representation for training data
    train_bneck_df = pd.DataFrame(train_bneck.T)  # Converting bottleneck representation to DataFrame
    
    train_bneck_df = pd.concat([train_bneck_df, pd.DataFrame(y_train).T], ignore_index=True)  # Concatenating bottleneck representation with labels
    print(train_bneck_df)  # Printing bottleneck representation with labels
    
    
    # Test VAE
    predictions = []  # List to store predictions
    bneck_preds = []  # List to store bottleneck predictions
    failure_count = 0  # Variable to count failures
    bneck_df = pd.DataFrame()  # DataFrame to store bottleneck representation

    for idx,i in enumerate(y_test): ## Mi sa che va cambiato l'indice. Prima al posto di y_test
    # c'era indices_test

        current_X_dense = X_test[idx].reshape(1, len(X_train[0]))  # Reshaping for dense input
        actual_y = y_test[idx]  # Actual label
        
        reconstruction_error = np.mean((vae.predict(current_X_dense) - current_X_dense)**2)  # Reconstruction error

        ANOMALY_THRESHOLD = 1  # Threshold for anomaly detection
        if reconstruction_error > ANOMALY_THRESHOLD:
            failure_count += 1

        else:
            failure_count = 0

        # Save prediction results
        predictions.append({
            'index': idx,
            'actual': actual_y,
            'reconstruction_error': reconstruction_error,
            'failure_detected': reconstruction_error > ANOMALY_THRESHOLD
        }) 
        
        bneck_preds = model_bottleneck.predict(current_X_dense)[2]  # Predicting bottleneck representation       
        bneck_df[idx] = (pd.DataFrame(bneck_preds).T)  # Storing bottleneck representation
        #print(bneck_preds)
        


    # Save predictions to a CSV
    predictions_df = pd.DataFrame(predictions).set_index('index')  # Converting predictions to DataFrame
    bneck_df.loc[len(bneck_df)] = (predictions_df['actual'].str[0])## Questa è da riaggiungere corretta
    # predictions_df.to_csv('predictions.csv')
    # print(predictions_df.head())
    print(predictions_df)  # Printing predictions DataFrame
    
    plot_metric(history_vae, 'loss')  # Plotting loss metrics   
    #plot_metric(history_vae, 'accuracy')
    #print(X_train.shape)

    return predictions_df, bneck_df.T.apply(pd.to_numeric, errors='ignore'), train_bneck_df.T.apply(pd.to_numeric, errors='ignore')


# Running the main function
predictions_df, bneck_df, bneck_df_train = main()
plot_pca(bneck_df, scaled=True)  # Plotting PCA for bottleneck representation of testing data
plot_pca(bneck_df_train, scaled=True)  # Plotting PCA for bottleneck representation of training data
