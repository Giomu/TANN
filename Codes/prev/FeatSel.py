#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:06:37 2024

@author: Giorgio
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("tab10")
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA

 

# # Ensure the model directory exists
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH)

path = '/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Project_Network/Data/df.csv'

def preprocess_data(path, scaler= None):
    
    if scaler is None:
        scaler = MinMaxScaler()
    
    data_frame = pd.read_csv(path, sep=";", decimal=",", index_col=0) 
    #data_frame = data_frame.reset_index()
    df = data_frame.iloc[: , :-1].apply(pd.to_numeric, errors = 'coerce').dropna()
    y  = data_frame.iloc[:, -1:].apply(pd.to_numeric, errors = 'coerce').dropna()
    y = np.asanyarray(y)
    
    X_scaled = scaler.fit_transform(df)
    
    
    return X_scaled, y, scaler, df


def plot_pca(df, scaled = True):
    
    if scaled is False:       
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_tonorm = df[df.columns[-1]]
        df_normalized = scaler.fit_transform(df_tonorm)
        df_normalized = pd.DataFrame(df_normalized)
        print(df_tonorm)
        
    else:
        df_normalized = df
    
    pca = PCA()
    pca.fit(df_normalized.iloc[:, :-1])

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Principal Components')
    plt.grid(True)
    plt.show()

    scores_pca = pca.transform(df_normalized.iloc[:, :-1])  
    
    sns.relplot(x = scores_pca[:,0], y = scores_pca[:,1], 
                hue = df_normalized[df_normalized.columns[-1]],
                palette=['r', 'g'])
    plt.grid(True)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')
    plt.show()

    

def plot_metric(history, metric):
    
    """  Plot dei risultati """
    
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

# def build_lstm_model(input_shape):

#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=input_shape))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     return model


def build_vae(input_dim, latent_dim=31):

    inputs = Input(shape=(input_dim,))
    h = Dense(256, activation='relu')(inputs)
    h = Dense(128, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

   
    def sampling(args):

        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)

        return z_mean + K.exp(z_log_var / 2) * epsilon
  
    z = Lambda(sampling)([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    
    x = Dense(64, activation='relu')(latent_inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    outputs = Dense(input_dim, activation='sigmoid')(x)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])

    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs) * input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    

    return vae



#def main(data_frame, features, target):
def main(path):
    
    scaler = MinMaxScaler()
    
    # X_scaled, y, scaler, df = preprocess_data(path, scaler=None)
    # indices = np.arange(len(y))
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_scaled, y, indices,test_size=0.2, shuffle=True)
    
    X_train = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTrain.csv',
                          sep=";", decimal=",", index_col=0).T
    X_test = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/countTest.csv',
                          sep=";", decimal=",", index_col=0).T
    y_train = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTrain.csv',
                          sep=";", decimal=",", index_col=0)
    y_test = pd.read_csv('/Users/giorgiomontesi/Desktop/Universita_di_Siena/A_PhD_Project/Biomarker_Prediction/TANN/Data/phenoTest.csv',
                          sep=";", decimal=",", index_col=0)
    
    # Scale X-train and X_test between 0 and 1 with min/max scaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # Transform factor variables L/D in 1/2 in y_train and y_test
    y_train.loc[y_train["condition"] == "L", "condition"] = 1
    y_train.loc[y_train["condition"] == "D", "condition"] = 2
    y_train["condition"] = pd.to_numeric(y_train["condition"])
    
    y_test.loc[y_test["condition"] == "L", "condition"] = 1
    y_test.loc[y_test["condition"] == "D", "condition"] = 2
    y_test["condition"] = pd.to_numeric(y_test["condition"])
    
    # Remove ID col from y_train and y_test and transform it to numpy array
    y_train = y_train.drop("ID", axis = 1)
    y_test = y_test.drop("ID", axis = 1)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    #vae = build_vae(len(X_train.columns))
    vae = build_vae(len(X_train[0]))
    vae.summary()
    
    # Train VAE
    EPOCHS = 300
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history_vae = vae.fit(X_train, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])
    
    # # bottleneck model
    # bottleneck = vae.get_layer('encoder').output
    model_bottleneck = Model(inputs = vae.input, outputs=vae.get_layer('encoder').output)
    
    train_bneck = model_bottleneck.predict(X_train)[2]
    train_bneck_df = pd.DataFrame(train_bneck.T)
    
    train_bneck_df = pd.concat([train_bneck_df, pd.DataFrame(y_train).T], ignore_index=True)
    print(train_bneck_df)
    
    
    # Test VAE
    predictions = []
    bneck_preds = []
    failure_count = 0
    bneck_df = pd.DataFrame()

    for idx,i in enumerate(y_test): ## Mi sa che va cambiato l'indice. Prima al posto di y_test
    # c'era indices_test

        current_X_dense = X_test[idx].reshape(1, len(X_train[0]))  # Reshape for dense input
        actual_y = y_test[idx]
        
        reconstruction_error = np.mean((vae.predict(current_X_dense) - current_X_dense)**2)  # VAE reconstruction

        ANOMALY_THRESHOLD = 1
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
        
        bneck_preds = model_bottleneck.predict(current_X_dense)[2]        
        bneck_df[idx] = (pd.DataFrame(bneck_preds).T)
        #print(bneck_preds)
        


    # Save predictions to a CSV
    predictions_df = pd.DataFrame(predictions).set_index('index')
    bneck_df.loc[len(bneck_df)] = (predictions_df['actual'].str[0])## Questa è da riaggiungere corretta
    # predictions_df.to_csv('predictions.csv')
    # print(predictions_df.head())
    print(predictions_df)
    
    plot_metric(history_vae, 'loss')    
    #plot_metric(history_vae, 'accuracy')

    return predictions_df, bneck_df.T.apply(pd.to_numeric, errors='ignore'), train_bneck_df.T.apply(pd.to_numeric, errors='ignore')




predictions_df, bneck_df, bneck_df_train = main(path)
plot_pca(bneck_df, scaled=True)
plot_pca(bneck_df_train, scaled=True)
