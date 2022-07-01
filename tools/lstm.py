import numpy as np
import pandas as pd
import pyaldata as pyal
from sklearn.decomposition import PCA
from typing import Callable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)



class LstmDecoder(nn.Module):
    """
    LSTM decoder for timeseries
    based on: https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
    """
    def __init__(self, input_size, hidden_dim, tagset_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, tagset_size)
        
    def forward(self, y, future_preds=0):
        "The `forward` method"
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs




class LstmDecoder(nn.Module):
    "LSTM decoder"
    def __init__(self, input_size, hidden_dim, tagset_size=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, signal):
        "The `forward` method"

        lstm_out, _ = self.lstm(signal.transpose(-1,1,1))
        tag_space = self.hidden2tag(lstm_out.view(len(signal), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    





class LSTMRegression:

    """
    Class for the gated recurrent unit (GRU) decoder
    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer
    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    num_epochs: integer, optional, default 10
        Number of epochs used for training
    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
        self.units=units
        self.dropout=dropout
        self.num_epochs=num_epochs
        self.verbose=verbose
        self.model = nn.LSTM(input_size=3, hidden_size=3, dropout=dropout)

    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder
        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),
            dropout_W=self.dropout,dropout_U=self.dropout))
            #Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),
            dropout=self.dropout,recurrent_dropout=self.dropout))
            #Within recurrent layer, include dropout
        if self.dropout!=0:
            model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        #Set loss function and optimizer
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder
        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.
        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted