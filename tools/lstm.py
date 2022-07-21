import logging
import numpy as np
import torch

torch.manual_seed(1)


def custom_r2_func(y_true, y_pred):
    "$R^2$ value as squared correlation coefficient, as per Gallego, NN 2020"

    mask = np.logical_and(np.logical_not(np.isnan(y_true)),
                          np.logical_not(np.isnan(y_pred)))
    c = np.corrcoef(y_true[mask].T, y_pred[mask].T) ** 2
    return np.diag(c[-int(c.shape[0]/2):,:int(c.shape[1]/2)])


class LSTM(torch.nn.Module):
    "The LSTM nework"
    def __init__(self, hidden_features=300, input_dims=10, output_dims = 2):
        super().__init__()
        self.hidden_features = hidden_features
        self.lstm1 = torch.nn.LSTMCell(input_dims, self.hidden_features)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_features, self.hidden_features)
        self.linear = torch.nn.Linear(self.hidden_features, output_dims)

    def forward(self, x):
        "The forward pass"
        outputs = []
        h_t = torch.zeros(1,self.hidden_features, dtype=torch.float32)
        c_t = torch.zeros(1,self.hidden_features, dtype=torch.float32)
        h_t2 = torch.zeros(1,self.hidden_features, dtype=torch.float32)
        c_t2 = torch.zeros(1,self.hidden_features, dtype=torch.float32)

        for time_step in x.split(1, dim=0):
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # initial hidden and cell states
            output = self.linear(h_t2)
            outputs.append(output)
        outputs = torch.vstack(outputs)
        return outputs

class LSTMDecoder():
    "LSTM Decoder object implemented for time-series "
    def __init__(self, input_dims=40, output_dims = 2):
        self.model = LSTM(input_dims=input_dims, output_dims=output_dims)
        self.criterion = None
        self.optimizer = None
        self.score = None

    def fit(self, x_train, y_train,
            criterion=None, optimizer=None, l_r=0.001, epochs = 10):
        "Train the decoder"
        if not criterion:
            self.criterion = torch.nn.MSELoss()
        if not optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_r)

        self.model.train()
        for _ in range(epochs):
            for j in range(x_train.shape[0]):
                inputs = torch.from_numpy(x_train[j, ...]).float()
                labels = torch.from_numpy(y_train[j, ...]).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer.step()
            logging.info(loss)

    def predict(self, x_test, y_test):
        "Predict using the decoder and save the prediction score"
        self.model.eval()
        x_test_ = torch.from_numpy(x_test).float()
        y_test_ = torch.from_numpy(y_test).float()

        test_labels = []
        test_pred = []
        for inputs, labels in zip(x_test_, y_test_):  # unravel the batches
            output = self.model(inputs)
            pred = output.detach().numpy()
            lab = labels.detach().numpy()
            test_labels.append(lab)
            test_pred.append(pred)

        pred = np.concatenate(test_pred, axis=0)
        lab = np.concatenate(test_labels, axis=0)
        cor_ = custom_r2_func(pred,lab)
        self.score = cor_

        return pred, lab
