import logging
import numpy as np
import torch
from tqdm import tqdm

rng = np.random.default_rng(np.random.SeedSequence(12345))

def custom_r2_func(y_true, y_pred):
    """
    $R^2$ value as Coefficient of determination
    Both inputs of shape: time x features
    """
    SS_res = np.nansum((y_true - y_pred)**2, axis=0)
    SS_tot = np.nansum((y_true - y_true.mean(axis=0))**2, axis=0)
    return 1 - (SS_res / SS_tot)


class LSTM(torch.nn.Module):
    "The LSTM network"
    def __init__(self, dtype, hidden_features=300, input_dims=10, output_dims = 2):
        super().__init__()
        torch.manual_seed(2147483647)
        self.hidden_features = hidden_features
        self.lstm1 = torch.nn.LSTMCell(input_dims, self.hidden_features)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_features, self.hidden_features)
        self.linear = torch.nn.Linear(self.hidden_features, output_dims)
        self.dtype = dtype

    def forward(self, x_in):
        "The forward pass"
        h_t = torch.zeros(1,self.hidden_features).type(self.dtype)
        c_t = torch.zeros(1,self.hidden_features).type(self.dtype)
        h_t2 = torch.zeros(1,self.hidden_features).type(self.dtype)
        c_t2 = torch.zeros(1,self.hidden_features).type(self.dtype)
        outputs = torch.zeros(x_in.shape[0], self.linear.out_features).type(self.dtype)

        for i, time_step in enumerate(x_in.split(1, dim=0)):
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # initial hidden and cell states
            outputs[i] = self.linear(h_t2)
        return outputs

class LSTMDecoder():
    "LSTM Decoder object implemented for time-series "
    def __init__(self, input_dims=40, output_dims = 2):
        torch.manual_seed(2147483647)
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.device = torch.device("cuda:{}".format(0))
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device("cpu")

        self.model = LSTM(self.dtype, input_dims=input_dims, output_dims=output_dims)
        if self.model.dtype == torch.cuda.FloatTensor:
            self.model = self.model.cuda()
        self.criterion = None
        self.optimizer = None
        self.score = None
        self._fitted = False

    def fit(self, x_train, y_train,
            criterion=None, optimizer=None, l_r=0.001, epochs = 10):
        """
        Train the decoder
        `x_train` and `y_train` of shape: trial x time x [component or output-variable]
        """
        if not criterion:
            self.criterion = torch.nn.MSELoss()
        if not optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_r)

        self.model.train()

        #shuffle trials
        shuffled_ind = rng.permutation(x_train.shape[0])

        for _ in tqdm(range(epochs)):
            for ind in shuffled_ind:
                self.optimizer.zero_grad()

                inputs = x_train[ind, ...]
                labels = y_train[ind, ...]
                mask = torch.logical_not(torch.isnan(labels[:,0]))
                inputs = inputs[mask,:]
                labels = labels[mask,:]

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer.step()

        self._fitted = True
        self.score = None

    def predict(self, x_test, y_test):
        """
        Predict using the decoder and save the prediction score
        `x_test` and `y_test` of shape: trial x time x [component or output-variable]
        """
        if not self._fitted:
            logging.error("Model hasn't trained yet. Run the `fit()` method first.")

        self.model.eval()
        x_test_ = torch.from_numpy(x_test).type(self.dtype)
        y_test_ = torch.from_numpy(y_test).type(self.dtype)

        test_labels = []
        test_pred = []
        for inputs, labels in zip(x_test_, y_test_):  # unravel the batches
            output = self.model(inputs)
            test_pred.append(output.cpu().detach().numpy())
            test_labels.append(labels.cpu().detach().numpy())

        pred = np.concatenate(test_pred, axis=0)
        label = np.concatenate(test_labels, axis=0)
        self.score = custom_r2_func(pred,label)

        return pred, label


if __name__ == "__main__":
    x = np.random.rand(50, 1000, 40)
    y = np.sin(x**2+2)*x**2
    y[:2,:10,:] = np.nan
    y = y[..., :3]
    model = LSTMDecoder(40, 3)
    model.fit(x,y)
    pr, lab = model.predict(x,y)
    print()
