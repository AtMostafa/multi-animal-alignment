import torch
import torch.nn as nn

class RNN(nn.Module):
    '''
        Class for RNN to model neural dynamics and motor output
    '''
    def __init__(self, n_inputs, n_outputs, n_neurons, alpha, dtype, noise=None, p_recurrent = 1.0):
        super(RNN, self).__init__()

        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dtype = dtype
        self.alpha = alpha

        if noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = noise

        #recurrent layer
        self.rnn_l1 = nn.RNN(n_inputs, n_neurons, num_layers=1,
                                nonlinearity='tanh', bias=True) 
        self.rnn_l1_hh_mask = torch.rand(n_neurons, n_neurons).type(dtype) < p_recurrent
        self.rnn_l1_hh_mask.requires_grad = False

        #freeze input bias
        self.rnn_l1.bias_hh_l0.requires_grad = False

        #output layer
        self.output = nn.Linear(n_neurons, n_outputs)

    def save_parameters(self):
        '''
        Save model parameters in a dictionary
        '''
        wihl1 = self.rnn_l1.weight_ih_l0.cpu().detach().numpy().copy() #input
        whhl1 = self.rnn_l1.weight_hh_l0.cpu().detach().numpy().copy() #recurrent
        wout = self.output.weight.cpu().detach().numpy().copy() #output
        bout = self.output.bias.cpu().detach().numpy().copy()
        alpha = self.alpha
        whhl1_mask = self.rnn_l1_hh_mask.cpu().detach().numpy().copy()
        dic = {'wihl1':wihl1,'whhl1':whhl1,
               'wout':wout,'bout':bout, 'alpha':alpha,'whhl1_mask':whhl1_mask,
                }
        return dic
    
    def init_hidden(self, batch_size = None):
        '''
        Initialize hidden activity state
        '''
        if batch_size is None:
            batch_size = self.batch_size
        # needs to be small !! as rates are regularized during training -> so going small
        return ((torch.rand(1,batch_size, self.n_neurons)-0.5)*0.2).type(self.dtype)

    def forward(self, X):
        '''
        Model neural activity and output

        Parameters
        ----------
        X: torch tensor
            inputs: timesteps x batchsize x ninputs 
        '''

        tsteps, self.batch_size, _ = X.shape

        # initial activity
        hidden1 = self.init_hidden()
        x1 = hidden1
        r1 = x1.tanh() #tanh activation

        out = torch.zeros(tsteps, self.batch_size, self.n_outputs).type(self.dtype) #initial output
        hiddenl1 = torch.zeros(tsteps, self.batch_size, self.n_neurons).type(self.dtype)
        
        # update activity and output for each time step
        for j in range(tsteps): 
            x1,r1 = self.f_step(X[j],x1,r1)
            
            hiddenl1[j] = r1
            out[j] = self.output(r1)
        return out, hiddenl1

    def f_step(self,xin,x1,r1, batch_size = None):
        '''
        Single forward step in time

        Parameters
        ----------
        xin: torch tensor
            input
        x1: torch tensor
            hidden state
        r1: torch tensor
            activated hidden state (rates)
        '''
        if batch_size is None:
            batch_size = self.batch_size

        #add noise if noisy
        if self.noisy:
            nx1 = self.noise_amp*torch.randn(1,batch_size, self.n_neurons).type(self.dtype)
            x1 = x1 + self.alpha*(-x1 + r1 @ (self.rnn_l1_hh_mask * self.rnn_l1.weight_hh_l0.T) 
                                      + xin @ self.rnn_l1.weight_ih_l0.T
                                      + nx1
                                 )                                                    
        else:
            x1 = x1 + self.alpha*(-x1 + r1 @ (self.rnn_l1_hh_mask * self.rnn_l1.weight_hh_l0.T) 
                                      + xin @ self.rnn_l1.weight_ih_l0.T
                                 )

        r1 = x1.tanh() #tanh activation

        return x1,r1