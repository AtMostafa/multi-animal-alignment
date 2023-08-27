import torch
import numpy as np
from rnn.simulation.networks import RNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rnn.simulation.task_data import Task_Params, Task_Dataset
from config_manager import base_configuration
from params import rnn_defs
from collections import OrderedDict
from pytorch_metric_learning.samplers import MPerClassSampler

class Runner:
    """ Object to run simulations ."""
    
    def __init__(self, config: base_configuration.BaseConfiguration, proj_dir: str, training=True) -> None:
        """ 
        Class for running simulations.

        Parameters
        ----------
        config: base_configuration.BaseConfiguration
            configuration object specifying experiment setup.
        proj_dir: str
            project directory
        training: boolean
            if you are training the model

        """
        self.PROJ_DIR = proj_dir + '/'
        self._config = config
        
        #only update config if training
        if training:
            self._config.save_configuration(folder_path=self.PROJ_DIR +self._config.outdir)

        self._outdir = self.PROJ_DIR + self._config.outdir
        self._setup(training)

    def _setup(self, training):
        """ 
        Method to set up device, data, and model.
        """
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.device = torch.device("cuda:{}".format(self._config.gpu_id))
            try:
                torch.cuda.set_device(self._config.gpu_id)
            except:
                pass
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device("cpu")

        self.task_params = Task_Params(self.PROJ_DIR + self._config.datadir)
        self.dataset = Task_Dataset(self.PROJ_DIR + self._config.datadir, training)
        self.model, self.criterion, self.optimizer = self._create_model()

    def _create_model(self):
        """ 
        Create new model, criterion, and optimizer. 

        Returns
        -------
        model: Pytorch model
            network model
        criterion: Pytorch criterion
            criterion for calculating the loss
        optimizer: Pytorch optimizer
            optimizer for training

        """
        # create model
        model = RNN(self.task_params.input_dim, self.task_params.output_dim, 
                    self._config.n1, 
                    self.task_params.dt/self._config.tau, 
                    self.dtype, noise= self._config.noise,
                    p_recurrent = self._config.p_recurrent)
        
        if self.dtype == torch.cuda.FloatTensor:
            model = model.cuda()
        
        # define loss function
        criterion = nn.MSELoss(reduction='none')

        # create optimizer
        if self._config.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self._config.lr)
        elif self._config.optimizer == 'FORCE':
            optimizer = None
        else:
            raise Exception('unknown optimizer')

        return model, criterion, optimizer

    def run_train(self):
        """ 
        Set up and train a model. 
        """

        # set up params/weights for model and optimizer
        self.model = self._create_and_initialize_weights(self.model)

        # get and save initial parameters
        params0 = self.model.save_parameters()

        #train
        if self._config.optimizer == 'FORCE':
            lc, training_trial = self.train_force()
        else:
            lc, training_trial = self.train()

        # get and save final parameters
        params1 = self.model.save_parameters()

        # save parameters
        dic = {'lc':np.array(lc), 'params0': params0, 'params1':params1}
        np.save(self._outdir+'training',dic)

        # save the final model after training
        torch.save({'epoch': training_trial, 
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model') 

        #check implementation
        import matplotlib.pyplot as plt
        plt.figure()
        print('lc shape', np.array(lc).shape)
        plt.plot(np.array(lc)[:,0,0], label = 'loss', color = 'k')
        plt.plot(np.array(lc)[:,0,1], label = 'output loss', color = 'b')
        plt.plot(np.array(lc)[:,0,2], label = 'ccareg', color = 'g')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Training trials: ' + str(training_trial))
        plt.savefig(self._outdir + "loss.png")

    def _create_and_initialize_weights(self, model):
        """ 
        Create and initialize new weights. 
        
        Parameters
        ----------
        model: Pytorch model

        Returns
        -------
        model: Pytorch model
            model with randomly initialized weights
        """

        # number of neurons
        n1 = self._config.n1

        # initialize weights
        state_dict = model.state_dict()

        ## input weights
        state_dict['rnn_l1.weight_ih_l0'] = torch.FloatTensor((np.random.rand(n1, self.task_params.input_dim)-0.5)*2.*self._config.gin) 
        state_dict['rnn_l1.bias_ih_l0'] = torch.FloatTensor(np.zeros(n1)) 

        ## recurrent weights
        state_dict['rnn_l1.weight_hh_l0'] = torch.FloatTensor(self._config.g1 / np.sqrt(n1) * np.random.randn(n1,n1)) 
        state_dict['rnn_l1.bias_hh_l0'] = torch.FloatTensor(np.zeros(n1)) 
            
        ## output/muscle weights
        state_dict['output.weight'] = torch.FloatTensor((np.random.rand(self.task_params.output_dim, n1)-0.5)*2.*self._config.gout) 
        state_dict['output.bias'] = torch.FloatTensor(np.zeros(self.task_params.output_dim))
    
        model.load_state_dict(state_dict, strict=True)
        return model

    def run_test(self, model_loaded = False):
        """ 
        Set up and test an existing model. 

        Parameters
        ----------
        epoch: int
            epoch that model was saved at during training
        model_loaded: boolean
            if the model has been loaded with parameters

        Returns
        ---------- 
        output: np array (trials x tsteps x noutputs)
            model output (motor output)  
        activity1: np array (trials x tsteps x neurons)
            activity for the first RNN

        """
        if not model_loaded:
            self.model = self._load_model(self._outdir, self.model)
        testout, testl1 = self.test_current_model()

        # save it
        output = testout.cpu().detach().numpy().transpose(1,0,2)
        activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)

        return self._config.datadir, output, activity1

    def _setup_testdata(self):
        """ 
        Set up data for testing 

        Returns
        -------
        test_stimulus: torch tensor
            stimulus input in torch format
        test_target: torch tensor
            target information in torch format

        """
        # get data
        test_data = Task_Dataset(self.PROJ_DIR + self._config.datadir, training=False)
        test_stimulus, test_target = test_data.get_stimulus_target()
        test_stimulus = test_stimulus.transpose(1,0).type(self.dtype)
        test_target = test_target.transpose(1,0).type(self.dtype)
        
        return test_stimulus, test_target

    def test_current_model(self, stim_test = None):
        """ 
        Test the current model that's loaded/being trained

        Parameters
        ----------
        stim_test: torch tensor
            stimulus for test data
        """
        self.model.eval()

        if stim_test is None:
            stim_test, _ = self._setup_testdata()

        # run model
        with torch.no_grad():
            testout, testl1 = self.model(stim_test)

        return testout, testl1

    def _load_model(self, dir, model):
        """ 
        Load state parameters in model.

        Parameters
        ----------
        dir: str
            directory where model (parameters) is saved
        model: pytorch model
            initial model
        epoch: int
            epoch that model was saved at during training

        Returns
        -------
        model: pytorch model
            model with trained parameters

        """
        try:
            temp = torch.load(dir+'model')['model_state_dict']
        except: # make sure model is loaded in available device
            temp = torch.load(dir+'model', map_location='cuda:0')['model_state_dict']
        model.load_state_dict(temp, strict = False)

        try:
            parameters = np.load(dir + 'training.npy', allow_pickle=True).item()['params1']
            model.rnn_l1_hh_mask = torch.FloatTensor(parameters['whhl1_mask']).type(self.dtype)
        except:
            pass
        
        return model

    def _load_optimizer(self, dir, optimizer, epoch = None):
        """ 
        Load state parameters in optimizer

        Parameters
        ----------
        dir: str
            directory where model is saved
        optimizer: Pytorch optimizer
            initial optimizer
        epoch: int
            epoch that model was saved at during training

        Returns
        -------
        optimizer: Pytorch optimizer
            optimizer with trained parameters

        """
        if optimizer is not None:
            epoch_str = ('_'+str(epoch)) if epoch else ""   
            try:
                temp = torch.load(dir+'model'+ epoch_str)
            except:
                temp = torch.load(dir+'model'+ epoch_str, map_location='cuda:0')
            optimizer.load_state_dict(temp['optimizer_state_dict'])

        return optimizer
    
    def cca_penalisation(self, rl1, pcas_calculated, onsets, dt):
        """ 
        Calculate CCA penalisation to discourage similar latent dynamics

        Parameters
        ----------
        rl1: torch tensor
            network activity
        pcas_calculated: torch tensor
            principal components calculated from a previously trained network
        onsets: torch tensor
            movement onsets
        dt: float
            integration time step

        Returns
        -------
        cca_pen: float
            CCA penalisation term 

        """
        import torch.nn.functional as F
        from tools.dataTools import canoncorr_torch
        from kornia.filters.kernels import get_gaussian_kernel1d

        tsteps, batch_size, n_neurons = rl1.shape
        #smooth signals
        ##setup signals: batch, neurons, tsteps
        rl1 = torch.flatten(rl1.permute(1,2,0), 0,1)
        rl1 = rl1.unsqueeze(0).permute(1,0,2) # (batchxneurons, 1, tsteps)

        ##get smoothing window
        std = 0.05
        bin_length = dt
        sigma = std/bin_length
        kernel_size =int(10*sigma)
        win = get_gaussian_kernel1d(kernel_size, sigma, force_even = True).unsqueeze(0).type(self.dtype)

        ##convolve
        rl1 = F.pad(rl1, (int(kernel_size/2), int(kernel_size/2)-1), mode='reflect')
        rl1 = F.conv1d(rl1.double(), win.double())
        rl1 = rl1.reshape(batch_size, n_neurons, tsteps)

        #restrict to interval
        rel_start = self._config.rel_start
        rel_end = self._config.rel_end
        rl1_interval = torch.zeros((batch_size, rel_end - rel_start + 1, n_neurons))
        for trial in range(rl1.shape[0]):
            cue = onsets[trial]
            rl1_interval[trial] = rl1[trial, :, (cue + rel_start):(cue + rel_end +1)].T

        #calculate PCA
        pca_dims = rnn_defs.n_components
        rl1_interval = torch.flatten(rl1_interval,0,1)
        _,_,v = torch.pca_lowrank(rl1_interval, q=pca_dims)
        pca = torch.matmul(rl1_interval, v[:, :pca_dims])

        #perform CCA on PCs
        ccs = canoncorr_torch(pca, pcas_calculated)

        #calculate penalisation on subset of CCs
        start = self._config.ccareg_components_start
        end = self._config.ccareg_components_end
        cca_pen = torch.sum(ccs[start-1:end]**2)

        return cca_pen
    
    def train_force(self):
        """ 
        Train a model. 

        Returns
        -------
        lc: list
            loss during training
        """
        torch.autograd.set_detect_anomaly(True)
        lc = [] # save loss
        training_trial = 0
        finished_training = False
        stim_test, target_test = self._setup_testdata()

        #no gradients needed in force training
        with torch.no_grad():

            # initialize the inverse correlation matrix
            P = torch.eye(self._config.n1)/self._config.lr
            P = P.type(self.dtype)

            # get inverse of output weights
            output_weights_inv = torch.linalg.pinv(self.model.output.weight.clone().detach())

            #batches include equal numbers of trials for each target
            for batch_idx, (stimulus, target) in enumerate(DataLoader(self.dataset, shuffle = True, batch_size = self._config.batch_size)): 
                stimulus, target = stimulus.transpose(1,0).type(self.dtype), target.transpose(1,0).type(self.dtype)

                train_running_loss = 0.0

                # one training step
                output, _, P, mean_total_dw = self.model.force_training(stimulus, P, output_weights_inv, target)

                #calculate loss
                loss = self.criterion(output[50:], target[50:]) # only look at time points > 50dt  
                loss_train = loss.mean() 

                #calculate testing error
                testout, _  = self.test_current_model(stim_test)

                test_error = self.criterion(testout[50:], target_test[50:]) # only look at time points > 50dt  
                test_error = test_error.mean()

                train_running_loss = [loss_train.detach().item(), test_error.detach().item(), mean_total_dw]

                toprint = OrderedDict()
                toprint['Loss'] = loss_train
                toprint['Error'] = test_error
                toprint['Weight change'] = mean_total_dw
    
                self._log(training_trial, toprint) 
                    
                lc.append([train_running_loss]) 
                
                training_trial += 1 
                if training_trial >= rnn_defs.MIN_TRAINING_TRIALS: #train for at least n trials...
                    if (np.mean(np.array(lc)[-10:,0,0]) <= rnn_defs.LOSS_THRESHOLD) or \
                     (training_trial >= rnn_defs.MAX_TRAINING_TRIALS): #...and up to m trials
                        finished_training = True
                        break

        return lc, training_trial


    def train(self):
        """ 
        Train a model. 

        Returns
        -------
        lc: list
            loss during training
        """
        torch.autograd.set_detect_anomaly(True)
        lc = [] # save loss
        training_trial = 0
        finished_training = False

        #training mode
        self.model.train()
 
        # load pcas if penalising CCs
        if self._config.ccareg:
            dict_ = np.load(self.PROJ_DIR + rnn_defs.RESULTS_FOLDER + self._config.pcas_file + '.npy', allow_pickle = True).item()
            pcas_calculated = torch.from_numpy(dict_['pca'])
            move_onsets = torch.from_numpy(dict_['move_onsets'])

        while (not finished_training):
            train_running_loss = 0.0
            
            #batches include equal numbers of trials for each target
            sampler = MPerClassSampler(self.dataset.labels,  self._config.batch_size/8, batch_size = self._config.batch_size)
            for batch_idx, (stimulus, target) in enumerate(DataLoader(self.dataset, drop_last = True, batch_size = self._config.batch_size, sampler = sampler)):                
                self.optimizer.zero_grad()

                stimulus, target = stimulus.transpose(1,0).type(self.dtype), target.transpose(1,0).type(self.dtype)
                output,rl1 = self.model(stimulus)
                
                #calculate loss
                loss = self.criterion(output[50:], target[50:]) # only look at time points > 50dt  
                loss_train = loss.mean() 

                #add regularization
                ## term 1: parameters
                reg1in = self._config.alpha1*self.model.rnn_l1.weight_ih_l0.norm(2)
                reg1rec = self._config.gamma1*self.model.rnn_l1.weight_hh_l0.norm(2)
                regout = self._config.alpha1*self.model.output.weight.norm(2)
                ## term 2: rates
                reg1act = self._config.beta1*rl1.pow(2).mean()
                
                ## penalise CCA
                if self._config.ccareg & (training_trial >= self._config.ccareg_start_trial) & (self._config.delta != 0):
                    regcca = self._config.delta * self.cca_penalisation(rl1, pcas_calculated, move_onsets, self.task_params.dt)
                else:
                    regcca = 0

                #calculate overall loss
                loss = loss_train+reg1in+reg1rec+regout+reg1act+regcca
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._config.clipgrad)
                self.optimizer.step()

                if regcca != 0:
                    train_running_loss = [loss.detach().item(), loss_train.detach().item(), regcca.detach().item()]
                else:
                    train_running_loss = [loss.detach().item(), loss_train.detach().item(), regcca]

                # values to print
                toprint = OrderedDict()
                toprint['Loss_total'] = loss
                toprint['Loss'] = loss_train
                toprint['R_l1in'] = reg1in
                toprint['R_l1rec'] = reg1rec
                toprint['R_l1rate'] = reg1act
                toprint['R_cca'] = regcca
                self._log(training_trial, toprint)          
                lc.append([train_running_loss])
                
                training_trial += 1 
                if training_trial >= rnn_defs.MIN_TRAINING_TRIALS: #train for at least n trials...
                    if (np.mean(np.array(lc)[-10:,0,0]) <= rnn_defs.LOSS_THRESHOLD) or \
                     (training_trial >= rnn_defs.MAX_TRAINING_TRIALS): #...and up to m trials
                        finished_training = True
                        break

        return lc, training_trial

    def _log(self, epoch, toprint):
        """ 
        Log and save information during training. 

        Parameters
        ----------
        epoch: int
            current epoch of training
        loss: double
            loss after regularization for current epoch
        toprint: OrderedDict()
            items to print during training

        """
        if epoch % rnn_defs.PRINT_EPOCH == 0:
            print(('Epoch=%d | '%(epoch)) +" | ".join("%s=%.4f"%(k, v) for k, v in toprint.items()))

        # save model if epoch falls on interval or was specified
        save_model = False
        if ((self._config.log_interval is not None) and (epoch % self._config.log_interval == 0)) \
            or ((self._config.log_epochs is not None) and (epoch in self._config.log_epochs)):
                save_model = True
        
        if save_model:
            torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model_'+str(epoch)) 


# if __name__ == "__main__":
