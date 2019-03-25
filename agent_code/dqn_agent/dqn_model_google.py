# -*- coding: future_fstrings -*-
# First line is to enable f-strings in python3.5 installation on servers

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


########################################################################################################################
#####                                           Experience Replay Buffer                                           #####
########################################################################################################################

class Buffer():
    '''
    Implementation of the Experience Replay as a cyclic memory buffer.
    Allows to take a random sample of defined maximal size from the buffer for learning.
    '''

    def __init__(self, buffersize, stateshape, device=T.device('cpu')):
        self.buffersize = buffersize
        self.pos = 0
        self.fullness = 0

        self.state = T.zeros((self.buffersize, *stateshape)).to(device)
        self.action = T.zeros((self.buffersize, 1)).long().to(device)
        self.reward = T.zeros((self.buffersize, 1)).to(device)
        self.nextstate = T.zeros((self.buffersize, *stateshape)).to(device)


    def __len__(self):
        return max(self.fullness, self.pos + 1)


    def store(self, e):
        '''
        Store a new experience in the buffer memory.
        :param e: The new buffer entry to be stored {state, action, reward, next state}.
        '''
        self.pos = self.pos % self.buffersize

        self.state[self.pos] = e[0]
        self.action[self.pos] = e[1]
        self.reward[self.pos] = e[2]
        self.nextstate[self.pos] = e[3]

        if self.pos == (self.buffersize - 1):
            self.fullness = self.buffersize
        self.pos += 1


    def store_batch(self, es):
        '''
        Store an entire batch of new experiences in the buffer memory.
        :param es: The batch of new buffer entries to be stored.
        '''
        for e in es:
            self.store(e)


    def sample(self, samplesize):
        '''
        Take a subsample of defined size from the buffer memory. If samplesize is smaller than
        the current size of the buffer, a permutation is taken from the entire buffer.
        :param samplesize: Size of the sample to taken.
        :return: Sample taken from memory.
        '''
        class Batchsample():
            def __init__(self, buffer, idcs):
                self.state = buffer.state[idcs]
                self.action = buffer.action[idcs]
                self.reward = buffer.reward[idcs]
                self.nextstate = buffer.nextstate[idcs]

        s = len(self) if samplesize > len(self) else samplesize
        si = np.random.choice(len(self), s)
        batch = Batchsample(self, si)
        return batch


########################################################################################################################
#####                                                Deep Q-Network                                                #####
########################################################################################################################

class DQN(nn.Module):

    def __init__(self, agent):
        super(DQN, self).__init__()
        agent.logger.info('DQN model created ...')
        self.agent = agent
        self.updates = 0
        self.stepsilon = 0
        self.stepsilon2 = 0
        self.agent.possibleact = self.agent.s.actions


    def network_setup(self, insize=(17, 17), channels=4, eps=(0.95, 0.001), eps2=(0.001, 0.001), minibatch=64, gamma=0.98,
                      lr=0.001, lint=8, tint=10000/8, sint=1000000, aint=False):

        ### Hyperparameters ###
        totsteps = (self.agent.s.max_steps * self.agent.s.n_rounds) - self.agent.startpolicy + 1
        self.decayfct = 'exp'
        self.eps = (                                                                # Match ε-decay to n_round
            eps[0],                                                                 # Starting value
            eps[1],                                                                 # Terminal value
            (eps[1]-eps[0])/totsteps,                                               # Linear decay slope
            np.log(eps[1]/eps[0])/totsteps)                                         # Exponential decay constant
        self.eps2 = (                                                               # Match ε-decay to n_round
            eps2[0],                                                                # Starting value
            eps2[1],                                                                # Terminal value
            (eps2[1] - eps2[0]) / totsteps,                                         # Linear decay slope
            np.log(eps2[1] / eps2[0]) / totsteps)                                   # Exponential decay constant
        self.gamma = gamma                                                          # Discount factor
        self.learningrate = lr                                                      # Learning rate (alpha)
        self.batchsize = minibatch                                                  # Batch size for learning
        self.learninginterval = lint                                                # Learning interval
        self.targetinterval = tint                                                  # Interval for updating target DQN

        ### Intervals ###

        self.saveinterval = sint                                                    # Saving full model, params, buffer
        self.analysisinterval = aint                                                # Saving data for later analysis


        ## Network architecture ###

        def conv_out(size, ks=1, s=1, p=0, d=1):
            out = np.array([0, 0])
            if type(ks) == type(int()):
                ks = np.array([ks, ks])
            if type(s) == type(int()):
                s = np.array([s, s])
            if type(p) == type(int()):
                p = np.array([p, p])
            if type(d) == type(int()):
                d = np.array([d, d])
            out[0] = (size[0] + 2 * p[0] - d[0] * (ks[0] - 1) - 1) / s[0] + 1
            out[1] = (size[1] + 2 * p[1] - d[1] * (ks[1] - 1) - 1) / s[1] + 1
            return out

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1)
        self.activ1 = nn.functional.relu
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.activ2 = nn.functional.relu
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.activ3 = nn.functional.relu
        self.fc4 = nn.Linear(64*np.prod(conv_out(conv_out(conv_out(insize, ks=3, s=1, p=0), ks=2, s=1), ks=2, s=2, p=0)), 256)
        self.activ4 = nn.functional.relu
        self.fc5 = nn.Linear(256, len(self.agent.possibleact))
        self.agent.logger.info('DQN is set up.')
        self.agent.logger.debug(self)

        ### Optimizer ###

        self.learningrate = lr
        #self.optimizer = optim.Adam(self.parameters(), lr=self.learningrate, weight_decay=0.0001)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learningrate, momentum=0.95, eps=0.01, weight_decay=0.0001)

        ### Loss function ###
        self.loss = nn.functional.smooth_l1_loss
        #self.loss = nn.functional.mse_loss

        self.info = {
            'lr': self.learningrate,
            'lint': self.learninginterval,
            'tint': self.targetinterval,
            'eps': self.eps,
            'eps2': self.eps2,
            'batchsize': self.batchsize,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'architecture': str(self)
        }

    def set_weights(self, random=True, file=False):

        def random_weights(model):
            if type(model) == nn.Conv2d or type(model) == nn.Linear:
                nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
                model.bias.data.fill_(0.01)
        if random:
            # Set initial weights randomly
            self.apply(random_weights)
        elif file:
            self.agent.logger.info('No weights found!')
            # load weights of trained model from file and initialize ...

        self.agent.logger.info('Model initialized.')


    def forward(self, input, silent=True):
        '''
        Forward calculation of neural activations ...
        :param input: Input tensor.
        :return: Output tensor (q-value for all possible actions).
        '''
        def noprint(*args):
            [print(*args) if not silent else None]

        noprint('Input:', input.shape)
        interm = self.activ1(self.conv1(input))
        noprint('Conv1, ReLU:', interm.shape)
        interm = self.activ2(self.conv2(interm))
        noprint('Conv2, ReLU:', interm.shape)
        interm = self.activ3(self.conv3(interm))
        noprint('Conv4, ReLU:', interm.shape)
        interm = interm.view(interm.size(0), -1)
        noprint('".view(.size(0), -1)":', interm.shape)
        interm = self.activ4(self.fc4(interm))
        noprint('FC4, ReLU:', interm.shape)
        output = self.fc5(interm)
        noprint('Output:', output.shape)

        return output


    def explore(self):
        '''
        Implementation of an ε-greedy policy.
        Choose whether to explore or exploit in this step.
        :param decay: Type of decay for the ε-threshold {'lin', 'exp'}
        :return: True if decided to explore.
        '''
        t = self.agent.trainingstep - self.agent.startpolicy + 1
        if self.agent.trainingstep < self.agent.startpolicy:
            self.stepsilon = 0.30
            if np.random.random() > self.stepsilon:
                choice = 'random'
            else:
                choice = 'rolemodel'
        else:
            start, end, slope, exponent = self.eps
            start2, end2, slope2, exponent2 = self.eps2
            if self.decayfct == 'exp':
                self.stepsilon = end + (start - end) * np.exp(exponent * t)
                self.stepsilon2 = end2 + (start2 - end2) * np.exp(exponent2 * t)
            else:
                if not self.decayfct == 'lin':
                    print(f'Decay function: {self.decayfct} not known! Choosing linear decay instead.')
                self.stepsilon = start + slope * t
                self.stepsilon2 = start2 + slope2 * t

            if np.random.random() > self.stepsilon:
                choice = 'policy'
            else:
                if np.random.random() > self.stepsilon2:
                    choice = 'random'
                else:
                    choice = 'rolemodel'
        return choice