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

    def __init__(self, buffersize, stateshape):
        self.buffersize = buffersize
        self.pos = 0
        self.fullness = 0

        self.state = T.zeros((self.buffersize, *stateshape))
        self.action = T.zeros((self.buffersize, 1)).long()
        self.reward = T.zeros((self.buffersize, 1))
        self.nextstate = T.zeros((self.buffersize, *stateshape))


    def __len__(self):
        return max(self.fullness, self.pos)


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

        if self.pos == 1000:
            self.fullness = 1000
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
        self.stepsilon2 =0
        self.agent.possibleact = self.agent.s.actions


    def network_setup(self, insize=3, channels=2, eps=(0.95, 0.05), eps2=(0.95, 0.001), minibatch=32, gamma=0.95,
                      lr=0.001, lint=8, tint=1000, sint=50000, aint=False):

        ### Hyperparameters ###
        totsteps = (self.agent.s.max_steps * self.agent.s.n_rounds) - self.agent.startlearning + 1
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

        def conv_out(insize, ks=2, s=1):
            return (insize - (ks - 1) - 1) // s + 1

        self.layer1 = nn.Conv2d(channels, 32, kernel_size=1, stride=1)
        self.activ1 = nn.functional.leaky_relu
        self.layer2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.convdrop = nn.functional.dropout2d
        self.activ2 = nn.functional.leaky_relu
        self.layer3 = nn.Linear(64*conv_out(conv_out(insize, ks=1, s=1),ks=1, s=1)**2, 1024)
        self.activ3 = nn.functional.leaky_relu
        self.layer4 = nn.Linear(1024, 512)
        self.activ4 = nn.functional.leaky_relu
        self.fc5 = nn.Linear(512, len(self.agent.possibleact))
        self.agent.logger.info('DQN is set up.')
        self.agent.logger.debug(self)

        ### Optimizer ###

        self.learningrate = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningrate, weight_decay=0.0001)

        # RMSprop???

        ### Loss function ###
        #self.loss = nn.functional.smooth_l1_loss
        self.loss = nn.functional.mse_loss


    def set_weights(self, random=True, file=False):

        def random_weights(model):
            if type(model) == nn.Conv2d or type(model) == nn.Linear:
                nn.init.kaiming_uniform_(model.weight, nonlinearity='leaky_relu')
                model.bias.data.fill_(0.01)
        if random:
            # Set initial weights randomly
            self.apply(random_weights)
        elif file:
            self.agent.logger.info('No weights found!')
            # load weights of trained model from file and initialize ...

        self.agent.logger.info('Model initialized.')


    def forward(self, input):
        '''
        Forward calculation of neural activations ...
        :param input: Input tensor.
        :return: Output tensor (q-value for all possible actions).
        '''
        interm = self.activ1(self.layer1(input))
        interm = self.activ2(self.convdrop(self.layer2(interm), p=0.05, training=self.agent.training))
        interm = self.activ3(self.layer3(interm))
        interm = self.activ4(self.layer4(interm.view(interm.size(0), -1)))
        output = self.fc5(interm)
        return output


    def explore(self):
        '''
        Implementation of an ε-greedy policy.
        Choose whether to explore or exploit in this step.
        :param decay: Type of decay for the ε-threshold {'lin', 'exp'}
        :return: True if decided to explore.
        '''
        t = self.agent.trainingstep - self.agent.startlearning + 1
        if self.agent.trainingstep < self.agent.startlearning:
            self.stepsilon = 0.5
            if np.random.random() < self.stepsilon:
                choice = 'random'
            else:
                choice = 'policy'
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