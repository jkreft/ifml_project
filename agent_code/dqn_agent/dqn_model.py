
import torch.nn as nn
import numpy as np
from collections import deque

class Buffer():
    '''
    Implementation of the Experience Replay as a cyclic memory buffer.
    Allows to take a random sample of defined maximal size from the buffer for learning.
    '''

    def __init__(self, buffersize):
        self.buffersize = buffersize
        self.state = deque(maxlen=buffersize)
        self.action = deque(maxlen=buffersize)
        self.reward = deque(maxlen=buffersize)
        self.nextstate = deque(maxlen=buffersize)

    def __len__(self):
        return len(self.state)

    def store(self, e):
        '''
        Store a new experience in the buffer memory.
        :param experience: The new buffer entry to be stored {state, action, reward, next state}.
        '''
        print('Trying to store experience.')
        self.state.append(e[0])
        self.action.append(e[1])
        self.reward.append(e[2])
        self.nextstate.append(e[3])
        print('experience stored')

    def sample(self, samplesize):
        '''
        Take a subsample of defined size from the buffer memory. If samplesize is smaller than
        the current size of the buffer, a permutation is taken from the entire buffer.
        :param samplesize: Size of the sample to taken.
        :return: Sample taken from memory.
        '''
        class Batchsample():
            def __init__(self, buffer, idcs):
                self.idcs = idcs
                self.state = [buffer.state[i] for i in self.idcs]
                self.action = [buffer.action[i] for i in self.idcs]
                self.reward = [buffer.reward[i] for i in self.idcs]
                self.nextstate = [buffer.nextstate[i] for i in self.idcs]

        s = len(self) if samplesize > len(self) else samplesize
        print(len(self))
        si = np.random.choice(len(self), s)
        print(si)
        batch = Batchsample(self, si)
        print('batch selected')
        return batch


class DQN(nn.Module):
    def __init__(self, agent):
        super(DQN, self).__init__()
        agent.logger.info('DQN model created ...')
        self.agent = agent
        self.updates = 0

    def network_setup(self, inputchannels, eps=(0.9, 0.001), lint=10, tint=100, sint=1000, bs=32, gamma=0.95):
        ## Hyperparameters
        self.eps = (eps[0], eps[1], -1. * self.agent.s.n_rounds / np.log(1 / eps[0])) # Match ε-decay to n_round
        self.explay = Buffer(1000) # Set up experience replay buffer
        self.gamma = gamma  # Discount factor
        self.batchsize = bs # Batch size for learning
        self.learninginterval = lint # Learning interval
        self.targetinterval = tint # Interval for updating target DQN

        self.saveinterval = sint # Interval for saving values and parameters to file

        # Possible actions
        self.agent.poss_act = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']

        ## Network architecture
        self.conv1 = nn.Conv2d(inputchannels, 32, kernel_size=2, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 2, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2304, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, len(self.agent.poss_act))
        self.agent.logger.info('DQN is set up.')
        self.agent.logger.debug(self)

    def set_weights(self, random=True, file=False):

        def random_weights(model):
            if type(model) == nn.Conv2d or type(model) == nn.Linear:
                nn.init.uniform_(model.weight, -0.01, 0.01)
                model.bias.data.fill_(0.01)
        if random:
            # Set initial weights randomly
            self.apply(random_weights)
        elif file:
            self.agent.logger.info('No weights found!')
            # load weights of trained model from file and initialize ...

        self.agent.logger.info('Model initialized.')

    def load_from_file(self):
        pass

    def forward(self, x):
        # Forward calculation of neural activations ...
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out