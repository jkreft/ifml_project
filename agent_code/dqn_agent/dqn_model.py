
import torch as T
import torch.nn as nn
import numpy as np


########################################################
#####           Experience Replay Buffer           #####
########################################################

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
                self.nextstate = buffer.nextstate[[idcs]]

        s = len(self) if samplesize > len(self) else samplesize
        si = np.random.choice(len(self), s)
        batch = Batchsample(self, si)
        return batch


########################################################
#####                Deep Q-Network                #####
########################################################

class DQN(nn.Module):
    def __init__(self, agent):
        super(DQN, self).__init__()
        agent.logger.info('DQN model created ...')
        self.agent = agent
        self.updates = 0

    def network_setup(self, channels=1, eps=(1, 0.1), lint=8, tint=1000, sint=50000, minibatch=32, gamma=0.95, analysis=False):
        ## Hyperparameters
        self.eps = (                                                                # Match ε-decay to n_round
            eps[0],                                                                 # Starting value
            eps[1],                                                                 # Terminal value
            (eps[1]-eps[0])/self.agent.s.n_rounds/self.agent.s.max_steps,           # Linear decay slope
            np.log(eps[1]/eps[0])/self.agent.s.n_rounds/self.agent.s.max_steps      # Exponential decay constant
        )
        self.gamma = gamma  # Discount factor
        self.batchsize = minibatch # Batch size for learning
        self.learninginterval = lint # Learning interval
        self.targetinterval = tint # Interval for updating target DQN

        # Intervals
        self.saveinterval = sint # Interval for saving full model, parameters and buffers to file
        self.analysisinterval = analysis # Interval for saving selected data for later analysis

        # Possible actions
        self.agent.poss_act = self.agent.s.actions

        ## Network architecture
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=2, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
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

    def forward(self, x):
        # Forward calculation of neural activations ...
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size(0), -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

    def explore(self, fct):
        '''
        Implementation of an ε-greedy policy.
        Choose whether to explore or exploit in this step.
        :param decay: Type of decay for the ε-threshold {'lin', 'exp'}
        :return: True if decided to explore.
        '''
        start, end, slope, exponent = self.eps

        if fct == 'exp':
            thresh = end + (start - end) * np.exp(exponent * self.agent.trainingstep)
        elif fct == 'lin':
            thresh = start + slope * self.agent.trainingstep
        else:
            print(f'Decay function: {fct} not known! Choosing linear decay instead.')
            thresh = start + slope * self.agent.trainingstep

        self.stepsilon = thresh
        return True if np.random.random() < thresh else False