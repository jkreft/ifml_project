
import torch.nn as nn
import numpy as np



class buffer(object):
    '''
    Implementation of the Experience Replay as a cyclic memory buffer.
    Allows to take a random sample of defined maximal size from the buffer for learning.
    '''

    def __init__(self, buffersize):
        self.size = buffersize
        self.memory = []
        self.pos = 0

    def __len__(self):
        return len(self.memory)

    def store(self, entry):
        '''
        Store a new state in the buffer memory.
        :param entry: The new buffer entry to be stored {state, action, reward, next state}.
        '''

        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.pos] = entry
        self.pos = (self.pos + 1) % self.size

    def sample(self, samplesize):
        '''
        Take a subsample of defined size from the buffer memory. If samplesize is smaller than
        the current size of the buffer, a permutation is taken from the entire buffer.
        :param samplesize: Size of the sample to taken.
        :return: Sample taken from memory.
        '''
        s = len(self.memory) if samplesize > len(self.memory) else samplesize
        return np.random.choice(self.memory, s, replace=False)



class DQN(nn.Module):
    def __init__(self, agent):
        super(DQN, self).__init__()
        agent.logger.info('DQN model created ...')
        self.agent = agent

    def network_setup(self, inputchannels):
        ## Hyperparameters
        # Calculate ε-decay to match number of rounds
        eps_start, eps_end = 0.2, 0.002
        self.eps = (eps_start, eps_end, -1. * self.agent.s.n_rounds / np.log(1 / eps_start))
        # Set up experience replay buffer
        self.explay = buffer(100)
        # Define size of batches to sample from buffer when learning
        self.gamma = 0.95  # discount factor
        self.batchsize = 32

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
        self.agent.logger.debug(self.model)

    def set_weights(self, random=True, file=False):
        ## Initialization of DQN weights

        if random:
            # Set initial weights randomly
            def random_weights(model):
                if type(model) == nn.Conv2d or type(model) == nn.Linear:
                    nn.init.uniform_(model.weight, -0.01, 0.01)
                    model.bias.data.fill_(0.01)

            self.apply(random_weights)
        elif file:
            self.agent.logger.info('No weights found!')
            # load weights of trained model from file and initialize

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