
import numpy as np
from time import sleep
import pytorch as torch


training = True


def dqn_model():

    class DQN(torch.nn.Module):

        def __init__(self, parameters):
            super(DQN, self).__init__()
            # Network structure goes here

        def forward(self, variable):
            # Forward calculation of neural activations
            pass # return tensor

        def initialize(self, guess=None):
            # Set initial guess
            if not guess:
                pass #random guess

class buffer(object):
    '''
    Implementation of the Experience Replay as a cyclic memory buffer
    '''

    def __init__(self, buffersize):
        self.size = buffersize
        self.memory = []
        self.pos = 0

    def __len__(self):
        return len(self.memory)

    def store(self, *args):
        '''
        Store a new state in the buffer memory.

        :param args: ???
        '''

        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = # state, action, reward, next state
        self.pos = (self.pos + 1) % self.size

    def sample(self, samplesize):
        '''
        Take a subsample of defined size from the buffer memory.

        :param samplesize: Size of the sample to taken.
        :return: Sample taken from memory.
        '''

        return np.random.choice(self.memory, samplesize, replace=False)


def explore(n, start, end, decay):
    '''
    Implementation of an ε-greedy policy.
    Choose whether to explore or exploit in this step.

    :param n: Number of completed steps in the episode.
    :param start: Starting value of ε-threshold.
    :param end: Final value of ε-threshold.
    :param decay: Constant of the exponential threshold decay.
    :return: True if decided to explore.
    '''

    thresh = end + (start - end) * np.exp(-1. * n / decay)
    return True if np.random.random() > thresh else False


def select_action(self, training):
    '''
    Selects one of the possible actions based on the networks decision's (or randomly).

    :return: The chosen action {'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'}
    '''

    policy = None
    randact = None
    if training:
        return policy
    else:
        return randact if explore else policy


def setup(self):
    self.model = dqn_model()
    np.random.seed()
    self.model.initialize()
    self.logger.debug('Completed setup code.')


def act(self):
    self.logger.info('Picking action')
    action = select_action(self, training=training)
    self.next_action = action


def reward_update(agent):
    pass


def learn(agent):
    pass