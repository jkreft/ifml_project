
import numpy as np
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim

from train_settings import s, e

training_mode = False

### Model definition and Initialization

def dqn_model(training = False):
    class DQN(nn.Module):

        def __init__(self, insize, stacked):
            super(DQN, self).__init__()
            print('Initializing Model ...')
            self.training = training

            ## Hyperparameters
            # Calculate ε-decay to match number of rounds
            eps_start, eps_end = 0.2, 0.002
            self.eps = (eps_start, eps_end, -1. * s.n_rounds / np.log(1/eps_start))
            # Set up experience replay buffer
            self.explay = buffer(100)
            # Define size of batches to sample from buffer when learning
            self.batchsize = 32
            # Possible actions
            self.poss_act = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

            ## Network architecture
            self.conv1 = nn.Conv2d(stacked, insize, kernel_size=4, stride=2)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(64, 64, 2, 1)
            self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(3136, 512)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(512, len(self.poss_act))

            ## Initialization
            def initial_weights(model):
                if type(model) == nn.Conv2d or type(model) == nn.Linear:
                    nn.init.uniform_(model.weight, -0.01, 0.01)
                    model.bias.data.fill_(0.01)

            # Set initial weights
            self.apply(initial_weights)
            print('Model initialized.')


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

    model = DQN(4, 32)
    return model


class buffer(object):
    '''
    Implementation of the Experience Replay as a cyclic memory buffer. Allows to take a
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

### Dedicing on actions

def explore(n, eps):
    '''
    Implementation of an ε-greedy policy.
    Choose whether to explore or exploit in this step.
    :param n: Number of completed steps in the episode.
    :param eps: Parameters for ε-threshold {Starting value, Final value, Exponential decay constant}.
    :return: True if decided to explore.
    '''

    start, end, decay = eps
    thresh = end + (start - end) * np.exp(-1. * n / decay)
    return True if np.random.random() > thresh else False


def select_action(self, training=False):
    '''
    Selects one of the possible actions based on the networks decision's (or randomly).
    :return: The chosen action {'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'}
    '''

    step = self.game_state['step']

    def policy():
       pass

    if not training:
        return policy()
    else:
        return np.random.choice(self.model.model.poss_act) if explore(step, self.model.eps) else policy()

def select_random_action(self, training=False):
    print('Selecting action')
    randomaction = np.random.choice(self.model.poss_act)
    print(self.model.poss_act)
    print(randomaction)
    return randomaction

def construct_experience(arguments):
    experience = arguments
    return experience


### Training/Learning

def reward(events, rewardtab=None):
    '''
    Calculate the reward for one single or a sequence of events (e.g. an episode)
    based on an optionally provided reward table. If the input ist a list of events,
    the cumulative reward is returned.
    :param events: An event or a list of events.
    :param rewardtab: A different reward table can be provided optionally.
    :return: The (cumulative) reward.
    '''
    if type(events) is not type([]):
        events = [events]
    # Reward table (Order is as found in e: bomb, coin, crate, suicide, kill, killed, wait, invalid)
    if not rewardtab:
        rewardtab = [0, +100, +30, -100, +100, -300, -5, -10]
    # Set reward to zero, loop through events and add up rewards
    reward = 0
    for event in events:
        reward += rewardtab[event]
    return reward

# call as: ... cumulative_reward(self.game_state['events'])


### Main functions

def setup(self):
    '''
    Called once, before the first round starts. Initialization, in particular of the model
    :param self: Agent object.
    '''
    print('Training' if training_mode else 'Testing')
    self.model = dqn_model(training=training_mode)
    print(dqn_model())
    np.random.seed(42)
    self.logger.debug('Sucessfully completed setup code.')


def act(self):
    '''
    Called once every step to choose the next action.
    Game state is accessible via the previously updated 'self.game_state'.
    Any action chosen up to the expiry of the time limit will be executed.
    :param self:  Agent object.
    '''
    self.logger.info('Agent ' + self.name + ' is picking action ...')

    if self.model.training:
        state = self.game_state
        #? = construct_experience()
        experience = [state, action, reward, nxtstate]
        self.model.explay.store(experience)
    else:
        action = select_random_action(self, training=False)

    self.next_action = action


def reward_update(agent):
    '''
    When in training mode, called after each step except for the final one.
    Used to update training data and do calculations.
    :param agent: Agent object.
    :return:
    '''
    pass


def end_of_episode(agent):
    '''
    When in training mode, called at the end of each episode.
    :param agent: Agent object.
    '''
    # Do end-of-episode learning.
    # Save parameters / weights to file?

    pass