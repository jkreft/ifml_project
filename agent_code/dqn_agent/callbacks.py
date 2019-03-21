
import numpy as np
from time import sleep

import torch as T
import torch.nn as nn
import torch.optim as optim

from agent_code.dqn_agent.dqn_model import DQN
from train_settings import s, e


training_mode = False


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


def select_action(agent):
    '''
    Selects one of the possible actions based on the networks decision's (or randomly).
    :return: The chosen action {'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'}
    '''
    agent.logger.debug('Entered action selection.')
    step = agent.game_state['step']

    def policy(agent):
        output = agent.model(agent.currentstate)[0]
        action = T.argmax(output) # .item() ?
        action = agent.poss_act[action]
        return action

    if not agent.training:
        return policy(agent)
    else:
        return np.random.choice(agent.poss_act) if explore(step, agent.model.eps) else policy(agent)

def select_random_action(agent):
    agent.logger.debug('Selecting random action.')
    return np.random.choice(agent.poss_act)

def construct_experience(arguments):
    experience = arguments
    return experience

def construct_state_tensor(agent):
    # Create state tensor from game_state data
    state = T.zeros(agent.stateshape)
    # Represent the arena (walls, ...)
    state[0, 0] = T.from_numpy(agent.game_state['arena'])
    # The agent's own position
    state[0, 1, agent.game_state['self'][0], agent.game_state['self'][1]] = 1
    # Positions of coins
    for coin in agent.game_state['coins']:
        state[0, 2, coin[0], coin[1]] = 1
    # Other players' positions
    #for other in agent.game_state['others']:
    #    state[0, 2, other[0], other[1]] = 1
    # Bomb position and countdown-timers
    #for bomb in agent.game_state['bombs']:
    #    state[0, 3, bomb[0], bomb[1]] = bomb[2]
    # Positions of explosions
    #state[0, 4] = T.from_numpy(agent.game_state['explosions'])

    return state


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
    if not rewardtab:
    # Reward table (Order is as found in e: bomb, coin, crate, suicide, kill, killed, wait, invalid)
        rewardtab = [0, +100, +30, -100, +100, -300, -5, -10]
    # Set reward to zero, loop through events and add up rewards
    reward = 0
    for event in events:
        reward += rewardtab[event]
    return reward


############################
###### Main functions ######
############################


def setup(self):
    '''
    Called once, before the first round starts. Initialization, in particular of the model
    :param self: Agent object.
    '''
    self.logger.info('Mode: Training' if training_mode else 'Mode: Testing')
    self.training = training_mode
    np.random.seed(42)
    self.s = s
    self.e = e

    statechannels = 3

    self.stateshape = (1, statechannels, s.cols, s.rows)
    self.model = DQN(self)
    self.model.network_setup(statechannels)

    self.model.set_weights(random=True)

    # optimizer
    lr = 0.001
    try:
        self.model.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    except Exception:
        self.agent.logger.info('Failed initializing model optimizer.')

    self.logger.debug('Sucessfully completed setup code.')


def act(self):
    '''
    Called once every step to choose the next action.
    Game state is accessible via the previously updated 'self.game_state'.
    Any action chosen up to the expiry of the time limit will be executed.
    :param self:  Agent object.
    '''
    self.logger.info(f'Agent {self.name} is picking action ...')

    self.currentstate = construct_state_tensor(self)

    if self.training:
        action = select_action(self)
        #state, action, reward, nxtstate = self.curent_state, 0, 0, 0
        #experience = construct_experience() # [state, action, reward, nxtstate]
        #self.model.explay.store(experience)
        # predict q, ...
    else:
        action = select_action(self)

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