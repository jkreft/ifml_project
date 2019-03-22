# -*- coding: future_fstrings -*-
# First line is to enable f-strings in python3.5 installation on servers

import os
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch as T
from settings import s, e

home = os.path.expanduser("~") + '/'


### Construct the tensor representing the game state in each step ###
def construct_state_tensor(agent):
    '''
    Create image-like (pixel-based) state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''
    state = T.zeros(agent.stateshape)
    # Represent the arena (walls, ...)
    state[0] = T.from_numpy(agent.game_state['arena'])
    # The agent's own position
    state[1, agent.game_state['self'][0], agent.game_state['self'][1]] = 1
    # Positions of coins
    for coin in agent.game_state['coins']:
        state[2, coin[0], coin[1]] = 1
    # Other players' positions
    #for other in agent.game_state['others']:
    #    state[2, other[0], other[1]] = 1
    # Bomb position and countdown-timers
    #for bomb in agent.game_state['bombs']:
    #    state[3, bomb[0], bomb[1]] = bomb[2]
    # Positions of explosions
    #state[4] = T.from_numpy(agent.game_state['explosions'])
    if T.cuda.is_available():
        state = state.cuda()
    return state


### Construct a feature-reduced version of the state tensor
def construct_reduced_state_tensor(agent, silent=True):
    '''
    Create reduced state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''
    # state = T.zeros(agent.reducedstateshape)

    def noprint(*args):
        [print(args) if not silent else None]

    state = np.zeros((2, 5, 5))
    # state = T.zeros(agent.stateshape)

    x, y, name, bombs_left, score = agent.game_state['self']

    radius = 1
    range = np.arange(s.rows)
    mask_x = np.logical_and(range >= x - radius, range <= x + radius)
    mask_y = np.logical_and(range >= y - radius, range <= y + radius)

    # arena = agent.game_state['arena'][mask_y, :][:,mask_x]
    state[0][0:3, 0:3] = agent.game_state['arena'][mask_y, :][:, mask_x]

    coins = agent.game_state['coins']

    def eucl_distance(dx, dy):
        return np.sqrt(dx ** 2 + dy ** 2)

    # maybe this works better with arcsin or another convention
    def get_sector(x, y, targets, size=8):  # size = 16 for 5x5 state
        dx = np.atleast_2d(targets)[:, 0] - x
        dy = y - np.atleast_2d(targets)[:, 1]
        dist = eucl_distance(dx, dy)
        # print("sign:", np.sign(dy))
        offset = np.where(np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)), 0, 2 * np.pi)
        sign = np.where(np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)), 1, -1)
        # print("offset:", offset)
        # print("distances:", dx, dist)
        noprint('Muenzen', coins)
        noprint("angles:", (sign * np.arccos(dx / dist) + offset) * (360 / (2 * np.pi)))
        sectors = size * ((sign * np.arccos(dx / dist) + offset) / (2 * np.pi)) - (1 / 2)
        sectors = np.where(sectors < 0, sectors + 8, sectors)
        return np.array(sectors, dtype=np.int64), dist

    def distance_to_value(dist):
        ''' norm distance to the maximal distance on the board and invert the values
        '''
        # value = 1 - dist/np.sqrt(s.rows**2 + s.cols**2)
        # print("value", value)
        return 1 - dist / np.sqrt(s.rows ** 2 + s.cols ** 2)

    def sector_to_state(sectors, values, size=8):
        ''' map the values according to their sectors (ring with #size segments) onto a state with size+1 entries
        '''
        # indice>sector mapping = [[4,5,6], [3,8,7], [2,1,0]]
        indices = [8, 7, 6, 3, 0, 1, 2, 5, 4]
        state = np.zeros(size + 1)
        noprint("sectors an stelle 0:", sectors[0])
        for i in np.arange(sectors.shape[0]):
            noprint("sector an stelle i: ", sectors[i])
            state[indices[sectors[i]]] += values[i]
        return state.reshape(3, 3)

    sectors, dist = get_sector(x, y, coins)
    noprint("sectors:", sectors)
    values = distance_to_value(dist)
    state[1][0:3, 0:3] = sector_to_state(sectors, values)

    noprint("state 0:", state[0])
    noprint("state 1:", state[1])

    reducedstate = T.from_numpy(state)
    if T.cuda.is_available():
        reduced = reducedstate.cuda()
    return reducedstate


### Loading and Saving Models and Data ###
def load_model(agent, modelpath=False, explaypath=False, trainingmode=False):
    try:
        if modelpath == False:
            d = home + 'models/load/model/'
            # Choose first file in directory d
            modelpath = d + [x for x in os.listdir(d)][0]
        data = T.load(modelpath, map_location=agent.device)
        agent.model.load_state_dict(data['model'])
        agent.targetmodel.load_state_dict(data['model'])
        agent.model.optimizer.load_state_dict(data['optimizer'])
        agent.analysis = data['analysis']
        agent.modelname = modelpath.split('/')[-1].split('.pth')[0]
        agent.trainingstep = data['trainingstep']
        agent.logger.info(f'Model was loaded from file at step {agent.trainingstep}')
        print(f'Model was loaded from file at step {agent.trainingstep}')
        if trainingmode:
            if modelpath == False:
                d = home + 'models/load/explay/'
                # Choose first file in directory d
                explaypath = d + [x for x in os.listdir(d)][0]
            data = T.load(explaypath)
            agent.explay = data['explay']
            agent.logger.info('Experience replay buffer was loaded from file.')
            print('Experience replay buffer was loaded from file.')
        else:
            agent.logger.info('Loaded model without Experience replay buffer.')
            print('Loaded model without Experience replay buffer.')

    except Exception as error:
        agent.logger.info(f'A file could not be found. Error: {error}\nModel and buffer were not loaded!')
        print(f'A file could not be found. Error: {error}\nModel and buffer were not loaded!')


def save_model(agent):
    if not os.path.exists(home + 'models/saved/'):
        if not os.path.exists(home + 'models/'):
            os.mkdir(home + 'models/')
        os.mkdir(home + 'models/saved/')
    modelpath = home + 'models/saved/' + 'model-' + agent.modelname + '_step-' + str(agent.trainingstep) \
               + '_aint-' + str(agent.model.analysisinterval) + '_lint-' + str(agent.model.learninginterval) + '.pth'
    T.save({
        'model': agent.model.state_dict(),
        'optimizer': agent.model.optimizer.state_dict(),
        'analysis': agent.analysis,
        'trainingstep': agent.trainingstep,
        'learninginterval': agent.model.learninginterval,
    }, modelpath)
    if not os.path.exists(home + 'explay/saved/'):
        if not os.path.exists(home + 'explay/'):
            os.mkdir(home + 'explay/')
        os.mkdir(home + 'explay/saved/')
    print(f'Saved model at step {agent.trainingstep}. Filename: {modelpath}')
    explaypath = home + 'explay/saved/' + 'model-' + agent.modelname + '_step-' + str(agent.trainingstep) \
               + '_aint-' + str(agent.model.analysisinterval) + '_lint-' + str(agent.model.learninginterval) + '.pth'
    T.save({
        'explay': agent.explay
    }, explaypath)
    print(f'Saved experience replay buffer. Filename: {explaypath}')


class Analysisbuffer:
    def __init__(self):
        self.action = []
        self.reward = []
        self.score = []
        self.epsilon = []
        self.expl = []
        self.loss = []
        self.q = []

def analysisbuffer():
    return Analysisbuffer()

def step_analysis_data(agent):
    agent.analysisbuffer.action.append(agent.stepaction.cpu().numpy())
    agent.analysisbuffer.reward.append(agent.stepreward)
    agent.analysisbuffer.score.append(agent.finalscore)
    agent.analysisbuffer.epsilon.append([agent.model.stepsilon, agent.model.stepsilon2])
    agent.analysisbuffer.expl.append(agent.exploration)
    agent.analysisbuffer.loss.append(agent.plotloss.detach().numpy())
    agent.analysisbuffer.q.append(agent.stepq.cpu().detach().numpy())


def average_analysis_data(agent):
    b = agent.analysisbuffer
    avgdata = {
        'learningstep': agent.model.learningstep,
        'action': np.array(b.action).mean(),
        'reward': np.array(b.reward).mean(),
        'score': np.array(b.score).mean(),
        'epsilon': np.array(b.epsilon).mean(axis=0),
        'exploration': np.array([b.expl.count('policy'), b.expl.count('random'), b.expl.count('rolemodel')]),
        'loss': np.array(b.loss).mean(),
        'q': np.array(b.q).mean()
    }
    agent.analysis.append(avgdata)
    agent.analysisbuffer = Analysisbuffer()

