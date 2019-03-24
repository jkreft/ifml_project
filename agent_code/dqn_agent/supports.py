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

### Construct the tensor representing the game state in each step ###
def construct_time_state_tensor(agent):
    '''
    Create image-like (pixel-based) single object channel, multi-frame state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''
    ss = agent.stateshape
    state = T.zeros(*ss)
    newest = ss[0]-1
    # Shift last screen into the past, drop oldest, leave room for one new screen
    state[0] = agent.laststate[1]
    state[1] = agent.laststate[2]
    state[2] = agent.laststate[3]
    # Represent the object layers in one single image-like tensor
    # Initialize numpy array layers
    coins, walls, crates, myself = [np.zeros_like(agent.game_state['arena']) for i in range(4)]
    # Set "Brightness value" for each
    walls[np.where(agent.game_state['arena'] == -1)] = 10
    crates[np.where(agent.game_state['arena'] == +1)] = 20
    myself[agent.game_state['self'][0], agent.game_state['self'][1]] = 100
    for coin in agent.game_state['coins']:
        coins[coin[0], coin[1]] = 40
    state[newest] += T.from_numpy(walls).to(dtype=T.float)
    state[newest] += T.from_numpy(crates).to(dtype=T.float)
    state[newest] += T.from_numpy(myself).to(dtype=T.float)
    state[newest] += T.from_numpy(coins).to(dtype=T.float)

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
    #print('State:')
    #print(state.shape)
    #for i in range(state.shape[0]):
    #    print(i)
    #    print(state[i])
    return state


### Loading and Saving Models and Data ###
def load_model(agent, modelpath=False, explaypath=False, trainingmode=False):
    try:
        if modelpath == False:
            d = home + 'models/load/'
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
            if explaypath == False:
                d = home + 'explay/load/'
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
        'info': agent.model.info,
    }, modelpath)
    '''
    if not os.path.exists(home + 'explay/saved/'):
        if not os.path.exists(home + 'explay/'):
            os.mkdir(home + 'explay/')
        os.mkdir(home + 'explay/saved/')
    print(f'Saved model at step {agent.trainingstep}. Filename: {modelpath}')
    explaypath = home + 'explay/saved/' + 'explay-' + agent.modelname + '_step-' + str(agent.trainingstep) \
               + '_aint-' + str(agent.model.analysisinterval) + '_lint-' + str(agent.model.learninginterval) + '.pth'
    T.save({
        'explay': agent.explay
    }, explaypath)
    print(f'Saved experience replay buffer. Filename: {explaypath}')
    '''

class Analysisbuffer:
    def __init__(self):
        self.action = []
        self.reward = []
        self.score = []
        self.epsilon = []
        self.expl = []
        self.loss = []
        self.q = []
        self.weights = []

def analysisbuffer():
    return Analysisbuffer()

def step_analysis_data(agent):
    buff = agent.analysisbuffer
    buff.action.append(agent.stepaction.cpu().numpy())
    buff.reward.append(agent.stepreward)
    buff.score.append(agent.finalscore)
    buff.epsilon.append([agent.model.stepsilon, agent.model.stepsilon2])
    buff.expl.append(agent.exploration)
    buff.loss.append(agent.plotloss.detach().numpy())
    buff.q.append(agent.stepq.cpu().detach().numpy())
    avgweights = np.linalg.norm(np.concatenate([l.cpu().detach().numpy().flatten() for l in agent.model.parameters()]))
    buff.weights.append(avgweights)


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
        'q': np.array(b.q).mean(),
        'weights': np.array(b.weights).mean(),
        'example': {
            'state': agent.stepstate,
            'q': agent.stepq,
            'loss': agent.steploss,

        }
    }
    agent.analysis.append(avgdata)
    agent.analysisbuffer = Analysisbuffer()

