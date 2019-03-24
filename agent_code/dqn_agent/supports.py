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
def construct_reduced_state_tensor(agent, silent=True, zeropad=2):
    '''
    Create reduced state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''

    ######### help functions #########

    def noprint(*args):
        [print(*args) if not silent else None]

    def eucl_distance(dx, dy):
        ''' normal euclidean distance '''
        return np.sqrt(dx ** 2 + dy ** 2)

    def filter_targets(x, y, targets, dist=1):
        ''' filters out all targets with a distance from the agent equal less than the distance (dist).
            This allow to filter out the targets the agent is standing on (dist = 0),
            as well as the targests that are already included in the 3x3 inner state ring (dist = 1) '''
        targets = np.array(targets)
        dx = np.atleast_2d(targets)[:, 0] - x
        dy = y - np.atleast_2d(targets)[:, 1]
        mask = (eucl_distance(dx, dy) > dist*1.45)  # all targets have to be at least 2 tiles away to be considers
        return targets[mask]

    # maybe this works better with arcsin or another convention
    def get_sector(x, y, targets, size=16):  # size = 8 for 3x3 state
        ''' project all targets onto a ring of #size segments in the state
        '''
        dx = np.atleast_2d(targets)[:, 0] - x
        dy = y - np.atleast_2d(targets)[:, 1]
        dist = eucl_distance(dx, dy)
        # print("sign:", np.sign(dy))
        offset = np.where(np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)), 0,
                          2 * np.pi)
        sign = np.where(np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)), 1, -1)
        # print("offset:", offset)
        # print("distances:", dx, dist)
        noprint('Targets', targets)
        noprint("angles:", (sign * np.arccos(dx / dist) + offset) * (360 / (2 * np.pi)))
        sectors = size * ((sign * np.arccos(dx / dist) + offset) / (2 * np.pi)) - (1 / 2)
        sectors = np.where(sectors < 0, sectors + size, sectors)
        return np.array(sectors, dtype=np.int64), dist

    # TODO: other value-function e.g. exponential instead of linear
    def distance_to_value(dist):
        ''' norm distance to the maximal distance on the board and invert the values
        '''
        # value = 1 - dist/np.sqrt(s.rows**2 + s.cols**2)
        # print("value", value)
        return 1 - dist / np.sqrt(s.rows ** 2 + s.cols ** 2)

    def sector_to_state(sectors, values, size=16):
        ''' map the values according to their sectors (ring with #size segments) onto a state with size+1 entries
        '''
        try:
            if size == 8:
                # indices -> sector mapping = [[4,5,6], [3,8,7], [2,1,0]]
                indices = [8, 7, 6, 3, 0, 1, 2, 5, 4]
                state = np.zeros(size + 1)
                for i in np.arange(sectors.shape[0]):
                    state[indices[sectors[i]]] += values[i]
                return state.reshape(3, 3)

            elif size == 16:
                # indices -> sector mapping = [[9,10,11,12,13], [9,20,21,22,14], [7,19,24,23,15], [6,18,17,16,0], [5,4,3,2,1]]
                indices = [19, 24, 23, 22, 21, 20, 15, 10, 5, 0, 1, 2, 3, 4, 9, 14, 18, 17, 16, 11, 6, 7, 8, 13, 12]
                state = np.zeros(size + 9)
                for i in np.arange(sectors.shape[0]):
                    state[indices[sectors[i]]] += values[i]
                return state.reshape(5, 5)

            else:
                raise Exception("You choose the wrong size for the inner (8) or the outer (16) ring.")

        except Exception as error:
            print("caught the following error:", error)

    ######### state construction #########

    shp = agent.stateshape
    state = np.zeros((shp[0], shp[1]-zeropad*2, shp[2]-zeropad*2))

    x, y, name, bombs_left, score = agent.game_state['self']

    radius, size = 1, 16
    range = np.arange(s.rows)
    mask_x = np.logical_and(range >= x - radius, range <= x + radius)
    mask_y = np.logical_and(range >= y - radius, range <= y + radius)
    
    ######### arena #########

    arena = agent.game_state['arena']

    # projection of the crates on the outer state ring
    try:
        crates = filter_targets(x, y,
                                np.argwhere(arena == 0))  #### TODO: replace target: free tiles = 0 with crates = 1
        sectors, dist = get_sector(x, y, crates, size=16)
    except Exception as error:
        print("hier ist auch schon ein Fehler aufgetreten:", error)
    state[0] = sector_to_state(sectors, distance_to_value(dist), size=16)

    # inner 3x3 map
    state[0][1:-1, 1:-1] = arena[mask_y, :][:, mask_x]

    ######### coins #########

    coins = np.array(agent.game_state['coins'])

    if coins.size > 0:
    
        # projection of the coins onto the outer state ring
        sectors, dist = get_sector(x, y, filter_targets(x, y, coins, dist=1), size=16)
        noprint("sectors:", sectors)
        
        state[1] = sector_to_state(sectors, distance_to_value(dist), size=16)
    
        # inner 3x3 map             #### TODO: vectorize
        coins_map = np.zeros((s.cols, s.rows))
        coins_map[tuple(np.array(coins).T)] = 1
    
        state[1][1:-1, 1:-1] = coins_map[mask_y, :][:, mask_x]

    ######### other players #########
    
    others = np.array([(x, y) for (x, y, n, b, s) in agent.game_state['others']])
    
    if shp[0] >= 3 and others.size > 0:
        
        # projection
        sectors, dist = get_sector(x, y, filter_targets(x, y, others, 1), size=16)
        noprint("sectors:", sectors)
        state[2] = sector_to_state(sectors, distance_to_value(dist), size=16)
        
        # inner 3x3 map
        others_map = np.zeros((s.cols,  s.rows))
        others_map[tuple(np.array(others).T)] = 1
        state[2][1:-1, 1:-1] = others_map[mask_y, :][:, mask_x]

        
    ######### bombs #########   at the moment with one layer
    
    bombs = np.array(agent.game_state['bombs']) # (x,y,t)
    
    if shp[0] >= 4 and bombs.size > 0:        
        #bombs_xy =  filter_targets(x, y, bombs[:,0:2], 1)
        #arg_bombs = np.argwhere( bombs_xy == bombs[:,0:2])
        #print(arg_bombs)
        print("bomb timer:", bombs[:,2])
        
        # bombs projection 5x5 map
        sectors, dist = get_sector(x, y, filter_targets(x, y, bombs[:,0:2], 1), size=16) 
        print(sectors, dist)
        if sectors.size > 0:
            values = distance_to_value(dist) + (1 / (filter_targets(x,y,bombs[:,2],1)+1)) # normal distance value + a 
            state[1] = sector_to_state(sectors, values, size=16)
        
        # inner 3x3 map
        bombs_map = np.zeros((s.cols,  s.rows))
        
        i = 0
        for b in np.atleast_2d(bombs[:,0:2]):
            print("b:", b)
            bombs_map[tuple(b)] = bombs[i:,2]
            i += 1
        print(bombs_map)
        state[1][1:-1, 1:-1] = bombs_map[mask_y, :][:, mask_x]
        
        print(state[1])

    ######### return state #########

    
    #state[0] = np.pad(state[0], pad_width=2, mode='constant', constant_values=0)
    #state[1] = np.pad(state[0], pad_width=2, mode='constant', constant_values=0)
    
    # pad state
    state = np.array([np.pad(s, pad_width = 2, mode="constant", constant_values = 0) for s in state])
    
    noprint("state 0:", state[0])
    noprint("state 1:", state[1])

    reducedstate = T.from_numpy(state)
    noprint("final reduced state:", reducedstate)
    if T.cuda.is_available():
        reducedstate = reducedstate.cuda()
    return reducedstate


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
    avgweights = np.linalg.norm(np.concatenate([l.detach().numpy().flatten() for l in agent.model.parameters()]))
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
        'weights': np.array(b.weights).mean()
    }
    agent.analysis.append(avgdata)
    agent.analysisbuffer = Analysisbuffer()

