# -*- coding: future_fstrings -*-
# First line is to enable f-strings in python3.5 installation on vserver

import numpy as np
import os
from datetime import datetime

import torch as T
import torch.nn as nn

from agent_code.simple_agent import callbacks as rolemodel

from agent_code.dqn_agent.dqn_model import DQN, Buffer
from settings import s, e



### Flags for choosing in which settings to run ###

training_mode = True
load_from_file = False
analysis_interval = 1000
save_interval = 400000
start_learning = 100000



########################################################################################################################
#####                                            Supporting Functions                                              #####
########################################################################################################################

### Loading and Saving Models and Data ###

def load_model(agent, filepath=False):
    try:
        if filepath == False:
            d = './models/load/'
            # Choose first file in directory d
            filepath = d + [x for x in os.listdir(d) if os.path.isfile(d + x)][0]
        data = T.load(filepath, map_location=agent.device)
        agent.model.load_state_dict(data['model'])
        agent.targetmodel.load_state_dict(data['model'])
        agent.model.optimizer.load_state_dict(data['optimizer'])
        agent.analysis = data['analysis']
        agent.modelname = filepath.split('/')[-1].split('.pth')[0]
        agent.explay = data['explay']
        agent.trainingstep = data['trainingstep']
        agent.logger.info(f'Model was loaded from file at step {agent.trainingstep}')
        print(f'Model was loaded from file at step {agent.trainingstep}')
    except Exception as error:
        agent.logger.info(f'No file could be found. Error: {error}\nModel was not loaded!')
        print(f'No file could be found. Error: {error}\nModel was not loaded!')


def save_model(agent):
    if not os.path.exists('./models'):
        os.mkdir('./models')
    filename = './models/' + 'model-' + agent.modelname + '_step-' + str(agent.trainingstep) \
               + '_aint-' + str(agent.model.analysisinterval) + '_lint-' + str(agent.model.learninginterval) + '.pth'
    T.save({
        'model': agent.model.state_dict(),
        'optimizer': agent.model.optimizer.state_dict(),
        'explay': agent.explay,
        'analysis': agent.analysis,
        'trainingstep': agent.trainingstep,
        'learninginterval': agent.model.learninginterval,
    }, filename)
    print(f'Saved model at step {agent.trainingstep}. Filename: {filename}')


class Analysisbuffer:
    def __init__(self):
        self.action = []
        self.reward = []
        self.epsilon = []
        self.explored = []
        self.loss = []
        self.q = []


def step_analysis_data(agent):
    agent.analysisbuffer.action.append(agent.stepaction.cpu().numpy())
    agent.analysisbuffer.reward.append(agent.stepreward)
    agent.analysisbuffer.epsilon.append(agent.model.stepsilon)
    agent.analysisbuffer.explored.append(agent.explored)
    agent.analysisbuffer.loss.append(agent.steploss.detach().numpy())
    agent.analysisbuffer.q.append(agent.stepq.detach().numpy())

def average_analysis_data(agent):
    buffer = agent.analysisbuffer
    avgdata = {
        'learningstep': agent.model.learningstep,
        'action': np.array(buffer.action).mean(),
        'reward': np.array(buffer.reward).mean(),
        'epsilon': np.array(buffer.epsilon).mean(),
        'explored': np.array(buffer.explored).mean(),
        'loss': np.array(buffer.loss).mean(),
        'q': np.array(buffer.q).mean
    }
    agent.analysis.append(avgdata)
    agent.analysisbuffer = Analysisbuffer()


### Deciding on Actions ###

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

def construct_reduced_state_tensor(agent):
    '''
    Create reduced state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''
    state = T.zeros(agent.reducedstateshape)


    if T.cuda.is_available():
        state = state.cuda()
    return state



def select_action(agent, rolemodel=False):
    '''
    Selects one of the possible actions based on the networks decision's (or randomly).
    :return: The chosen action {'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'}
    '''
    agent.logger.debug('Entered action selection.')

    def policy(agent):
        input = agent.stepstate[None,:,:,:]
        output = agent.model(input)[0]
        return T.argmax(output)

    if agent.training:
        if agent.model.explore('exp'):
            agent.explored = 1
            action = T.tensor(np.random.choice(len(agent.possibleact)))
            agent.logger.debug('Exploring: Chose ' + str(action))
        else:
            agent.explored = 0
            if rolemodel:
                rolemodel.act(agent)
                action = T.tensor(agent.possibleact.index(agent.next_action))
                agent.logger.debug('Using rolemodel: Chose ' + str(action))
            else:
                action = policy(agent)
                agent.logger.debug('Using policy: Chose ' + str(action))
    else:
        action = policy(agent)
    return action.to(agent.device)


def select_random_action(agent):
    agent.logger.debug('Selecting random action.')
    return T.tensor(np.random.choice(len(agent.possibleact)))


### Training/Learning ###

def get_cookies(agent, rewardtab=None):
    '''
    Calculate the reward for one single or a sequence of events (e.g. an episode)
    based on an optionally provided reward table. If the input ist a list of events,
    the cumulative reward is returned.
    :param events: An event or a list of events.
    :param rewardtab: A different reward table can be provided optionally.
    :return: The (cumulative) reward.
    '''
    events = agent.events
    if type(events) is not type([]):
        events = [events]
    if not rewardtab:
        # Reward table. The order is as found in e:
        # 'MOVED_LEFT', 'MOVED_RIGHT', 'MOVED_UP', 'MOVED_DOWN', 'WAITED', 'INTERRUPTED', 'INVALID_ACTION', 'BOMB_DROPPED',
        # 'BOMB_EXPLODED','CRATE_DESTROYED', 'COIN_FOUND', 'COIN_COLLECTED', 'KILLED_OPPONENT', 'KILLED_SELF', 'GOT_KILLED',
        # 'OPPONENT_ELIMINATED', 'SURVIVED_ROUND'
        rewardtab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0]

    # Initialize reward, loop through events, and add up rewards
    reward = -0.001
    for event in events:
        reward += rewardtab[event]
    return reward


def construct_experience(agent):
    return agent.laststate, T.LongTensor([[agent.lastaction]]), T.tensor([agent.stepreward]).float(), agent.stepstate


def terminal_state():
    pass



########################################################################################################################
#####                                               Main Functions                                                 #####
########################################################################################################################


def setup(self):
    '''
    Called once, before the first round starts. Initialization, in particular of the model
    :param self: Agent object.
    '''
    self.training = training_mode
    modestr = 'TRAINING' if self.training else 'TEST'
    print(f'Running in {modestr} mode')
    self.logger.info(f'Mode: {modestr}')
    self.s, self.e = s, e
    self.analysisbuffer = Analysisbuffer()
    self.startlearning = start_learning
    self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    self.cuda = T.cuda.is_available()
    print(f'Cuda is{"" if self.cuda else " not"} available.')
    self.logger.info(f'Cuda is{"" if self.cuda else " not"} available.')
    # Adapt state-tensor to current task (bombs, other players, etc)
    channels = 3
    self.stateshape = (channels, s.cols, s.rows)
    self.reducedstateshape = ()

    # Create and setup model and target DQNs
    self.model = DQN(self)
    self.targetmodel = DQN(self)
    self.model.network_setup(channels=self.stateshape[0], aint=analysis_interval, sint=save_interval, tint=2000)
    self.targetmodel.network_setup(channels=self.stateshape[0])
    # Put DQNs on cuda if available
    self.model, self.targetmodel = self.model.to(self.device), self.targetmodel.to(self.device)
    # Load previous status from file or start training from the beginning
    if load_from_file:
        load_model(self)
    else:
        # Setup new experience replay
        self.explay = Buffer(500000, self.stateshape)
        self.modelname = str(datetime.now())[:-7]
        print('Modelname:', self.modelname)
        self.logger.info('Modelname:' + self.modelname)

        # Initialize model DQN and target DQN weights randomly
        self.model.set_weights(random=True)
        self.targetmodel.set_weights(random=True)
        print('Initializing model weights randomly.')

        self.trainingstep = 1
        self.model.learningstep = 1
        self.analysis = []

    self.steploss = T.tensor(100)
    self.stepq = T.tensor(np.zeros(len(self.possibleact)))

    self.model.explay = self.explay
    self.targetmodel.explay = self.explay
    self.episodeseq = []

    # Setting up simple_agent as "role model" for collecting good training data in the first steps
    rolemodel.setup(self)
    self.logger.debug('Sucessfully completed setup code.')


def act(self):
    '''
    Called once every step to choose the next action.
    Game state is accessible via the previously updated 'self.game_state'.
    Any action chosen up to the expiry of the time limit will be executed.
    :param self:  Agent object.
    '''
    self.logger.info(f'Agent {self.name} is picking action ...')
    try:

        # Build state tensor
        self.stepstate = construct_state_tensor(self)

        if not self.training:
            # Choose next action
            self.stepaction = select_action(self)
            print(f'Step {self.game_state["step"]}, choosing action {self.possibleact[self.stepaction.item()]}.')

        else:
            t = self.trainingstep
            if (t % 1000 == 0) or (t < 101 and t % 10 == 0) or (t < 1001 and t % 100 == 0):
                print(f'Training step {t}')

            if t <= self.startlearning:
                # Check if this is the first step in episode and initialize variables
                if t == 1:
                    self.laststate = None
                    self.lastaction = None
                    self.lastevents = None
                # Choose random action
                self.stepaction = select_action(self, rolemodel=rolemodel)
            else:
                # Choose next action
                self.stepaction = select_action(self)

            # Calculate reward for the events leading to this step
            self.stepreward = get_cookies(self)
            # Construct and store experience tuple
            if self.lastaction is not None:
                s, a, r, n = construct_experience(self)
                self.episodeseq.append([s, a, r, n, self.lastevents])
            # Update state and action variables
            self.laststate = self.stepstate
            self.lastaction = self.stepaction
            self.lastevents = self.events

            if (t % self.model.learninginterval == 0) and (t > self.s.max_steps):
                self.logger.debug('Learning step ...')


                # Sample batch of batchsize from experience replay buffer
                batch = self.explay.sample(self.model.batchsize)

                # Non final check (like in pytorch RL tutorial)
                nf = T.LongTensor([i for i in range(len(batch.nextstate)) if
                                   (batch.nextstate[i] == 0).sum().item() != np.prod(
                                       np.array(self.stateshape))])
                nfnext = batch.nextstate[nf]

                if T.cuda.is_available():
                    batch.state = batch.state.cuda()
                    batch.action = batch.action.cuda()
                    nfnext = nfnext.cuda()
                #print('marker0')
                self.stepq = self.model(batch.state) # Get q-values from state using the model
                #print('marker1')
                self.stepq = self.stepq.gather(1, batch.action) # Put together with actions
                nextq = T.zeros((len(batch.nextstate), len(self.possibleact))).cpu()
                nfnextq = self.targetmodel(nfnext).cpu()

                # Let nextq only contain the output for which the input states were non-final
                nextq.index_copy_(0, nf, nfnextq)
                nextq = nextq.max(1)[0]

                # Expected q-values for current state
                expectedq = ( (nextq * self.model.gamma) + batch.reward ).to(self.device)
                self.steploss = self.model.loss(self.stepq, expectedq)
                self.steploss = self.steploss.cpu()
                batch.state = batch.state.cpu()
                batch.action = batch.action.cpu()

                #print('marker3')
                '''
                q = self.model(batch.state)  # Get q-values from state using the model
                q = q.gather(1, batch.action)  # Put together with actions
                nextq = T.zeros((len(batch.nextstate), len(self.possibleact)))
                nfnextq = self.targetmodel(nfnext)
                
                # Make nextq so that it only contains the output for which the input states were non-final
                nextq.index_copy_(0, nf, nfnextq)
                nextq = nextq.max(1)[0]
                # Expected q-values for current state
                expectedq = (nextq * self.model.gamma) + batch.reward
                self.steploss = self.model.loss(q, expectedq)
                '''
                self.logger.info('The loss in this learning step was ' + str(self.steploss))
                self.model.optimizer.zero_grad()
                self.steploss.backward()
                self.model.optimizer.step()

                if self.model.learningstep % self.model.targetinterval == 0:
                    self.targetmodel.load_state_dict(self.model.state_dict())
                self.model.learningstep += 1

            # If analysisinterval True, save data and average over every interval
            if self.model.analysisinterval:
                step_analysis_data(self)
                if t % self.model.analysisinterval == 0:
                    average_analysis_data(self)

            if t % self.model.saveinterval == 0:
                save_model(self)

    except Exception as error:
        print('Exception in act()\n ' + str(error))
        self.stepaction = select_random_action(self)

    self.next_action = self.possibleact[int(self.stepaction.item())]


def reward_update(self):
    '''
    When in training mode, called after each step except for the final one.
    Used to update training data and do calculations.
    :param agent: Agent object.
    '''
    self.trainingstep += 1


def end_of_episode(self):
    '''
    When in training mode, called at the end of each episode.
    :param agent: Agent object.
    '''
    finalscoretab = [0,0,0,0,0,0,0,0,0,0,0,+1,0,0,0,+5,0]
    finalscore = 0
    for E in [x[4] for x in self.episodeseq]:
        for e in E:
            finalscore += finalscoretab[e]
    self.logger.debug(f'Final score was: {finalscore}')

    for i in range(len(self.episodeseq)):
        r = self.episodeseq[i][2]
        r += finalscore/10
        self.episodeseq[i][2] = r

    self.model.explay.store_batch(self.episodeseq)
    self.trainingstep += 1
    self.episodeseq = []
