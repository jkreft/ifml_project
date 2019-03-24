# -*- coding: future_fstrings -*-
# First line is to enable f-strings in python3.5 installation on servers

import numpy as np
from datetime import datetime
import os
import sys
import torch as T
import torch.nn as nn
from settings import s, e
from agent_code.simple_agent import callbacks as rolemodel
from agent_code.dqn_agent.supports import construct_state_tensor, construct_reduced_state_tensor,\
    construct_time_state_tensor, load_model, save_model, step_analysis_data, average_analysis_data, analysisbuffer



### Flags for choosing in which settings to run ###
resume_training = False
training_mode = False if s.gui else True
load_from_file = resume_training if training_mode else True
max_trainingsteps = 1200000
analysis_interval = 1000
save_interval = 500000
start_learning = 0
replay_buffer_size = 400000
feature_reduction = False

if feature_reduction:
    from agent_code.dqn_agent.dqn_model_reduced import DQN, Buffer
else:
    from agent_code.dqn_agent.dqn_model_google import DQN, Buffer


########################################################################################################################
#####                                              Action Functions                                                #####
########################################################################################################################


### Deciding on Actions ###

def construct_state(agent):
    '''
    Create state tensor from game_state data.
    :param agent: Agent object.
    :return: State tensor (on cuda if available).
    '''
    return construct_reduced_state_tensor(agent) if feature_reduction else construct_time_state_tensor(agent)


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

    def action(agent, string):
        if string == 'policy':
            a = policy(agent)
            agent.logger.debug('Using policy: Chose ' + str(a))
        if string == 'random':
            a = T.tensor(np.random.choice(len(agent.possibleact)))
            agent.logger.debug('Exploring: Chose ' + str(a))
        if string == 'rolemodel':
            rolemodel.act(agent)
            a = T.tensor(agent.possibleact.index(agent.next_action))
            agent.logger.debug('Using rolemodel: Chose ' + str(a))
        return a

    if agent.training:
        agent.exploration = agent.model.explore()
        action = action(agent, agent.exploration)
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
        rewardtab = [0, 0, 0, 0, -0.1, 0, -10, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0] # coins
        #rewardtab = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # down

    # Initialize reward, loop through events, and add up rewards
    reward = -5
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
    self.homedir = os.path.expanduser("~") + '/'
    modestr = 'TRAINING' if self.training else 'TEST'
    print(f'Running in {modestr} mode')
    self.logger.info(f'Mode: {modestr}')
    self.s, self.e = s, e
    self.analysisbuffer = analysisbuffer()
    self.startlearning = start_learning
    self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    self.cuda = T.cuda.is_available()
    print(f'Cuda is{"" if self.cuda else " not"} available.')
    self.logger.info(f'Cuda is{"" if self.cuda else " not"} available.')
    # Adapt state-tensor to current task (bombs, other players, etc)
    historylen = 4
    channels = 3
    self.stateshape = (2, 9, 9) if feature_reduction else (historylen, s.cols, s.rows)

    # Create and setup model and target DQNs
    self.model = DQN(self)
    self.targetmodel = DQN(self)
    self.model.network_setup(aint=analysis_interval, sint=save_interval)
    self.targetmodel.network_setup()
    # Put DQNs on cuda if available
    self.model, self.targetmodel = self.model.to(self.device), self.targetmodel.to(self.device)
    # Load previous status from file or start training from the beginning
    if load_from_file:
        load_model(self, trainingmode=training_mode)
    else:
        # Setup new experience replay
        self.explay = Buffer(replay_buffer_size, self.stateshape, device=self.device)
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

    self.model.explay = self.explay
    self.targetmodel.explay = self.explay
    self.episodeseq = []

    self.plotloss = T.zeros(1)
    self.stepq = T.zeros((1, self.model.batchsize))
    self.laststate = T.zeros(self.stateshape).to(self.device)
    self.lastaction = None
    self.lastevents = None
    self.finalscore = 0

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
        self.stepstate = construct_state(self)

        if not self.training:
            # Choose next action
            self.stepaction = select_action(self)
            print(f'Step {self.game_state["step"]}, choosing action {self.possibleact[self.stepaction.item()]}.')

        else:
            t = self.trainingstep
            if (t % 1000 == 0) or (t < 101 and t % 10 == 0) or (t < 1001 and t % 100 == 0):
                print(f'Training step {t}')

            # Choose next action
            self.stepaction = select_action(self, rolemodel=rolemodel)
            #print('marker1')

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
                                       np.array(self.stateshape))]).to(self.device)
                nfnextstate = batch.nextstate[nf]

                if T.cuda.is_available():
                    batch.state = batch.state.cuda()
                    batch.action = batch.action.cuda()
                    nfnextstate = nfnextstate.cuda()
                #print('marker0')
                self.stepq = self.model(batch.state) # Get q-values from state using the model
                self.stepq = self.stepq.gather(1, batch.action) # Put together with actions
                nextq = T.zeros((len(batch.nextstate), len(self.possibleact))).to(self.device)
                nfnextq = self.targetmodel(nfnextstate).to(self.device)

                ##### Version without double-Q-learning #####
                #nfnextq = self.model(nfnextstate).to(self.device)

                # Let nextq only contain the output for which the input states were non-final
                nextq.index_copy_(0, nf, nfnextq)
                nextq = nextq.max(1)[0]

                # Expected q-values for current state
                expectedq = (batch.reward + (nextq * self.model.gamma)).to(self.device)
                self.steploss = self.model.loss(expectedq, self.stepq)
                self.plotloss = self.steploss.cpu()
                batch.state = batch.state.cpu()
                batch.action = batch.action.cpu()

                self.logger.info('The loss in this learning step was ' + str(self.steploss))
                self.model.optimizer.zero_grad()
                self.steploss.backward()
                self.model.optimizer.step()

                if self.model.learningstep % self.model.targetinterval == 0:
                    self.targetmodel.load_state_dict(self.model.state_dict())
                self.model.learningstep += 1

            # If an analysisinterval is set, average over the interval and save data
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
    #finalscore = 0
    self.finalscore = self.game_state['self'][4] * 100
    self.logger.info(f'Final score was: {self.finalscore}')
    print(f'Final score was: {self.finalscore}')

    for i in range(len(self.episodeseq)):
        r = self.episodeseq[i][2]
        r += self.finalscore/5
        self.episodeseq[i][2] = r

    self.model.explay.store_batch(self.episodeseq)
    self.trainingstep += 1
    self.episodeseq = []

    if self.trainingstep > max_trainingsteps:
        save_model(self)
        if not os.path.exists(self.homedir + 'traininglog'):
            os.mkdir(self.homedir + 'traininglog/')
            with open(self.homedir + 'traininglog/' + self.modelname + '_reached_maxstep.info', 'w') as info:
                info.write('\n' + self.modelname + ' has reached maximum training steps:\n' + str(max_trainingsteps))