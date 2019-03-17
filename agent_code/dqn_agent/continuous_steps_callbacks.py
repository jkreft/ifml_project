
import numpy as np
import os
from datetime import datetime

import torch as T
import torch.nn as nn

from agent_code.dqn_agent.dqn_model import DQN, Buffer
from train_settings import s, e



### Flags for choosing in which settings to run ###

training_mode = True
load_from_file = False
analysis = 20



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
        data = T.load(filepath)
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
               + '_interval-' + str(agent.model.analysisinterval) + '.pth'
    T.save({
        'model': agent.model.state_dict(),
        'optimizer': agent.model.optimizer.state_dict(),
        'explay': agent.explay,
        'analysis': agent.analysis,
        'trainingstep': agent.trainingstep
    }, filename)
    print(f'Saved model at step {agent.trainingstep}. Filename: {filename}')


def analysis_data(agent):
    data = {
        'action': agent.stepaction,
        'reward': agent.stepreward,
        'epsilon': agent.model.stepsilon,
        'explored': agent.explored,
        'loss': agent.model.loss
    }
    agent.analysis.append(data)


### Deciding on Actions ###

def construct_state_tensor(agent):
    # Create state tensor from game_state data
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
    return state


def select_action(agent):
    '''
    Selects one of the possible actions based on the networks decision's (or randomly).
    :return: The chosen action {'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'}
    '''
    agent.logger.debug('Entered action selection.')

    def policy(agent):
        input = agent.stepstate[None,:,:,:]
        output = agent.model(input)[0]
        action = T.argmax(output)
        return action

    if agent.training:
        if agent.model.explore('exp'):
            agent.explored = 1
            action = T.tensor(np.random.choice(len(agent.poss_act)))
            agent.logger.debug('Exploring: Chose ' + str(action))
            return action
        else:
            agent.explored = 0
            action = policy(agent)
            agent.logger.debug('Using policy: Chose ' + str(action))
            return action
    else:
        return policy(agent)


def select_random_action(agent):
    agent.logger.debug('Selecting random action.')
    return T.tensor(np.random.choice(len(agent.poss_act)))


### Training/Learning ###

def get_reward(events, rewardtab=None):
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
        # Reward table. The order is as found in e:
        # 'MOVED_LEFT', 'MOVED_RIGHT', 'MOVED_UP', 'MOVED_DOWN', 'WAITED', 'INTERRUPTED', 'INVALID_ACTION', 'BOMB_DROPPED',
        # 'BOMB_EXPLODED','CRATE_DESTROYED', 'COIN_FOUND', 'COIN_COLLECTED', 'KILLED_OPPONENT', 'KILLED_SELF', 'GOT_KILLED',
        # 'OPPONENT_ELIMINATED', 'SURVIVED_ROUND'
        rewardtab = [0, 0, 0, 0, -10, -20, -20, 0, 0, +30, 0, +100, +100, -100, -300, +50, +50]

    # Initialize reward, loop through events, and add up rewards
    reward = -1
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

    # Adapt state-tensor to current task (bombs, other players, etc)
    channels = 3
    self.stateshape = (channels, s.cols, s.rows)

    # Create and setup model and target DQNs
    self.model = DQN(self)
    self.targetmodel = DQN(self)
    self.model.network_setup(channels=self.stateshape[0], analysis=analysis)
    self.targetmodel.network_setup(channels=self.stateshape[0])

    # Load previous status from file or start training from the beginning
    if load_from_file:
        load_model(self)
    else:
        # Setup new experience replay
        self.explay = Buffer(100000, self.stateshape)
        self.modelname = str(datetime.now())[:-7]
        print('Modelname:', self.modelname)

        # Initialize model DQN and target DQN weights randomly
        self.model.set_weights(random=True)
        self.targetmodel.set_weights(random=True)
        print('Initializing model weights randomly.')

        self.trainingstep = 1
        self.analysis = []

    self.model.explay = self.explay
    self.targetmodel.explay = self.explay
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
        # Choose next action
        self.stepaction = select_action(self)

        if self.training:
            if self.trainingstep % 100 == 0:
                print(f'Training step {self.trainingstep}')
            # Check this is first step in episode and initializa sequence and variables
            # Initialize sequences and variables for next round
            if self.game_state['step'] == 1:
                self.seq = []
                # ??? self.prepseq = []
                self.laststate = None
                self.lastaction = None

            # Calculate reward for the events that occurred in this step
            self.stepreward = get_reward(self.events)

            # Construct and store experience tuple
            if self.lastaction is not None:
                s, a, r, n = construct_experience(self)
                self.explay.store([s, a, r, n])

            # Update state and action variables
            self.laststate = self.stepstate
            self.lastaction = self.stepaction


            if self.game_state['step'] % self.model.learninginterval == 0:
                self.logger.info('Learning step ...')
                # Sample batch of batchsize from experience replay buffer
                batch = self.explay.sample(self.model.batchsize)

                #! Non final shit from wapu
                nf = T.LongTensor([i for i in range(len(batch.nextstate)) if (batch.nextstate[i] == 0).sum().item() != np.prod(np.array(self.stateshape))])
                nfnext = batch.nextstate[nf]

                q = self.model(batch.state) # Get q-values from state using the model
                q = q.gather(1, batch.action) # Put together with actions

                nextq = T.zeros((len(batch.nextstate), len(self.poss_act))).type(T.FloatTensor)
                nfnextq = self.targetmodel(nfnext)

                # Make nextq so that it only contains the output for which the input states were non-final
                #nextq = nfnextq[nf]
                nextq.index_copy_(0, nf, nfnextq)
                nextq = nextq.max(1)[0]

                # Expected q-values for current state
                expectedq = (nextq * self.model.gamma) + batch.reward
                self.model.loss = nn.functional.smooth_l1_loss(q, expectedq)
                self.steploss =
                self.logger.info('The loss in this learning step was ' + str(self.model.loss))
                self.model.optimizer.zero_grad()
                self.model.loss.backward()
                self.model.optimizer.step()

                self.model.updates += 1
                if self.model.updates % self.model.targetinterval == 0:
                    self.targetmodel.load_state_dict(self.model.state_dict())

                if self.trainingstep % self.model.analysisinterval == 0:
                    analysis_data(self)

                if self.trainingstep % self.model.saveinterval == 0:
                    save_model(self)
        else:
            print(f'Step {self.game_state["step"]}, chossing action {self.poss_act[self.stepaction.item()]}.')
    except Exception as error:
        print('Exception in act()\n ' + str(error))
        self.stepaction = select_random_action(self)

    self.next_action = self.poss_act[self.stepaction.item()]


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
    self.trainingstep += 1
    # Save parameters / weights to file?
