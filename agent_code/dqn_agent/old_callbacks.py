
import numpy as np
from time import sleep

import torch as T
import torch.nn as nn
import torch.optim as optim

from agent_code.dqn_agent.dqn_model import DQN, Buffer
from train_settings import s, e


trainingmode = True


### Deciding on actions

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

    def policy(agent):
        output = agent.model(agent.stepstate)[0]
        action = T.argmax(output)
        return action

    if agent.training:
        if explore(agent.game_state['step'], agent.model.eps):
            action = T.tensor(np.random.choice(len(agent.poss_act)))
            print('Exploring: Chose', action)
            return action
        else:
            action = policy(agent)
            print('Using policy: Chose:', action)
            return action
    else:
        return policy(agent)


def select_random_action(agent):
    agent.logger.debug('Selecting random action.')
    return T.tensor(np.random.choice(len(agent.poss_act)))





### Training/Learning

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
        rewardtab = [0, 0, 0, 0, -5, -10, -20, 0, 0, +30, 0, +100, +100, -100, -300, +50, +50]

    # Initialize reward, loop through events, and add up rewards
    reward = -1
    for event in events:
        reward += rewardtab[event]
    return reward

def construct_experience(agent):
    return agent.laststate, T.LongTensor([[agent.lastaction]]), T.tensor([agent.stepreward]).float(), agent.stepstate

def terminal_state():
    pass


############################
#####  Main functions  #####
############################


def setup(self):
    '''
    Called once, before the first round starts. Initialization, in particular of the model
    :param self: Agent object.
    '''
    self.logger.info('Mode: Training' if trainingmode else 'Mode: Testing')
    self.training = trainingmode
    np.random.seed(42)
    self.s, self.e = s, e

    # Adapt state-tensor to current task (bombs, other players, etc)
    self.channels = 3
    self.stateshape = (1, self.channels, s.cols, s.rows)

    # Setup experience replay using cyclic buffer
    self.explay = Buffer(1000, self.stateshape)

    # Create DQN and initialize weights
    self.model = DQN(self)
    self.model.network_setup(self.channels)
    self.model.set_weights(random=True)

    # Create and initialize target DQN
    self.targetmodel = DQN(self)
    self.targetmodel.network_setup(self.channels)
    self.targetmodel.set_weights(random=True)

    # Optimizer
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
    try:
        # Build state tensor
        self.stepstate = construct_state_tensor(self)
        # Choose next action
        self.stepaction = select_action(self)
        print('Action:', self.stepaction)

        if self.training:
            print('Training step ' + str(self.game_state['step']))
            # Check this is first step in episode and initializa sequence and variables
            # Initialize sequences and variables for next round
            if self.game_state['step'] == 1:
                self.seq = []
                # ??? self.prepseq = []
                self.laststate = None
                self.lastaction = None

            # Calculate reward for the events that occurred in this step
            print('Events:', self.events)
            self.stepreward = get_reward(self.events)
            print('Reward:', self.stepreward)

            # Construct and store experience tuple
            if self.lastaction is None:
                print('First step.')
            else:
                s, a, r, n = construct_experience(self)
                self.explay.store([s, a, r, n])

            # Update state and action variables
            self.laststate = self.stepstate
            self.lastaction = self.stepaction


            if self.game_state['step'] % self.model.learninginterval == 0:
                print('Learning step ...')
                self.logger.info('Learning step ...')
                # Sample batch of batchsize from experience replay buffer
                batch = self.explay.sample(self.model.batchsize)

                #! Non final shit from wapu
                nf = T.LongTensor([i for i in range(len(batch.nextstate)) if (batch.nextstate[i] == 0).sum().item() != self.channels*17*17])
                nfnext = batch.nextstate[nf]

                print(batch.state.shape)
                q = self.model(batch.state) # Get q-values from state using the model
                q = q.gather(1, batch.action) # Put together with actions
                print('marker')

                nextq = T.zeros((self.model.batchsize, self.channels)).type(T.FloatTensor)
                nfnextq = self.targetmodel(nfnext)

                nextq.index_copy(0, nf, nfnextq)
                nextq = nextq.max(1)[0]

                # Expected q-values for current state
                expectedq = (nextq * self.model.gamma) + batch.reward
                loss = nn.functional.smooth_l1_loss(q, expectedq)
                self.logger.info('The loss in this learning step was ' + loss)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

                self.model.updates += 1
                if self.model.updates % self.model.targetinterval == 0:
                    self.targetmodel.load_state_dict(self.model.state_dict())

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
    pass

def end_of_episode(self):
    '''
    When in training mode, called at the end of each episode.
    :param agent: Agent object.
    '''
    pass
    # Save parameters / weights to file?
