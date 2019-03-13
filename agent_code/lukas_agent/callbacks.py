import numpy as np
from time import sleep
from collections import deque
import random

# For Testing:
import gym

from settings import s, e # settings and events
from agent_code.lukas_agent.model import DecisionTree, GBM
from agent_code.lukas_agent.model import RegressionForest

# constants / hyperparameter
GAMMA = 0.95
LEARNING_RATE = 0.2#0.2 # 0.001

EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.90 #0.96

BUFFER_SIZE = 1000
BATCH_SIZE = 100#20

# addition
ACTION_SPACE = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])

# Parameters to controll model
# param = {}
STATE_VERSION = 3

def setup(self):    
    self.logger.debug("Run setup code")
    np.random.seed()
    
    # parameters:
    self.train = s.training # training is active
    self.logger.info("Trainvergleich-mode is: " + str(self.train))
    
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.action_space = ACTION_SPACE
    
    # initialize model
    self.model = GBM() #RegressionForest()
    self.isFit = False
    self.load_data = True
    self.step = 0
    
    self.exploration_rate = EPSILON_MAX
    
    if self.load_data == True:
        try:
            states_fit = np.load("states_" + str(self.model.name) + ".npy")
            q_values_fit = np.load("q_values_" + str(self.model.name) + ".npy")
            self.logger.info('loading training data successfully, states size: ' + str(len(states_fit)) + "q_values size:" + str(len(q_values_fit)))
            self.model.fit(states_fit, q_values_fit)
            self.logger.info("Model has been fitted successfully")
            self.isFit = True
        except IOError:
            self.logger.info("No training data found, continue without fit.")
            print("File does not exist, continue without fit.")
        except ValueError:
            print("Data has the wrong format")
        except:
            print("Unexpected error: interrupt kernel")
            raise

def get_current_state(self, state_version = STATE_VERSION):
    if state_version == 1:
        arena = self.game_state['arena']
        x, y, _, bombs_left, score = self.game_state['self']
        bombs = self.game_state['bombs']
        bomb_xys = [(x,y) for (x,y,t) in bombs]
        others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
        coins = self.game_state['coins']
        crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
        ###bomb_map = np.ones(arena.shape) * 5
        ###for xb,yb,t in bombs:
        ###    for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
        ###        if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
        ###            bomb_map[i,j] = min(bomb_map[i,j], t)
        free_space = arena == 0
        targets = []
        targets.append(look_for_targets(free_space, (x,y), coins))
        #countdown = 0
        #for (x,y,t) in bombs:
        #    if (x,y) == targets[-1]:
        #        countdown = t;
        #        break;
        
        state = []  
        
        print("############## targets:", targets)
        
        for target in targets:
            if target != None:
                target = (target[0]-x, target[1]-y)
            else:
                target  = (0,0)
                
            state.append(target[0])
            state.append(target[1])
        
        return np.atleast_2d(state)
    
    if state_version == 2:
        ## construct relevant state:
        #state = np.zeros((1, 3, 17, 17))
        ## write arena into state
        #state[0, 0] = self.game_state['arena']
        ## write position of agent into state
        #state[0, 1, self.game_state['self'][0], self.game_state['self'][1]] = 1
        ## write position of others into state
        #for other in self.game_state['others']:
        #    state[0, 2, other[0], other[1]] = 1
        ## write position of bombs with timer into state
        #for bomb in self.game_state['bombs']:
        #    state[0, 3, bomb[0], bomb[1]] = bomb[2]
        ## write position of explosions into state
        #state[0, 4] = torch.from_numpy(agent.game_state['explosions'])
        ## write position of coins into state
        #for coin in self.game_state['coins']:
        #    state[0, 2, coin[0], coin[1]] = 1
            
        state = np.zeros((1, 17, 17))
        state[0] = arena
        state[0, self.game_state['self'][0], self.game_state['self'][1]] = 1
        for coin in self.game_state['coins']:
            state[0, coin[0], coin[1]] = 2
        
        #state = state.reshape(3, 17*17)
        state = state.reshape(1, 17*17)
    
        #print("so sieht der geflattete state aus: ", state.shape)
        return state
    
    if state_version == 3:
        num_targets = 2
        # agent position:
        x, y, name, bombs_left, score = self.game_state['self']
        # coins position:
        coins = self.game_state['coins']
        
        #free_space = arena == 0
        #targets = []
        #targets.append(look_for_targets(free_space, (x,y), coins))
        
        dist = []
        for coin in coins:
            dist.append(np.sqrt( (x-coin[0])**2 + (y-coin[1])**2 ))        
        sort = np.argsort(np.array(dist))
        
        state = [] 
        
        #print("distance: ", dist)
        #print("sort:", sort)
        rel_coords = coins - np.array([x,y])
        #print("relative coords:", rel_coords)
        coins_state = []
        
        for i in range(num_targets):
            #coins_state.append([dist[sort[i]], rel_coords[sort[i], 0], rel_coords[sort[i], 1]] )
            coins_state.append([rel_coords[sort[i], 0], rel_coords[sort[i], 1]] )
        #print("bs coins state:",coins_state)
        # construct state
        state = np.atleast_2d(np.array([x, y, *coins_state[0], *coins_state[1]]))
        #print("final state:", state)
        #return state
        return np.array(state)

def to_buffer(self, s, a, r, n_s, terminal):
    ''' append new entry to the buffer
        add s: state, a: action, r: reward, n_s: next state, terminal: if the next_state is terminal
    '''
    self.buffer.append((s, a, r, n_s, terminal))

def compute_action(self, state, isFit):
    action_index = 4  # default value 'Wait'
    if isFit:
        q_values = self.model.predict(self.state)
        print("q-values according to the model:", q_values, "of shape:", q_values.shape, "for the state:", state)
    else:
        q_values = np.zeros(self.action_space.shape).reshape(1, -1)
        print("model not fit yet, q_values:", q_values, "of shape:", q_values.shape)
    action_index = np.argmax(q_values[0])
    return action_index

def act(self):
    self.next_action = "WAIT" # default action
    
    #### Only for testing
    arena = self.game_state['arena']
    ####
    
    self.state = get_current_state(self)
    #print("############# der state sieht so aus:", self.state)
    
    # chose next action
    action_index = 4 # default action "Wait"
    # epsilon-greedy
    if np.random.rand() < self.exploration_rate:        
        action_index =  np.random.randint(len(self.action_space))
        self.logger.info("Pick action with exploration, at rate: " + str(self.exploration_rate))
        #print("exploration, with rate:", self.exploration_rate)
    # if model has already been fittet 
    elif self.isFit == True:
        action_index = compute_action(self, self.state, self.isFit)
        self.logger.info("Pick action with exploitation of fitted model")
        print("choosen action is:", self.action_space[action_index], "this action is: valid =", is_valid(self.state, arena, action_index))
    # if model has not been fitted yet
    else:
        action_index = compute_action(self, self.state, self.isFit)        
        self.logger.info("get_current_state with isFit = " + str(self.isFit))
        
    self.next_action = self.action_space[action_index]
    return

def reward_update(self):
    # TODO: integrate into buffer-function
    #if self.game_state['step'] == 2: 
    #    self.rewards = np.vstack((self.rewards,0))
    self.step += 1
    
    terminal = False
    if e.GOT_KILLED in self.events or e.SURVIVED_ROUND in self.events: #check for terminal events
        terminal = True
        
    #               last state | choosen action  | reward of next state | next state             | is state terminal
    to_buffer(self, self.state,  self.next_action, compute_reward(self),  get_current_state(self), terminal)
    
    # TODO: how to save the q-values?
    # Online update rule for 1-step TD-value estimation:
    #if self.isFit == True:
    #    #print("isFit - run prediction")
    #    next_state = self.buffer[-1]
    #    #new_optimal_action[1] = compute_action(self, self.buffer[0], True)
    #    #new_state_reward = self.model.predict(np.asarray(next_state).reshape(1, -1))
    #    n_q_values = self.model.predict(next_state) # predict q-values for the next state
    #    new_state_reward = np.power(GAMMA, self.game_state['step']+1) * new_state_reward
    #    self.rewards[-1] = (1-LEARNING_RATE)*self.rewards[-1] + LEARNING_RATE*new_state_reward  
    
    #self.rewards = np.vstack((self.rewards, reward))  

def compute_reward(self): 
    reward = -1
    for event in self.events:
        if event == e.BOMB_DROPPED:
            reward += 0
        elif event == e.COIN_COLLECTED:
            reward += 100
        elif event == e.CRATE_DESTROYED:
            reward += 10 # 30
        elif event == e.KILLED_SELF:
            reward += 0
        elif event == e.KILLED_OPPONENT:
            reward += 100
        elif event == e.GOT_KILLED:
            reward += -300
        elif event == e.WAITED:
            reward += -5
        elif event == e.INVALID_ACTION:
            reward += -10
        elif event == e.SURVIVED_ROUND:
            reward += 0
    #return reward
    return reward * GAMMA**self.game_state['step'] # Discounted Reward
    
    #print("events:"self.events)
    #print("current reward:", reward)

def end_of_episode(self):
    # if the list of states is to short, just skip the evaluation
    if len(self.buffer) < BATCH_SIZE:
        return
    # otherwise pick a list of samples of the whole buffer size out 
    ######batch = random.sample(self.buffer, int(len(self.buffer)/1)) #original
    batch = random.sample(self.buffer, int(BATCH_SIZE))
    #print("Vergleich: \t erster Eintrag im batch:", batch[0], "\n\t\tund der erste Eintrag im buffer:", self.buffer[0],
    #      "\nund ein vergleich der Laengen: batch:", len(batch), "buffer:", len(self.buffer))
    
    #print("the whole buffer:", self.buffer)
    
    states_fit = []
    q_values_fit = []
    # go through all entries in batch
    for state, action, reward, next_state, terminal in batch:
        
        # calculate Q-update values
        q_update = 0
        if not terminal: # does not matter for terminal states, since the reward is 0 and the next state does not exist
            if self.isFit:
                q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
                #print("Predict value of next state:", self.model.predict(state_next))
                self.logger.info("not terminal, isFit: q update: " + str(q_update))
            # if the model isn't fitted yet
            else:
                q_update = reward 
                #print(str(q_update))
                self.logger.info("not terminal, not isFit: q update: " + str(q_update))
                
        if self.isFit:
            q_values = self.model.predict(state)
            self.logger.info("general, isFit: q-values: " + str(q_values))
        else:
            q_values = np.zeros(self.action_space.shape).reshape(1, -1)
            self.logger.info("general, not isFit: q-values: "+ str(q_values))

        action_id = np.argwhere(self.action_space == action)[0][0] # get the index of the selected action in the action space
        #print("index of current action", action_id)
        
        # change the actual function
        #q_values[0][action_id] = q_update
        q_values[0][action_id] = q_values[0][action_id] + LEARNING_RATE * (q_update - q_values[0][action_id])
        #print("######## state[0]:", state[0], "\n######## q_values[0]", q_values[0])
        states_fit.append(list(state[0])) # TODO: not sure about the indices
        q_values_fit.append(q_values[0])   
    
    self.model.fit(states_fit, q_values_fit)
    
    self.isFit = True
    self.exploration_rate *= EPSILON_DECAY
    self.exploration_rate = max(EPSILON_MIN, self.exploration_rate)
    
    # save data
    
    print("######## save data #######")
    np.save("states_" + str(self.model.name) + ".npy", states_fit)
    np.save("q_values_" + str(self.model.name) + ".npy", q_values_fit)
    
    self.load_data = False # prevent loading data that is already fitted again
    
    #self.model.fit(self.observations, self.rewards.ravel())
    #self.observations = np.zeros((0,3))
    #self.rewards = np.zeros((0,1))                   

def look_for_targets(free_space, start, targets, max_count = 2, logger=None):
    """Find direction of closest target that can be reached via free tiles.
    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.
    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of closest target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    
    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    return best

###### for testing:

def tile_is_free(arena, x, y):
    return (arena[x, y] == 0) #and (bombs[x, y] == 0)

def is_valid(state, arena, i):
    position = state[0,0:2]
    if i == 0:
        return tile_is_free(arena, position[0] + 1, position[1])
    elif i == 1:
        return tile_is_free(arena, position[0] - 1, position[1])
    elif i == 2:
        return tile_is_free(arena, position[0], position[1] - 1)
    elif i == 3:
        return tile_is_free(arena, position[0], position[1] + 1)
    #elif i == 4:
    #    # return state[0, 3][position[0], position[1]] == 0  # check if already a bomb is placed at the current spot
    #    return state[0, 3].sum() == 0
    else:
        return True # wait is always allowed
