import numpy as np
from time import sleep
from collections import deque
import random

# For Testing:
import gym

from settings import s, e # settings and events
from agent_code.lukas_agent.model import RegressionForest, GBM

from agent_code.lukas_agent.model import Buffer

# Logging 
LOGGING = True
DEBUGGING = True
PRINTING = False

# constants / hyperparameter
GAMMA = 0.95
LEARNING_RATE = 0.05#0.2 # 0.001

EPSILON_MAX = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.96 #0.96

BUFFER_SIZE = 1000
BATCH_SIZE = 100#20

# addition
ACTION_SPACE = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])
#ACTION_SPACE = np.array(['LEFT', 'RIGHT'])

# Parameters to controll model
# param = {}
STATE_VERSION = 5
#           max tree leaves | max tree depth | self implemented | #boosted trees to fit | #samples for constr. bin | 
gbm_args = {'num_leaves':31, 'max_depth':-1, 'learning_rate':0.1, 'n_estimators':100,     'subsample_for_bin':20000,
             'class_weight':None}
#           dont need this    

rf_args  = 1

def setup(self):    
    if LOGGING: self.logger.info("Run setup code")
    np.random.seed()
    
    # parameters:
    self.train = s.training # training is active
    if LOGGING: self.logger.info("Training-mode is: " + str(self.train))
    
    #self.buffer = deque(maxlen=BUFFER_SIZE)
    self.buffer = Buffer(BUFFER_SIZE)
    self.action_space = ACTION_SPACE
    
    # initialize model
    self.model = GBM(gbm_args)
    #self.model = RegressionForest() 
    self.isFit = False
    self.load_data = True
    self.step = 0
    
    self.exploration_rate = EPSILON_MAX
    
    if self.load_data == True:
        try:
            states_fit = np.load("states_" + str(self.model.name) + ".npy")
            q_values_fit = np.load("q_values_" + str(self.model.name) + ".npy")
            self.load_data = False # prevent loading data that is already fitted again
            if LOGGING: self.logger.info('loading training data successfully, states size: ' + str(len(states_fit)) + "q_values size:" + str(len(q_values_fit)))
            self.model.fit(states_fit, q_values_fit)
            if LOGGING: self.logger.info("Model has been fitted successfully")
            self.isFit = True
            print("Fitted tree with training data.")
        except IOError:
            if LOGGING: self.logger.info("No training data found, continue without fit.")
            print("File does not exist, continue without fit.")
        except ValueError:
            print("Data has the wrong format")
        except:
            print("Unexpected error: interrupt kernel")
            raise

def get_current_state(self, state_version = STATE_VERSION):
    if state_version == 1:
        arena = self.game_state['arena']
        x, y, name, bombs_left, score = self.game_state['self']
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
            
        state = np.zeros((3, 17, 17))
        state[0] = self.game_state['arena']
        state[1, self.game_state['self'][0], self.game_state['self'][1]] = 1
        for coin in self.game_state['coins']:
            state[2, coin[0], coin[1]] = 1
        
        #state = state.reshape(3, 17*17)
        state = state.reshape(3, 17*17).flatten()
    
        #print("so sieht der geflattete state aus: ", state.shape)
        return np.atleast_2d(state)
    
    if state_version == 3:
        ### agent position:
        x, y, name, bombs_left, score = self.game_state['self']
        
        ### fields around the agent
        arena = self.game_state['arena']
        u = arena[x,y+1]
        r = arena[x+1,y]
        d = arena[x,y-1]
        l = arena[x-1,y]
        
        ### closest 2 coins position:
        coins = self.game_state['coins']
        num_targets = 1
        #if len(coins > 1):
        #    num_targets = 2
        #else: 
        #    num_targets = 1
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
        
        #try:
        for i in range(num_targets):
            #coins_state.append([dist[sort[i]], rel_coords[sort[i], 0], rel_coords[sort[i], 1]] )
            coins_state.append([rel_coords[sort[i], 0], rel_coords[sort[i], 1]] )
            #print("bs coins state:",coins_state)
        #except:
        #    print("################### irgendwas stimmt hier nicht")
        #    print("rel_coords:", rel_coords)
        #    print("sort:", sort)
        #    print("i:", i)
        #    print("coins:", coins)
        #    print("dist:", dist)
        #    raise

        #if num_targets == 1:
        #    coins_state.append([np.PINF,np.PINF])
        
        ### construct state
        state = np.atleast_2d(np.array([x, y, u, r, d, l, *coins_state[0]]))#, *coins_state[1]]))
        #print("state:", state)
        return np.array(state)
    if state_version == 4:
        x, y, name, bombs_left, score = self.game_state['self']
        
        state = np.array([x,y])
        return np.atleast_2d(state)
    
    if state_version == 5:
        ''' 
        Create reduced state tensor from game_state data.
        :param self: Agent object.
        :return: State tensor (on cuda if available).
        '''
        #state = T.zeros(self.reducedstateshape)
        
        state = np.zeros((2,3,3))
        
        x,y,name, bombs_left, score = self.game_state['self']
        
        
        radius = 1
        range = np.arange(s.rows)
        mask_x = np.logical_and(range >= x-radius, range <= x+radius)
        mask_y = np.logical_and(range >= y-radius, range <= y+radius)
    
        #arena = self.game_state['arena'][mask_y, :][:,mask_x]
        state[0] = self.game_state['arena'][mask_y, :][:,mask_x]
        
        coins = self.game_state['coins']
        
        def eucl_distance(dx, dy):
            return np.sqrt(dx**2 + dy**2)
        
        # maybe this works better with arcsin or another convention
        def get_sector(x, y, targets, size = 8): # size = 16 for 5x5 state
            dx = np.atleast_2d(targets)[:,0] - x 
            dy = y - np.atleast_2d(targets)[:,1]
            dist = eucl_distance(dx,dy) 
            #print("sign:", np.sign(dy))
            offset = np.where( np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)) , 0, 2*np.pi)
            sign = np.where( np.logical_or(np.sign(dy) < 0, np.logical_and(np.sign(dy) == 0, np.sign(dx) < 0)) , 1, -1)
            #print("offset:", offset)
            #print("distances:", dx, dist)
            #print('Muenzen', coins)
            #print("angles:", (sign * np.arccos(dx/dist) + offset ) * (360 /(2*np.pi))) 
            sectors = size * ((sign * np.arccos(dx/dist) + offset) / (2*np.pi)) - (1/2)
            sectors = np.where(sectors < 0, sectors + 8, sectors)
            return np.array(sectors, dtype=np.int32), dist
        
        def distance_to_value(dist):
            ''' norm distance to the maximal distance on the board and invert the values
            '''
            #value = 1 - dist/np.sqrt(s.rows**2 + s.cols**2)
            #print("value", value) 
            return 1 - dist/np.sqrt(s.rows**2 + s.cols**2)
        
        def sector_to_state(sectors, values, size=8):
            ''' map the values according to their sectors (ring with #size segments) onto a state with size+1 entries
            '''
            #indice>sector mapping = [[4,5,6], [3,8,7], [2,1,0]]
            indices = [8,7,6,3,0,1,2,5,4]
            state = np.zeros(size+1)
            #print("sectors an stelle 0:", sectors[0])
            try:
                for i in np.arange(sectors.shape[0]):
                    state[indices[sectors[i]]] += values[i]
                return state.reshape(3,3)
            except:
                print("######################")
                print("sector shape:", sectors.shape, "values shape:", values.shape)
        
    
    sectors, dist = get_sector(x, y, coins)
    #print("sectors:", sectors)
    state[1] = sector_to_state(sectors, distance_to_value(dist))
    
    #print("state 0:", state[0])
    #print("state 1:", state[1])

    return np.atleast_2d(state.flatten())


def select_action(self, state, isFit):
    action_index = 4  # default value 'Wait'
    if isFit:
        q_values = self.model.predict(state)
        if PRINTING: print("q-values according to the model:", q_values, "of shape:", q_values.shape, "for the state:", state)
    else:
        q_values = np.zeros(self.action_space.shape).reshape(1, -1)
        if PRINTING: print("model not fit yet, q_values:", q_values, "of shape:", q_values.shape)
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
    if np.random.rand() < self.exploration_rate and self.train:        
        action_index =  np.random.randint(len(self.action_space))
        if LOGGING: self.logger.info("Pick action with exploration, at rate: " + str(self.exploration_rate))
        #print("exploration, with rate:", self.exploration_rate)
    # if model has already been fittet 
    elif self.isFit == True:
        action_index = select_action(self, self.state, self.isFit)
        if LOGGING: self.logger.info("Pick action with exploitation of fitted model")
        if PRINTING: print("choosen action is:", self.action_space[action_index], "this action is: valid =", is_valid(self.state, arena, action_index))
    # if model has not been fitted yet
    else:
        action_index = select_action(self, self.state, self.isFit)        
        if LOGGING: self.logger.info("get_current_state with isFit = " + str(self.isFit))
        
    self.next_action = self.action_space[action_index]
    return

def reward_update(self):
    self.step += 1
    
    terminal = False
    if e.GOT_KILLED in self.events or e.KILLED_SELF in self.events or e.SURVIVED_ROUND in self.events: #check for terminal events
        if PRINTING: print("This is a terminal state")
        terminal = True
    
    q_values = None
    if self.isFit:
        q_values = self.model.predict(self.state)
    else:
        q_values = np.atleast_2d(np.zeros(len(self.action_space)))
    
    reward = compute_reward(self)
    action = self.next_action
    next_state = get_current_state(self)
    
    #q_values = get_q_update(self, q_values, action, reward, next_state, terminal)
    
    #               last state | choosen action  | reward of next state | next state             | is state terminal
    self.buffer.store([self.state,  action, reward,  next_state, q_values, terminal])
    # TODO: how to save the q-values?
    # Online update rule for 1-step TD-value estimation:
    #if self.isFit == True:
    #    #print("isFit - run prediction")
    #    next_state = self.buffer[-1]
    #    #new_optimal_action[1] = select_action(self, self.buffer[0], True)
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
            reward += -300
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
    #if PRINTING: print("events:", self.events, "current reward:", reward)
    return reward ######### * GAMMA**self.game_state['step'] # Discounted Reward

# def compute_reward(self):
#     reward = -1
#     for event in self.events:
#         if event == e.MOVED_RIGHT:
#             reward += 10
#         if event == e.MOVED_DOWN:
#             reward += 10
#     return reward

def end_of_episode(self): 
    # if the list of states is to short, just skip the evaluation
    if len(self.buffer) < BATCH_SIZE:
        return
    # otherwise pick a list of samples of the whole buffer size out 
    ######batch = random.sample(self.buffer, int(len(self.buffer)/1)) #original
    #batch = random.sample(self.buffer, int(BATCH_SIZE))
    
    batch = self.buffer.sample(int(BATCH_SIZE))

    #print("Vergleich: \t erster Eintrag im batch:", batch[0], "\n\t\tund der erste Eintrag im buffer:", self.buffer.memory[0],
    #      "\nund ein vergleich der Laengen: batch:", len(batch), "buffer:", self.buffer.len)
    
    #print("the whole buffer:", self.buffer.memory)
    
    states_fit = []
    q_values_fit = []
    # go through all entries in batch
    for state, action, reward, next_state, q_values, terminal in batch:
        
        #if self.isFit:
        #    q_values = self.model.predict(state)
        #    if DEBUGGING: self.logger.debug("general, isFit: q-values: " + str(q_values))
        #else:
        #    #print("noch nicht gefittet.")
        #    q_values = np.zeros(self.action_space.shape).reshape(1, -1)
        #    #print("q_values: ", q_values)
        #    if DEBUGGING: self.logger.debug("general, not isFit: q-values: "+ str(q_values))

        q_values = get_q_update(self, q_values, action, reward, next_state, terminal)
        
        states_fit.append(state[0]) # TODO: not sure about the indices
        q_values_fit.append(q_values[0])
        
        #print("######## state[0]:", state.shape, "\n######## q_values[0]", q_values.shape)
   
    
    #print(np.array(q_values_fit).shape, np.array(states_fit).shape)
    #print(q_values_fit)
    self.model.fit(states_fit, q_values_fit)
    
    self.isFit = True
    self.exploration_rate *= EPSILON_DECAY
    self.exploration_rate = max(EPSILON_MIN, self.exploration_rate)
    
    # save data
    
    if PRINTING: print("######## save data #######")
    np.save("states_" + str(self.model.name) + ".npy", states_fit)
    np.save("q_values_" + str(self.model.name) + ".npy", q_values_fit)
    
    
    
    #self.model.fit(self.observations, self.rewards.ravel())
    #self.observations = np.zeros((0,3))
    #self.rewards = np.zeros((0,1))                   

def get_q_update(self, q_values, action, reward, next_state, terminal):
    action_id = np.argwhere(self.action_space == action)[0][0] # get the index of the selected action in the action space
    q_update = 0
    if not terminal: # does not matter for terminal states, since the reward is 0 and the next state does not exist
        if self.isFit:
            q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
            #print("Predict value of next state:", self.model.predict(state_next))
            if DEBUGGING: self.logger.debug("not terminal, isFit: q update: " + str(q_update))
        # if the model isn't fitted yet
        else:
            q_update = reward
            #print(str(q_update))
            if DEBUGGING: self.logger.debug("not terminal, not isFit: q update: " + str(q_update))
            
    q_values[0][action_id] = q_values[0][action_id] + LEARNING_RATE * (q_update - q_values[0][action_id])
     
    return q_values    
    

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
    # Determine the first step towards the best found target tile
    current = best
    return best

###### for testing:

def tile_is_free(arena, x, y):
    try:
        bool = (arena[int(x), int(y)] == 0) 
    except:
        print("##### die arena:", arena)
        print("##### die Koordinaten:", x,y)
        raise
    return bool #and (bombs[x, y] == 0)

def is_valid(state, arena, i):
    #print("############################## der state siehr so aus: ", state)
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
