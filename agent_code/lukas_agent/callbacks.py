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
LEARNING_RATE = 1#0.05#0.2 # 0.001

EPSILON_MAX = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.97 #0.96

BUFFER_SIZE = 5000
BATCH_SIZE = 1000#20

# addition
ACTION_SPACE = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])
#ACTION_SPACE = np.array(['LEFT', 'RIGHT'])

# Parameters to controll model
# param = {}
STATE_VERSION = 5
#           max tree leaves | max tree depth | self implemented | #boosted trees to fit | #samples for constr. bin | 
gbm_args = {'num_leaves':500, 'max_depth':10, 'learning_rate':0.1, 'n_estimators':1000,     'subsample_for_bin':20000,
             'class_weight':None}
#           dont need this    

rf_args  = {'n_estimators':1000, 'min_samples_split':2, 'max_features':None, 'bootstrap':True, 'n_jobs':-1, }

def setup(self):    
    if LOGGING: self.logger.info("Run setup code")
    np.random.seed()
    
    # parameters:
    self.train = s.training # training is active
    if LOGGING: self.logger.info("Training-mode is: " + str(self.train))
    
    #self.buffer = deque(maxlen=BUFFER_SIZE)
    self.buffer = Buffer(BUFFER_SIZE)
    self.action_space = ACTION_SPACE
    self.rewards = np.array([])
    self.actions = np.array([])
    self.scores = np.array([])
    self.total_rewards = np.array([])
    
    # initialize model
    self.model = GBM(gbm_args)
    #self.model = RegressionForest() 
    self.isFit = False
    self.load_data = True
    self.step = 0
    
    self.exploration_rate = EPSILON_MAX
    np.save("data/rewards_" + str(self.model.name) + ".npy", self.rewards)
    np.save("data/total_rewards_" + str(self.model.name) + ".npy", self.total_rewards)
    np.save("data/actions_" + str(self.model.name) + ".npy", self.actions)
    np.save("data/scores_" + str(self.model.name) + ".npy", self.scores)
    
    
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

def construct_reduced_state(agent, silent=True):
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
 
    state = np.zeros((2,5,5))
 
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
     
#     others = np.array([(x, y) for (x, y, n, b, s) in agent.game_state['others']])
#     
#     if shp[0] >= 3 and others.size > 0:
#         
#         # projection
#         sectors, dist = get_sector(x, y, filter_targets(x, y, others, 1), size=16)
#         noprint("sectors:", sectors)
#         state[2] = sector_to_state(sectors, distance_to_value(dist), size=16)
#         
#         # inner 3x3 map
#         others_map = np.zeros((s.cols,  s.rows))
#         others_map[tuple(np.array(others).T)] = 1
#         state[2][1:-1, 1:-1] = others_map[mask_y, :][:, mask_x]
         
    return np.atleast_2d(state.flatten())

def select_action(self, state, isFit):
    action_index = 4  # default value 'Wait'
    if isFit:
        q_values = self.model.predict(state)
        if PRINTING: print("q-values according to the model:", q_values, "of shape:", q_values.shape, "for the state:", state)
    else:
        q_values = np.random.random(self.action_space.shape).reshape(1, -1)
        if PRINTING: print("model not fit yet, q_values:", q_values, "of shape:", q_values.shape)
    action_index = np.argmax(q_values[0])
    return action_index

def act(self):
    self.next_action = "WAIT" # default action
    
    #### Only for testing
    arena = self.game_state['arena']
    ####
    self.state = construct_reduced_state(self)
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
        if LOGGING: self.logger.info("construct_reduced_state with isFit = " + str(self.isFit))
        
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
    next_state = construct_reduced_state(self)
    
    
    self.rewards = np.append(self.rewards, reward)
    self.actions = np.append(self.actions, action)
    
    #q_values = get_q_update(self, q_values, action, reward, next_state, terminal)
    
    #               last state | choosen action  | reward of next state | next state             | is state terminal
    self.buffer.store([self.state,  action, reward,  next_state, q_values, terminal])

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
            reward += -2
        elif event == e.INVALID_ACTION:
            reward += -10
        elif event == e.SURVIVED_ROUND:
            reward += 0
    #if PRINTING: print("events:", self.events, "current reward:", reward)
    return reward ######### * GAMMA**self.game_state['step'] # Discounted Reward

def end_of_episode(self): 
    # if the list of states is to short, just skip the evaluation
    if len(self.buffer) < BATCH_SIZE:
        return
    
    batch = self.buffer.sample(int(BATCH_SIZE))
    
    states_fit = []
    q_values_fit = []
    # go through all entries in batch
    for state, action, reward, next_state, q_values, terminal in batch:
        action_id = np.argwhere(self.action_space == action)[0][0]

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
            

        q_values = get_q_update(self, q_values, action, reward, next_state, terminal)
        
        states_fit.append(state[0]) # TODO: not sure about the indices
        q_values_fit.append(q_values[0])
        
    
    #print(np.array(q_values_fit).shape, np.array(states_fit).shape)
    #print(q_values_fit)
    self.model.fit(states_fit, q_values_fit)
    
    self.isFit = True
    self.exploration_rate *= EPSILON_DECAY
    self.exploration_rate = max(EPSILON_MIN, self.exploration_rate)
    
    score = self.game_state["self"][4]
    print(score)
    self.logger.info("Score at end of episode: " + str(score))
    
    # save data
    self.scores = np.append(self.scores, score)
    self.total_rewards = np.append(self.total_rewards, np.sum(self.rewards[-400:]))
    
    if PRINTING: print("######## save data #######")
    np.save("states_" + str(self.model.name) + ".npy", states_fit)
    np.save("q_values_" + str(self.model.name) + ".npy", q_values_fit)              

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
