import numpy as np
from time import sleep
from collections import deque
import random

# For Testing:
import gym

from settings import s, e # settings and events
from agent_code.lukas_agent_RegressionForest.model import DecisionTree, GBM
from agent_code.lukas_agent_RegressionForest.model import RegressionForest

# constants / hyperparameter
GAMMA = 0.95
LEARNING_RATE = 0.8

EPSILON_MAX = 0.2
EPSILON_MIN = 0.2
EPSILON_DECAY = 0.90 #0.96

BUFFER_SIZE = 1000
BATCH_SIZE = 20

# addition
ACTION_SPACE = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])

def setup(self):    
    np.random.seed()
    self.logger.debug("Run setup code")
    
    # parameters:
    self.train = s.training # training is active
    self.logger.info("Train-mode is: " + str(self.train))
    
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.action_space = ACTION_SPACE
    self.state_space = None 
    
    # initialize model
    self.model = RegressionForest()
    self.isFit = False
    self.reset = False
    self.step = 0
    self.done = False
    
    ##### additional
    self.observations = np.zeros((0,3))
    self.rewards = np.zeros((0,1))
    
    if self.reset == False:
        self.logger.debug('Loading weights')
        observations = np.load('observations.npy')
        rewards = np.load('rewards.npy')
        self.logger.debug('Weights loaded. Training regression model...')
        self.model.fit(observations, rewards.ravel())
        self.logger.debug('Model trained')
       # self.logger.info('Previous highscore: {0}'.format(reward_highscore))
    #####
    
    self.exploration_rate = EPSILON_MAX
    #self.last_action = None
    #self.last_state = None
    
def to_buffer(self, s, a, r, n_s, done):
    self.buffer.append((s, a, r, n_s, done))

def find_best_action(self, observation):
    max_reward = np.NINF
    max_action = 0
    
    for i in range(len(self.action_space)):
        observation[-1] = i
        current_reward = self.model.predict(np.asarray(observation).reshape(1,-1))
        
        if current_reward > max_reward:
            max_action = i
            max_reward = current_reward
    return max_action 

def act(self):
    self.state = get_current_state(self)
    observation = get_current_state(self)
    #self.state = get_reduced_state(self)
    
    self.next_action = "WAIT" # default action
    
    # chose next action
    action_index = 0
    observation.append(0)
    if np.random.rand() < self.exploration_rate:        
        self.logger.info('Pick action at random (full exploration)')
        #self.next_action = np.random.choice(self.action_space)
        action_index =  np.random.randint(len(self.action_space))
        #self.last_state = state
        #print("exploration, with rate:", self.exploration_rate)
    # if model has already been fittet 
    #elif self.isFit == True:
    #    self.logger.info("c")
    #    selfnect_actoin = find_best 
    #    print("##### Fitted q-values:", q_values)
    else:
        self.logger.info("get_current_state with isFit = " + str(self.isFit))
        #q_values = np.zeros(self.action_space.shape).reshape(1, -1)
        action_index = find_best_action(self, observation)
        #print("##### With Fit action_index: " +  str(action_index))
        
    self.next_action = self.action_space[action_index]
    
    # TODO: austauschen 
    observation[-1] = action_index
    self.observations = np.vstack((self.observations, np.asarray(observation)))
        
    #self.next_action = self.action_space[np.argmax(q_values[0])]
    #self.last_state = state
    return

def get_current_state(self):
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
    
    observation = []  
    
    for target in targets:
        if target != None:
            target = (target[0]-x, target[1]-y)
        else:
            target  = (0,0)
            
        observation.append(target[0])
        observation.append(target[1])
    
    return observation
    
    
        
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
        
    state2 = np.zeros((1, 17, 17))
    state2[0] = self.game_state['arena']
    state2[0, self.game_state['self'][0], self.game_state['self'][1]] = 1
    for coin in self.game_state['coins']:
        state2[0, coin[0], coin[1]] = 2
    
        
    #print("so sieht der state aus:", state)
    # turn state into 2-dimensional shap
    #print("shape of state: ", state.shape)
        
    
    #state = state.reshape(3, 17*17)
    state2 = state2.reshape(1, 17*17)
    
    #print("so sieht der geflattete state aus: ", state.shape)
    return state2

def reward_update(self):
    # TODO: integrate into buffer-function
    if self.game_state['step'] == 2: 
        self.rewards = np.vstack((self.rewards,0))
        
    self.step += 1
    next_state = get_current_state(self) # get current state after completing the action
    #next_state = get_reduced_state(self)
    action = self.next_action # the action, that led to the next_state
    reward = compute_reward(self)
    terminal = self.done #TODO: richtig integrieren
    to_buffer(self, self.state, action, reward, next_state, terminal)
    
    if self.isFit == True:
        #print("isFit - run prediction")
        new_state_optimum = self.observations[-1]
        new_state_optimum[-1] = find_best_action(self, self.observations[-1])
        new_state_reward = self.model.predict(np.asarray(new_state_optimum).reshape(1, -1))
        new_state_reward = np.power(GAMMA, self.game_state['step']+1) * new_state_reward
        self.rewards[-1] = (1-LEARNING_RATE)*self.rewards[-1] + LEARNING_RATE*new_state_reward  
    
    self.rewards = np.vstack((self.rewards, reward))  

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
    #self.done = True
    self.model.fit(self.observations, self.rewards.ravel())
    self.isFit = True
    print("######## save data #######")
    np.save("observations.npy", self.observations)
    np.save("rewards.npy", self.rewards)
    
    self.observations = np.zeros((0,3))
    self.rewards = np.zeros((0,1))
                                 
                                    
    #experience_replay(self)
    self.exploration_rate *= EPSILON_DECAY
    self.exploration_rate = max(EPSILON_MIN, self.exploration_rate)
    
    self.done = False
    #train_data = self.model.Dataset('train.svm.txt')
    #train_data.save_binary('train.bin')

def experience_replay(self):
    if len(self.buffer) < BATCH_SIZE:
        return
    batch = random.sample(self.buffer, int(len(self.buffer)/1))
    #print("######## Das hier ist der erste Eintrage des Batch:", batch[0])
    #print("und hier ist er schon zu ende")
    X = []
    targets = []
    for state, action, reward, state_next, terminal in batch:
        q_update = reward
        if not terminal:
            if self.isFit:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                #print("Predict value of next state:", self.model.predict(state_next))
                self.logger.info("not Terminal, isFit: q update: " + str(q_update))
            else:
                q_update = reward
                #print(str(q_update))
                self.logger.info("not Terminal, not isFit: q update: "+ str(q_update))
        if self.isFit:
            q_values = self.model.predict(state)
            self.logger.info("general, isFit: q-values: " + str(q_values))
        else:
            q_values = np.zeros(self.action_space.shape).reshape(1, -1)
            self.logger.info("general, not isFit: q-values: "+ str(q_values))
        action_id = np.argwhere(self.action_space == action)[0][0]
        #print("index of current action", action_id)
        q_values[0][action_id] = q_update
        
        #print(state)
        #print(action)
        #print(q_values)
        X.append(list(state[0]))
        targets.append(q_values[0])   
             
    #print("hier wird gefittet")
    #print("mit X:", X)
    print("unt targets:", targets)
    self.model.fit(X, targets)
    self.isFit = True
    self.exploration_rate *= EPSILON_DECAY
    self.exploration_rate = max(EPSILON_MIN, self.exploration_rate)
 

#def learn(self):
#    pass

def look_for_targets(free_space, start, targets, logger=None):
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

