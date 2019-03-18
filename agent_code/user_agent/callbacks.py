# -*- coding: future_fstrings -*-
# First line is to enable f-strings in python3.5 installation on vserver

import numpy as np
from time import sleep


def setup(agent):
    pass

def act(agent):
    agent.logger.info('Pick action according to pressed key')
    agent.next_action = agent.game_state['user_input']

def reward_update(agent):
    pass

def learn(agent):
    pass
