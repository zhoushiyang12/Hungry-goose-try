import collections

import kaggle_environments
from kaggle_environments import evaluate, make, utils
from env.Hungry_goose import Environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from Double_DQN.DDQN_H import DQN  
import copy
import random
import torch
kaggle_environments.__version__

list_names = [
    "simple_toward", 
    "greedy",
    "risk_averse_greedy",
    "simple_bfs",
    "straightforward_bfs",
    "boilergoose",
    "crazy_goose",
    "smart_reinforcement_learning",
]

list_agents = [agent_name + ".py" for agent_name in list_names]

def test():
    e = Environment()
    dqn = DQN(e)
    # 模型参数保存
    dqn.load('./model_mix/dqn_model_param_{}.pkl'.format(2500)) 
    for _ in range(1):
        states = e.reset()
        while not e.terminal():
            print(e)  
            actions = [dqn.choose_action(state) for state in states]
            print(actions)
            next_state, rewards , dones,_ = e.step(actions) 
            # next_state, rewards , dones = e.step({p: dqn.choose_action(states[p]) for p in e.turns()}) 
            states = next_state 
        print(e)
        print(e.outcome())
if __name__ == '__main__':  
    test()