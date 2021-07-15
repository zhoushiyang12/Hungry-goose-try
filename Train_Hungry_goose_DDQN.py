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

env = Environment() 
dqn = DQN(env)

def main():#TD
    episodes = 10000
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    loss = None
    inf =  None
    for e in range(episodes):
        states = env.reset()  #[4, [16,7,11]
        ep_reward = 0
        dead = []
        rew = -1
        while not env.terminal():
            actions = [dqn.choose_action(state) for state in states]
            next_state, rewards , dones, inf = env.step(actions) 
            for i in env.players():
                if not dones[i] :
                    dqn.store_transition(states[i].reshape(-1) ,actions[i] ,0.01, next_state[i].reshape(-1) ,dones[i] )
                elif dones[i] not in dead:
                    dead.append(dones[i]) 
                    dqn.store_transition(states[i].reshape(-1) ,actions[i] ,rew ,  next_state[i].reshape(-1) ,True )
                    rew +=0.66 

            ep_reward += 1

            if dqn.memory_counter >= 2000:
                loss = dqn.learn()  
            states = next_state
        print("episode: {} , the episode keep {} steps, loss is {} ".format(e, ep_reward,loss))
           
        r = copy.copy(inf)
        reward_list.append(r)
        ax.set_xlim(0,len(reward_list))
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        # 模型参数保存
        dqn.save('./model/dqn_model_param_{}.pkl'.format(e)) 

def main_2():#MC
    episodes = 10000
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    loss = None
    inf =  None
    for e in range(episodes):
        stas = env.reset()  #[4, [16,7,11]
        ep_reward = 0
        dead = []
        rew = -1
        states, actions, rewards, next_states, dones = {p:[] for p in env.players()},{p:[] for p in env.players()},{p:[] for p in env.players()},{p:[] for p in env.players()},{p:[] for p in env.players()}
        
        while not env.terminal():
            acts = [dqn.choose_action(state) for state in stas]
            next_stas, rews , deads, inf = env.step(acts) 
            for i in env.players(): 
                if not deads[i] :         # agent don't dead
                    reward = 0
                    done = False
                elif deads[i] not in dead:# agent  dead
                    dead.append(deads[i]) 
                    reward = rew
                    done = True 
                    rew +=0.66
                else:###fixed bug
                    assert deads[i] in dead
                    continue
                states[i].append(stas[i].reshape(-1))
                actions[i].append(acts[i])
                rewards[i].append(reward)
                next_states[i].append(next_stas[i].reshape(-1))
                dones[i].append(done) 
            ep_reward += 1
            stas = next_stas

        # collect experience
        for i in env.players(): 
            dqn.store_episode(states[i], actions[i], rewards[i], next_states[i], dones[i])
        # train when collect enough trajectroys(above min_buffer_size ) 
        if dqn.memory_counter >= 10000:#Min BUFFER_SIZE to train
            for _ in range(6):
                loss = dqn.learn()  
        print("episode: {} , the episode keep {} steps, loss is {} ".format(e, ep_reward,loss))

        r = copy.copy(inf)
        reward_list.append(r)
        ax.set_xlim(0,len(reward_list))
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        # 模型参数保存
        dqn.save('./model/dqn_model_param_{}.pkl'.format(e)) 

if __name__ == '__main__':

    main_2() 