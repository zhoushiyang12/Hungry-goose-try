# This is a lightweight ML agent trained by self-play.
# After sharing this notebook,
# we will add Hungry Geese environment in our HandyRL library.
# https://github.com/DeNA/HandyRL
# We hope you enjoy reinforcement learning!


import pickle
import bz2
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env.Hungry_goose import Environment
from Double_DQN.DDQN_H import DQN  


# Neural Network for Hungry Geese

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


# class GeeseNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         layers, filters = 12, 32
#         self.conv0 = TorusConv2d(17, filters, (3, 3), True)
#         self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
#         self.head_p = nn.Linear(filters, 4, bias=False)
#         self.head_v = nn.Linear(filters * 2, 1, bias=False)

#     def forward(self, x):
#         h = F.relu_(self.conv0(x))
#         for block in self.blocks:
#             h = F.relu_(h + block(h))
#         h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
#         h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
#         p = self.head_p(h_head)
#         v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

#         return {'policy': p, 'value': v}


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__() 
        layers, filters = 2, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.conv_m =  TorusConv2d(filters, filters*2, (3, 3), True) 
        self.blocks2 = nn.ModuleList([TorusConv2d(filters*2, filters*2, (3, 3), True) for _ in range(layers)])
        self.conv_l =  TorusConv2d(filters*2, filters*4, (3, 3), True) 
        self.head_p = nn.Linear(filters*4, filters, bias=False)
        self.head_v = nn.Linear(filters*8, filters * 2, bias=False)
        self.head_p_l = nn.Linear(filters, 4, bias=False)
        self.head_v_l = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h = F.relu_(self.conv_m(h))
        for block in self.blocks2:
            h = F.relu_(h + block(h))
        h = F.relu_(self.conv_l(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p_l(self.head_p(h_head))
        v = torch.tanh(self.head_v_l(self.head_v(torch.cat([h_head, h_avg], 1))))

        return {'policy': p, 'value': v}

# Input for Neural Network

def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs['index']) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs['index']) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs['index']) % 4, pos] = 1
            
    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - obs['index']) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    return b.reshape(-1, 7, 11)


# Load PyTorch Model

PARAM = b''

from enum import auto, Enum
class Action(Enum):
    NORTH = auto()
    SOUTH = auto()
    WEST = auto()
    EAST = auto()

    def to_row_col(self):
        if self == Action.NORTH:
            return -1, 0
        if self == Action.SOUTH:
            return 1, 0
        if self == Action.EAST:
            return 0, 1
        if self == Action.WEST:
            return 0, -1
        return 0, 0

    def opposite(self):
        if self == Action.NORTH:
            return Action.SOUTH
        if self == Action.SOUTH:
            return Action.NORTH
        if self == Action.EAST:
            return Action.WEST
        if self == Action.WEST:
            return Action.EAST
        raise TypeError(str(self) + " is not a valid Action.")

# state_dict = pickle.loads(bz2.decompress(base64.b64decode(PARAM)))
# model = GeeseNet()
# model.load_state_dict(state_dict)
# model.eval()
from kaggle_environments.envs.hungry_geese.hungry_geese import adjacent_positions,min_distance,translate

def look_one_step_ahead(observation):
    rows, columns = 7,11

    food = observation.food
    geese = observation.geese
    opponents = [
        goose
        for index, goose in enumerate(geese)
        # if index != observation.index and len(goose) > 0
        if index != observation.index and len(goose) > len(geese[observation.index])
    ]

    # print('len(goose) > len(geese[observation.index]',len(geese[observation.index]))
    # Don't move adjacent to any heads
    head_adjacent_positions = {
        opponent_head_adjacent
        for opponent in opponents
        for opponent_head in [opponent[0]]
        for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
    }
    # Don't move into any bodies
    bodies = {position for goose in geese for position in goose}
    # Move to the closest food
    position = geese[observation.index][0]
    actions = [
        action
        for action in Action
        for new_position in [translate(position, action, columns, rows)]
        if (
            # new_position not in head_adjacent_positions and
            new_position in bodies 
            # (self.last_action is None or action != self.last_action.opposite())
        )
    ]
    h_actions = [
        action
        for action in Action
        for new_position in [translate(position, action, columns, rows)]
        if (
            new_position in head_adjacent_positions 
        )
    ]

    # action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])
    # self.last_action = action
    return actions,h_actions#the direction there is a body

# Main Function of Agent
from random import choice
import sys
# if you have many scripts add this line before you import them
sys.path.append('/kaggle_simulations/agent/') 
from algo.simple_toward import agent as simple_toward
from algo.risk_averse_greedy import agent as risk_averse_greedy
from algo.simple_bfs import agent as simple_bfs
from algo.boilergoose import agent as boilergoose
from algo.crazy_goose import agent as crazy_goose
from algo.straightforward_bfs import agent as straightforward_bfs
from algo.smart_reinforcement_learning import agent as smart_reinforcement_learning
import algo

# algo = [risk_averse_greedy,straightforward_bfs,crazy_goose,smart_reinforcement_learning]

# algo = [risk_averse_greedy,crazy_goose,smart_reinforcement_learning]

algo = [risk_averse_greedy,crazy_goose]
print("algo--------------------------------",algo)
e = Environment()
dqn = DQN(e)
# 模型参数保存
dqn.load('./model/dqn_model_param_{}.pkl'.format(1956)) 
obses = []
last_action = None   
legend =[ "",'SOUTH', 'NORTH','EAST','WEST']
def agent(obs, conf):
    global last_action
    global obses
    obses.append(obs)
    x = make_input(obses)
    # print('x---------------------',x.shape)
    with torch.no_grad():
        # xt = torch.from_numpy(x)#.unsqueeze(0)
        # print('xt---------------------',xt.shape)
        p = dqn.choose_action(x,eval=True)
    act_name = [algo[i](obs,conf,obses,last_action) for i in range(len(algo))]
    print('p---------------------',p,act_name)
    # if p >=3: p = 2
    act_name = act_name[p]#(obs,conf)  
    act_name = algo[p](obs,conf,obses,last_action)
    last_action = legend.index(act_name)
    return act_name

    # act = Action[act_name]
    # # act = Action(np.argmax(p)+1)
    # bodies,h_actions = look_one_step_ahead(obs)
    # if last_action:
    #     if (act != Action[legend[last_action]].opposite() 
    #                     and act not in bodies
    #                     and act not in h_actions) :
    #         action = act 
    #     else:
    #         good_act = [action for action in Action if (action !=Action[legend[last_action]].opposite() and action not in bodies) ]
    #         if good_act:
    #             # print('AAAAAA~',h_actions,good_act)
    #             good_act = list(set(good_act).difference(set(h_actions)))   #差集
    #             print('After Avoid collision~h_actions,good_act,act',h_actions,good_act,act)
    #             if good_act:
    #                 action = choice(good_act)
    #             else:
    #                 action = act
    #                 print('Random choice~','h_actions',h_actions)
    #         else :
    #             action = act
    #             print("No way~ Over!~",act)
    # else:
    #     action = act 
    # last_action = legend.index(action.name)
    # # print(action.name,"---------------------------------")
    # return action.name