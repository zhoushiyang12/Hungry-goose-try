
import sys
import os

sys.path.append("/kaggle_simulations/agent")
working_dir = "/kaggle_simulations/agent"

# if os.path.exists("sub/model"):
#     model_f = "sub/model"
# elif os.path.exists(os.path.join(working_dir,"model")):
#     model_f = os.path.join(working_dir,"model")
# else:
#     raise ValueError("No model file")


model_f = "2plan_mix/finally"
print(model_f)


import numpy as np
import tensorflow as tf

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

GOOSE = -1.0
RISK = GOOSE/2 # Half of GOOSE (= -0.5)
NONE = 0.0
FOOD = 1.0

act_shape = 4

WIDTH = 11
HEIGHT = 7

xc = WIDTH//2 + 1
yc = HEIGHT//2 + 1

EAST_idx  = (xc+1,yc  )
NORTH_idx = (xc  ,yc-1)
WEST_idx  = (xc-1,yc  )
SOUTH_idx = (xc  ,yc+1)


AROUND = ([xc+1,xc  ,xc-1,xc  ],
          [yc  ,yc-1,yc  ,yc+1])


code2dir = {0:'EAST', 1:'NORTH', 2:'WEST', 3:'SOUTH'}
dir2code = {"EAST":0, "NORTH": 1, "WEST":2, "SOUTH": 3}


policy = tf.keras.models.load_model(model_f, compile=False)
LAST_ACT = "NORTH"

def pos(index):
    return index%WIDTH, index//WIDTH

def centering(z,dz,Z):
    z += dz
    if z < 0:
        z += Z
    elif Z >= Z:
        z -= Z
    return z
    

def encode_board(obs,idx=0):
    """
    Player goose is always set at the center
    """
    global LAST_ACT
    act = LAST_ACT

    board = np.zeros((WIDTH,HEIGHT))

    if len(obs["geese"][idx]) == 0:
        return board
        
    x0, y0 = pos(obs["geese"][idx][0])
    dx = xc - x0
    dy = yc - y0
    
    for goose in obs["geese"]:
        if len(goose) == 0:
            continue

        for g in goose[:-1]:
            x, y = pos(g)
            x = centering(x,dx,WIDTH)
            y = centering(y,dy,HEIGHT)
            board[x,y] = GOOSE
            
        # Tail as Risk
        x, y = pos(goose[-1])
        x = centering(x,dx,WIDTH)
        y = centering(y,dy,HEIGHT)
        board[x,y] = RISK

            
    for food in obs["food"]:
        x, y = pos(food)
        x = centering(x,dx,WIDTH)
        y = centering(y,dy,HEIGHT)
        board[x,y] = FOOD
        
    # Set RISK for around enemy geese head
    for i, goose in enumerate(obs["geese"]):
        if (i == idx) or (len(goose) == 0):
            continue
        x, y = pos(goose[0])
        if (y < HEIGHT-1) and (board[x,y+1] != GOOSE):
            board[x,y+1] += RISK
        if (y > 0) and (board[x,y-1] != GOOSE):
            board[x,y-1] += RISK
        if (x < WIDTH-1) and (board[x+1,y] != GOOSE):
            board[x+1,y] += RISK
        if (x > 0) and (board[x-1,y] != GOOSE):
            board[x-1,y] += RISK
        
    board[xc,yc] = len(obs["geese"][idx]) # self length

    # Avoid Body Hit add psudo GOOSE
    if act == "EAST":
        board[WEST_idx] = GOOSE
    elif act == "NORTH":
        board[SOUTH_idx] = GOOSE
    elif act == "WEST":
        board[EAST_idx] = GOOSE
    elif act == "SOUTH":
        board[NORTH_idx] = GOOSE
    else:
        raise

    return board

from algo import load  
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation
# Input for Neural Network

def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]
    # print('obses',obses)
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
 
list_names = [
        # "simple_toward", 
        # "greedy",
        "risk_averse_greedy",
        # "simple_bfs",
        # "straightforward_bfs",
        # "boilergoose",
        "crazy_goose",
        # "smart_reinforcement_learning",
        ]
obs_list = []
last_action = None   
legend =[ "",'SOUTH', 'NORTH','EAST','WEST']       
def get_action(obs_dict,config_dict):
    global policy
    global LAST_ACT
    global last_action
    obs_list.append(obs_dict)
    idx = Observation(obs_dict).index
    board = encode_board(obs_dict,idx)

    # Plan = tf.squeeze(policy(board.reshape(1,-1))).numpy()
    Plan = int(tf.math.argmax(tf.squeeze(policy(board.reshape(1,-1)))))
 
    algo = load(list_names[Plan] + ".py").agent

    # OK = (board[AROUND] != GOOSE)

    act_name = algo(obs_dict , config_dict ,obs_list ,last_action ) 

    # new_act = 0
    # max_v = -99999
    # for i, (q,ok) in enumerate(zip(Q,OK)):
    #     if (q > max_v) and ok:
    #         new_act = i
    #         max_v = q
    
    last_action = legend.index(act_name)
    LAST_ACT = act_name

    return LAST_ACT
