
# You need to install kaggle_environments, requests
from kaggle_environments import make
import random
from matplotlib.pyplot import inferno
import numpy as np


from algo import load
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation

class Environment():
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    NUM_AGENTS = 4

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.reset()

        self.list_names = [
        # "simple_toward", 
        # "greedy",
        "risk_averse_greedy",
        # "simple_bfs",
        # "straightforward_bfs",
        # "boilergoose",
        "crazy_goose",
        # "smart_reinforcement_learning",
        ]
        self.algo = [load(n + ".py").agent for n in self.list_names]

        self.obs_list_each = [[] for _ in range (self.NUM_AGENTS)]
        self.last_action_each = [None for _ in range (self.NUM_AGENTS)]
        self.legend =[ "",'SOUTH', 'NORTH','EAST','WEST']

    def reset(self, args={}):
        obs = self.env.reset(num_agents=self.NUM_AGENTS)
        # print('obs',obs)
        self.reset_info((obs, {}))
        #return next_obs
        return [self.observation(n) for n in range(self.NUM_AGENTS)]

    def reset_info(self, info):
        obs, last_actions = info
        self.obs_list = [obs]
        self.obs_list_each= [[{ **obs[0]['observation'],**obs[i]['observation']}] for i in range(4)]
        self.last_action_each = [None for _ in range (self.NUM_AGENTS)]
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_to is None:
            return None
        x_from, y_from = pos_from // 11, pos_from % 11
        x_to, y_to = pos_to // 11, pos_to % 11
        if x_from == x_to:
            if (y_from + 1) % 11 == y_to:
                return 3
            if (y_from - 1) % 11 == y_to:
                return 2
        if y_from == y_to:
            if (x_from + 1) % 7 == x_to:
                return 1
            if (x_from - 1) % 7 == x_to:
                return 0

    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]['observation']
        colors = ['\033[33m', '\033[34m', '\033[32m', '\033[31m']
        color_end = '\033[0m'

        def check_cell(pos):
            for i, geese in enumerate(obs['geese']):
                if pos in geese:
                    if pos == geese[0]:
                        return i, 'h'
                    if pos == geese[-1]:
                        return i, 't'
                    index = geese.index(pos)
                    pos_prev = geese[index - 1] if index > 0 else None
                    pos_next = geese[index + 1] if index < len(geese) - 1 else None
                    directions = [self.direction(pos, pos_prev), self.direction(pos, pos_next)]
                    return i, directions
            if pos in obs['food']:
                return 'f'
            return None

        def cell_string(cell):
            if cell is None:
                return '.'
            elif cell == 'f':
                return 'f'
            else:
                index, directions = cell
                if directions == 'h':
                    return colors[index] + '@' + color_end
                elif directions == 't':
                    return colors[index] + '*' + color_end
                elif max(directions) < 2:
                    return colors[index] + '|' + color_end
                elif min(directions) >= 2:
                    return colors[index] + '-' + color_end
                else:
                    return colors[index] + '+' + color_end

        cell_status = [check_cell(pos) for pos in range(7 * 11)]

        s = 'turn %d\n' % len(self.obs_list)
        for x in range(7):
            for y in range(11):
                pos = x * 11 + y
                s += cell_string(cell_status[pos])
            s += '\n'
        for i, geese in enumerate(obs['geese']):
            s += colors[i] + str(len(geese) or '-') + color_end + ' '
        return s

    def plays(self, actions, typ="int"):
        acts = []
        for p in self.players():
            plan =  actions.get(p, "dead") 
            if plan == "dead":
                act='NORTH'
            else:
                obs = { **self.obs_list[-1][0]['observation'],**self.obs_list[-1][p]['observation']}
                # obs =  Observation(obs)
                # print('obs',obs)
                # print('plan',plan,p)
                algo = self.algo[plan]
                act = algo(obs , self.env.configuration ,self.obs_list_each[p],self.last_action_each[p] )
            self.last_action_each[p] =  self.legend.index(act)
            acts.append(act)
        
        obs = self.env.step(acts)
        # state transition 
        # if typ=="str":
        #     obs = self.env.step(actions)
        # else:
        #     obs = self.env.step([self.action2str(actions.get(p, None) or 0) for p in self.players()])
        self.play_info((obs, actions))

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def play_info(self, info):
        obs, actions = info
        self.obs_list.append(obs)
        for i in range(4): 
            obs_each = { **obs[0]['observation'],**obs[i]['observation']}
            self.obs_list_each[i].append(obs_each) 
        self.last_actions = actions

    def turns(self):
        # players to move
        return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs['status'] == 'ACTIVE':
                return False
        return True

    def outcome(self):#reward
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):# return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        action_map = {'N': Action.NORTH, 'S': Action.SOUTH, 'W': Action.WEST, 'E': Action.EAST}

        agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
        agent.last_action = action_map[self.ACTION[self.last_actions[player]][0]] if player in self.last_actions else None
        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def net(self):
        return 

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, pos] = 1
            # whole position
            for pos in geese:
                b[8 + (p - player) % self.NUM_AGENTS, pos] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                for pos in geese[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

    def step(self, actions, player=None): #action num 0-action_n, ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
        if actions[0] in self.ACTION: 
            self.plays(actions,typ="str")
        else:
            actions = {p:actions[p] for p in self.turns()}
            self.plays(actions,typ="int")

        # check whether done state or not
        # o['observation']['index']: o['reward'] for o in self.obs_list[-1]
        dones = []
        i = 0 
        for obs in self.obs_list[-1]:
            assert obs['observation']['index'] == i
            i +=1
            if obs['status'] == 'ACTIVE':
                dones.append(False)  
            else:
                dones.append(i+10)# avoid 0 ,where 0==False
        info = None
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        # print('rewards',rewards,len(self.obs_list))
        info = max(rewards.values())

        #return next_obses, rewards, done
        return [self.observation(n) for n in range(self.NUM_AGENTS)], self.outcome(),dones,info
        

