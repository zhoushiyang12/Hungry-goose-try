from gym.logger import error
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 1024
LR = 0.0001
GAMMA = 0.80
EPISILO = 0.9
MEMORY_CAPACITY = 200000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = 2#env.action_space.n
NUM_STATES = 17*7*11#env.observation_space.shape[0]
ENV_A_SHAPE = 0  
 

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


class GeeseNet(nn.Module):
    def __init__(self, env, ):
        super().__init__()
        input_shape = env.observation().shape#(16,7,11)
        layers, filters = 2, 32
        self.conv0 = TorusConv2d(input_shape[0], filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.conv_m =  TorusConv2d(filters, filters*2, (3, 3), True) 
        self.blocks2 = nn.ModuleList([TorusConv2d(filters*2, filters*2, (3, 3), True) for _ in range(layers)])
        self.conv_l =  TorusConv2d(filters*2, filters*4, (3, 3), True) 
        self.head_p = nn.Linear(filters*4, filters, bias=False)
        self.head_v = nn.Linear(filters*8, filters * 2, bias=False)
        self.head_p_l = nn.Linear(filters, NUM_ACTIONS, bias=False)
        self.head_v_l = nn.Linear(filters * 2, NUM_ACTIONS, bias=False)

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
        # v = torch.tanh(self.head_v_l(self.head_v(torch.cat([h_head, h_avg], 1))))

        Q = F.relu_((self.head_v_l(F.relu_(self.head_v(torch.cat([h_head, h_avg], 1))))))

        # return {'policy': p, 'value': v}
        return Q

class DQN():
    """docstring for DQN"""
    def __init__(self,env): 
        self.eval_net, self.target_net = GeeseNet(env), GeeseNet(env)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 3))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, eval = False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO or eval:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state, done):
        transition = np.hstack((state, [action, reward], next_state, [done]))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def _discount_and_norm_rewards(self,ep_rs):
        # discount episode rewards
        discounted_ep_rs = []
        reversed_ep_rs = reversed(list(enumerate(ep_rs)))
        running_add = 0
        for i,re_rew in reversed_ep_rs:
            running_add = running_add * GAMMA + re_rew
            discounted_ep_rs.append(running_add )
        discounted_ep_rs = np.array(discounted_ep_rs)
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def store_episode(self, states, actions, rewards, next_states, dones):
        rewards = self._discount_and_norm_rewards(rewards) 
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.store_transition(state, action, reward, next_state, done) 
        
    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).reshape(-1,17,7,11)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-(NUM_STATES+1):-1]).reshape(-1,17,7,11)
        batch_dones = torch.FloatTensor(batch_memory[:,-1])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * (1-batch_dones).view(-1,1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, path= './model/dqn_model_param.pkl'): 
        # 模型参数保存
        try:
            torch.save(self.eval_net.state_dict(), path)
        except:
            print("Model save faild!~")
    def load(self, path= './model/dqn_model_param.pkl'):  
        self.eval_net.load_state_dict( torch.load(path)) 
        print("Model load ok!~") 

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

# def main():
#     dqn = DQN()
#     episodes = 400
#     print("Collecting Experience....")
#     reward_list = []
#     plt.ion()
#     fig, ax = plt.subplots()
#     for i in range(episodes):
#         state = env.reset()
#         ep_reward = 0
#         while True:
#             env.render()
#             action = dqn.choose_action(state)
#             next_state, _ , done, info = env.step(action)
#             x, x_dot, theta, theta_dot = next_state
#             reward = reward_func(env, x, x_dot, theta, theta_dot)
#             if not done:
#                 dqn.store_transition(state, action, reward, next_state, done)
#             ep_reward += 1

#             if dqn.memory_counter >= MEMORY_CAPACITY:
#                 dqn.learn()
#                 if done:
#                     print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
#             if done: 
#                 dqn.store_transition(state, action, reward, next_state, done)
#                 break
#             state = next_state
#         r = copy.copy(reward)
#         reward_list.append(r)
#         ax.set_xlim(0,3000)
#         #ax.cla()
#         ax.plot(reward_list, 'g-', label='total_loss')
#         plt.pause(0.001)
        

# if __name__ == '__main__':
#     main()