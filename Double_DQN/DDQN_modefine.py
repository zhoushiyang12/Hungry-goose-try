import torch
# from torch._C import namedtuple_solution_cloned_coefficient
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 256
LR = 0.01
GAMMA = 0.95
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 3))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
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
        # rewards = self._discount_and_norm_rewards(rewards) 
        # rewards = np.ones(len(rewards)) * rewards[-1]
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
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-(NUM_STATES+1):-1])  # Jone
        batch_dones = torch.FloatTensor(batch_memory[:,-1]) # # Jone

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach() 
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * (1-batch_dones).view(-1,1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
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

def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()

    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            # reward = reward_func(env, x, x_dot, theta, theta_dot)
            # ep_reward += reward
            ep_reward += 1
            if done:
                reward = 1
            else:
                reward = 0
            # dqn.store_transition(state, action, reward, next_state, done)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done) 
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                dqn.store_episode(states, actions, rewards, next_states, dones)
                break
            state = next_state
        r = copy.copy(ep_reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        

if __name__ == '__main__':
    main()