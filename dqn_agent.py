import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class AgentDQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(AgentDQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims, dtype=torch.float64)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, dtype=torch.float64)
        self.fc3 = nn.Linear(self.fc2_dims, self.output, dtype=torch.float64)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = state.to(torch.float64)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=10_000, eps_end=0.001,
                 eps_dec=0.002):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.eps_end = eps_end
        self.eps_dec = eps_dec

        self.Q_val = AgentDQN(self.lr, n_actions=n_actions, input_dims=input_dims,
                              fc1_dims=256, fc2_dims=128)

        self.Q_target = AgentDQN(self.lr, n_actions=n_actions, input_dims=input_dims,
                                 fc1_dims=256, fc2_dims=128)

        self.Q_target.load_state_dict(self.Q_val.state_dict())
        self.Q_target.eval()

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float64)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation.clone().detach()).to(self.Q_val.device)
            actions = self.Q_val.forward(state)
            action = torch.argmax(actions).item()
            return action
        else:
            action = np.random.choice(self.action_space)
            return action

    def learn(self):
        global total_loss
        total_loss = 0.0
        if self.mem_cntr < self.batch_size:
            return

        self.Q_val.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.Tensor(self.state_memory[batch]).to(self.Q_val.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_val.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_val.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_val.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_val.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_val.loss(q_target, q_eval).to(self.Q_val.device)
        total_loss = loss.item()
        loss.backward()
        self.Q_val.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

        return total_loss
