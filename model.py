import torch
from torch import nn, optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.block2(self.block1(x)))
    
    def save(self, file_name='model.pth'):
        model_folder_path = 'Saved Models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()     # mean squared error loss

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)       # state size is (n, x) where n is batch size and x is input size
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)      # convert to long tensor
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:       # this is needed for the first step
            state = torch.unsqueeze(state, dim=0)       # add a dimension to the beginning of the tensor
            next_state = torch.unsqueeze(next_state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            done = (done, )     # tuple
        
        # 1: predicted Q values with current state
        pred = self.model(state)

        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone() -> clone the predicted Q values
        # preds[argmax(action)] = Q_new -> get the predicted Q value of the action that was taken

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new        


        self.optimizer.zero_grad()      # reset the gradients to zero
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()