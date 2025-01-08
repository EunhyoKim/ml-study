import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.

from sklearn.metrics import mean_squared_error        # regression 문제의 모델 성능 측정을 위해서 MSE를 불러온다.

# export PATH=~/opt/anaconda3/envs/silab/bin:$PATH

class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len
    
from sklearn.model_selection import train_test_split


class Regressor(nn.Module):
    def __init__(self, node_num):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(5, node_num, bias=True) # 입력층(5) -> 은닉층1(15)으로 가는 연산
        self.fc2 = nn.Linear(node_num, node_num, bias=True) # 은닉층1(15) -> 은닉층2(10)으로 가는 연산
        self.fc3 = nn.Linear(node_num, 1, bias=True) # 은닉층2(10) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(15개의 20%인 3개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
      
        return x


def evaluation(dataloader):

  predictions = torch.tensor([], dtype=torch.float) 
  actual = torch.tensor([], dtype=torch.float) 

  with torch.no_grad():
    model.eval()
    for data in dataloader:
      inputs, values = data
      outputs = model(inputs)

      predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
      actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.

  predictions = predictions.numpy() # 넘파이 배열로 변경.
  actual = actual.numpy() # 넘파이 배열로 변경.
  rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.

  return rmse

class EarlyStopping:
    def __init__(self,model, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        self.model = model

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    # 모델 저장
                    torch.save(self.model.state_dict(), f'best_model.pth')
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    # 모델 저장
                    torch.save(model.state_dict(), f'best_model.pth')
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_num", type=int, default="15")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wandb.init(project="mlp", config=vars(args))

    df = pd.read_csv("/Users/eunhyokim/Desktop/SILAB/ml-study/df.csv")
    X = df.drop('score', axis = 1).to_numpy()
    y = df['score'].to_numpy().reshape((-1,1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.7, random_state=42, shuffle=True)

    # 학습 데이터, 시험 데이터 배치 형태로 구축하기
    trainsets = TensorData(X_train, Y_train)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=40, shuffle=True)

    testsets = TensorData(X_test, Y_test)
    testloader = torch.utils.data.DataLoader(testsets, batch_size=40, shuffle=False)

    model = Regressor(args.node_num)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-7)

    loss_ = [] 
    n = len(trainloader)
    val_loss = []
    es = EarlyStopping(model)

    for epoch in range(1000):

        running_loss = 0.0 

        for i, data in enumerate(trainloader, 0): 
            model.train()
            inputs, values = data # data에는 X, Y가 들어있다.

            optimizer.zero_grad() # 최적화 초기화.

            outputs = model(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
            loss = criterion(outputs, values) # 손실함수를 계산. error 계산.
            loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
            optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

            running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
        
        loss_.append(running_loss/n) # MSE(Mean Squared Error) 계산
        test_mse = evaluation(testloader)
        val_loss.append(test_mse)

        wandb.log({"train_loss": running_loss/n, "test_loss": test_mse})

        es(test_mse)
        if es.early_stop:
            print(epoch)
            break





