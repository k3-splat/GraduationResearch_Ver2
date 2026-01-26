import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import math
import numpy as np
import optuna
import copy
import sys
import os
import csv
from datetime import datetime
import pandas as pd

CONFIG = {
    'alpha_gen' : 0,
    'alpha_disc' : 1.0,
    'gamma' : 0.025,
    'alpha' : 0.0005,
    'a' : 1.0,
    'm' : 2.0,
    'n' : 5.0,
    'T' : 75,
    'epochs' : 10,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/bPC_SNN',
}

class Diff_bPCLayer(nn.Module):
    def __init__(self, size, config):
        super().__init__()
        self.size = size
        self.config = config

        self.s_in_gen = None
        self.s_in_disc = None
        self.s_e_gen = None
        self.s_e_disc = None
        self.s_A = None

        self.x_F_gen = None
        self.x_F_disc = None
        self.x_T = None
        self.x_A = None

        self.e_T_gen = None
        self.e_T_disc = None
        self.e_in_gen = None
        self.e_in_disc = None
        self.e_B_gen = None
        self.e_B_disc = None
        self.e_A_gen = None
        self.e_A_disc = None

    def init_state(self, batch_size, device):
        self.s_in_gen = torch.zeros(batch_size, self.size, device=device)
        self.s_in_disc = torch.zeros(batch_size, self.size, device=device)
        self.s_e_gen = torch.zeros(batch_size, self.size, device=device)
        self.s_e_disc = torch.zeros(batch_size, self.size, device=device)
        self.s_A = torch.zeros(batch_size, self.size, device=device)

        self.x_F_gen = torch.zeros(batch_size, self.size, device=device)
        self.x_F_disc = torch.zeros(batch_size, self.size, device=device)
        self.x_T = torch.zeros(batch_size, self.size, device=device)
        self.x_A = torch.zeros(batch_size, self.size, device=device)

        self.e_T_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_T_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_in_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_in_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_B_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_B_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_A_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_A_disc = torch.zeros(batch_size, self.size, device=device)

    def update_state(self, topdown_pred, bottomup_pred, topdown_feedback, bottomup_feedback, gamma, T_theta):
        self.s_in_gen = topdown_pred
        self.s_in_disc = bottomup_pred

        self.x_F_gen += self.s_in_gen * T_theta
        self.x_F_disc += self.s_in_disc * T_theta

        if gamma > 0:
            self.e_T_gen = self.config['alpha_gen'] * (self.x_T - self.x_F_gen)
            self.e_T_disc = self.config['alpha_disc'] * (self.x_T - self.x_F_disc)

        self.x_T += gamma * (-self.e_T_gen - self.e_T_disc + (self.x_T > 0) * (self.e_B_gen + self.e_B_disc))

        if gamma > 0:
            self.e_T_gen = self.config['alpha_gen'] * (self.x_T - self.x_F_gen)
            self.e_T_disc = self.config['alpha_disc'] * (self.x_T - self.x_F_disc)

        self.s_A = torch.sign(self.x_T - self.x_A) * (abs(self.x_T - self.x_A) > T_theta)
        self.s_A *= (self.x_A + self.s_A * T_theta > 0)
        self.x_A += T_theta * self.s_A

        self.e_in_gen = topdown_feedback
        self.e_in_disc = bottomup_feedback
        self.e_B_gen += T_theta * self.e_in_gen
        self.e_B_disc += T_theta * self.e_in_disc

        self.s_e_gen = torch.sign(self.e_T_gen - self.e_A_gen) * (abs(self.e_T_gen - self.e_A_gen) > T_theta)
        self.s_e_disc = torch.sign(self.e_T_disc - self.e_A_disc) * (abs(self.e_T_disc - self.e_A_disc) > T_theta)

        self.e_A_gen += T_theta * self.s_e_gen
        self.e_A_disc += T_theta * self.s_e_disc


class dataLayer(nn.Module):
    def __init__(self, size, config):
        super().__init__()
        self.size = size
        self.config = config

        self.s_in_gen = None
        self.s_e_gen = None
        self.s_A = None

        self.x_F_gen = None
        self.x_T = None
        self.x_A = None

        self.e_T_gen = None
        self.e_in_disc = None
        self.e_B_disc = None
        self.e_A_gen = None

    def init_state(self, batch_size, device):
        self.s_in_gen = torch.zeros(batch_size, self.size, device=device)
        self.s_e_gen = torch.zeros(batch_size, self.size, device=device)
        self.s_A = torch.zeros(batch_size, self.size, device=device)

        self.x_F_gen = torch.zeros(batch_size, self.size, device=device)
        self.x_T = torch.zeros(batch_size, self.size, device=device)
        self.x_A = torch.zeros(batch_size, self.size, device=device)

        self.e_T_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_in_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_B_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_A_gen = torch.zeros(batch_size, self.size, device=device)

    def update_state(self, topdown_pred, bottomup_feedback,  gamma, T_theta):
        self.s_in_gen = topdown_pred
        self.x_F_gen += self.s_in_gen * T_theta

        if gamma > 0:
            self.e_T_gen = self.config['alpha_gen'] * (self.x_T - self.x_F_gen)

        self.s_A = torch.sign(self.x_T - self.x_A) * (abs(self.x_T - self.x_A) > T_theta)
        self.s_A *= (self.x_A + self.s_A * T_theta > 0)
        self.x_A += T_theta * self.s_A

        self.e_in_disc = bottomup_feedback
        self.e_B_disc += T_theta * self.e_in_disc

        self.s_e_gen = torch.sign(self.e_T_gen - self.e_A_gen) * (abs(self.e_T_gen - self.e_A_gen) > T_theta)

        self.e_A_gen += T_theta * self.s_e_gen


class labelLayer(nn.Module):
    def __init__(self, size, config):
        super().__init__()
        self.size = size
        self.config = config

        self.s_in_disc = None
        self.s_e_disc = None
        self.s_A = None

        self.x_F_disc = None
        self.x_T = None
        self.x_A = None

        self.e_T_disc = None
        self.e_in_gen = None
        self.e_B_gen = None
        self.e_A_disc = None

    def init_state(self, batch_size, device):
        self.s_in_disc = torch.zeros(batch_size, self.size, device=device)
        self.s_e_disc = torch.zeros(batch_size, self.size, device=device)
        self.s_A = torch.zeros(batch_size, self.size, device=device)

        self.x_F_disc = torch.zeros(batch_size, self.size, device=device)
        self.x_T = torch.zeros(batch_size, self.size, device=device)
        self.x_A = torch.zeros(batch_size, self.size, device=device)

        self.e_T_disc = torch.zeros(batch_size, self.size, device=device)
        self.e_in_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_B_gen = torch.zeros(batch_size, self.size, device=device)
        self.e_A_disc = torch.zeros(batch_size, self.size, device=device)

    def update_state(self, bottomup_pred, topdown_feedback, gamma, T_theta, is_Training):
        self.s_in_disc = bottomup_pred
        self.x_F_disc += self.s_in_disc * T_theta

        if gamma > 0:
            self.e_T_disc = self.config['alpha_disc'] * (self.x_T - self.x_F_disc)

        if not is_Training:
            self.x_T += gamma * (- self.e_T_disc + (self.x_T > 0) * self.e_B_gen)

            if gamma > 0:
                self.e_T_disc = self.config['alpha_disc'] * (self.x_T - self.x_F_disc)

        self.s_A = torch.sign(self.x_T - self.x_A) * (abs(self.x_T - self.x_A) > T_theta)
        self.s_A *= (self.x_A + self.s_A * T_theta > 0)
        self.x_A += T_theta * self.s_A

        self.e_in_gen = topdown_feedback
        self.e_B_gen += T_theta * self.e_in_gen

        self.s_e_disc = torch.sign(self.e_T_disc - self.e_A_disc) * (abs(self.e_T_disc - self.e_A_disc) > T_theta)
        
        self.e_A_disc += T_theta * self.s_e_disc


class Diff_bPC(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.buffer_s_e = []

        self.a = self.config['a']
        self.m = self.config['m']
        self.n = self.config['n']
        self.T = self.config['T']
        self.gamma = None
        self.T_theta = None

        # レイヤー生成
        for i, size in enumerate(layer_sizes):
            if i == 0:
                self.layers.append(dataLayer(size=size, config=config))
            elif i == len(self.layer_sizes) - 1:
                self.layers.append(labelLayer(size=size, config=config))
            else:
                self.layers.append(Diff_bPCLayer(size=size, config=config))
            
        # --- 重み定義 ---
        self.W = nn.ParameterList() # Top-down (Upper -> Lower) [Generative]
        self.V = nn.ParameterList() # Bottom-up (Lower -> Upper) [Discriminative]
                    
        # Layer[i] <-> Layer[i+1]
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            # Xavier Uniform or Small Random
            w = nn.Parameter(torch.empty(dim_lower, dim_upper))
            v = nn.Parameter(torch.empty(dim_upper, dim_lower))
            
            # Xavier Initialization
            nn.init.xavier_uniform_(w)
            nn.init.xavier_uniform_(v)
            
            self.W.append(w)
            self.V.append(v)

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def feedfoward_initialize(self, x_data, y_target=None):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.x_T = x_data
                current = torch.relu(x_data)
            elif i < len(self.layers) - 1:
                layer.x_T = torch.matmul(current, self.V[i-1].t())
                current = torch.relu(layer.x_T)
            else:
                if y_target is not None:
                    layer.x_T = y_target
                else:
                    layer.x_T = torch.matmul(current, self.V[i-1].t())

    def gamma_cycle(self, t):
        dt = 1 - (1 - self.a * (t % self.T) / self.T)
        self.T_theta = dt * 2 ** self.m / 2 ** (t % self.n)
        if t % self.n == 0:
            self.gamma = self.config['gamma']
        else:
            self.gamma = 0

    def forward_dynamics(self, t, is_Training=True):
        self.gamma_cycle(t)

        buffer_s_A = []
        buffer_s_e_gen = []
        buffer_s_e_disc = []

        for i, layer in enumerate(self.layers):
            buffer_s_A.append(layer.s_A)
            if i < len(self.layers) - 1:
                buffer_s_e_gen.append(layer.s_e_gen)
            if i > 0:
                buffer_s_e_disc.append(layer.s_e_disc)

        for i, layer in enumerate(self.layers):
            if i == 0:
                topdown_spike = buffer_s_A[i+1]
                topdown_pred = torch.matmul(topdown_spike, self.W[i].t())

                bottomup_feedspike = buffer_s_e_disc[i]
                bottomup_feedback = torch.matmul(bottomup_feedspike, self.V[i])

                layer.update_state(topdown_pred, bottomup_feedback, self.gamma, self.T_theta)

            elif i < len(self.layers) - 1:
                topdown_spike = buffer_s_A[i+1]
                topdown_pred = torch.matmul(topdown_spike, self.W[i].t())

                topdown_feedspike = buffer_s_e_gen[i-1]
                topdown_feedback = torch.matmul(topdown_feedspike, self.W[i-1])

                bottomup_spike = buffer_s_A[i-1]
                bottomup_pred = torch.matmul(bottomup_spike, self.V[i-1].t())

                bottomup_feedspike = buffer_s_e_disc[i]
                bottomup_feedback = torch.matmul(bottomup_feedspike, self.V[i])

                layer.update_state(topdown_pred, bottomup_pred, topdown_feedback, bottomup_feedback, self.gamma, self.T_theta)
            
            else:
                topdown_feedspike = buffer_s_e_gen[i-1]
                topdown_feedback = torch.matmul(topdown_feedspike, self.W[i-1])

                bottomup_spike = buffer_s_A[i-1]
                bottomup_pred = torch.matmul(bottomup_spike, self.V[i-1].t())

                layer.update_state(bottomup_pred, topdown_feedback, self.gamma, self.T_theta, is_Training)

    def weight_update(self, x_data, y_target=None):
        for i in range(len(self.layers)):
            # Vの更新 (Discriminative weights)
            if i < len(self.layers) - 1:
                e_T_disc = self.layers[i+1].e_T_disc
                if i == 0:
                    batch_grad_V = torch.matmul(e_T_disc.unsqueeze(2), x_data.unsqueeze(1))
                else:
                    x_T = self.layers[i].x_T
                    phi = torch.relu(x_T)
                    batch_grad_V = torch.matmul(e_T_disc.unsqueeze(2), phi.unsqueeze(1))

                final_grad_V = - batch_grad_V.mean(dim=0)

                if self.V[i].grad is None:
                    self.V[i].grad = final_grad_V
                else:
                    self.V[i].grad += final_grad_V

            # --- Wの更新 (Generative: Upper -> Lower) ---
            # W[i-1] は Layer[i-1] と Layer[i] を接続
            if i > 0:
                e_T_gen = self.layers[i-1].e_T_gen # 下層のエラー
                
                if i == len(self.layers) - 1:
                    # 最上層(ラベル層)が入力となる場合
                    if y_target is not None:
                         # (Batch, Lower, 1) x (Batch, 1, Upper) -> (Batch, Lower, Upper)
                        batch_grad_W = torch.matmul(e_T_gen.unsqueeze(2), y_target.unsqueeze(1))
                    else:
                        # 推論時などでTargetがない場合はx_Tを使う（通常ここには来ない設計）
                        x_T = self.layers[i].x_T
                        phi = torch.relu(x_T)
                        batch_grad_W = torch.matmul(e_T_gen.unsqueeze(2), phi.unsqueeze(1))

                else:
                    # ★修正: inputは layers[i+1] ではなく layers[i] (現在の上層)
                    x_T = self.layers[i].x_T
                    phi = torch.relu(x_T)
                    batch_grad_W = torch.matmul(e_T_gen.unsqueeze(2), phi.unsqueeze(1))

                final_grad_W = - batch_grad_W.mean(dim=0)

                # ★修正: リストではなく個別のパラメータ(W[i-1])にアクセスする
                if self.W[i-1].grad is None:
                    self.W[i-1].grad = final_grad_W
                else:
                    self.W[i-1].grad += final_grad_W


def run_experiment(dataset_name='MNIST'):
    print(f"\n=== Running bPC-SNN on {dataset_name} ===")
    
    # --- ファイル名と実行日時設定 ---
    try:
        # スクリプトファイル名を取得 (拡張子なし)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        script_name = "notebook_execution"

    # 日時を取得
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存パスの決定: log_[スクリプト名]_[データセット]_[日時].csv
    save_file_path = f"{CONFIG['save_dir']}/log_{script_name}_{dataset_name}_{timestamp}.csv"
    print(f"Results will be saved to: {save_file_path}")
    # ----------------------------

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        #transforms.Normalize((0.1307,), (0.3081,)),
        
        # ★ここに追加: データを平坦化する処理
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # テスト用も同様に
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        
        # ★ここに追加
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError("Unknown dataset")

    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    # モデル構築
    layer_sizes = [784, 500, 10]
    model = Diff_bPC(layer_sizes=layer_sizes, config=CONFIG).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['alpha'])

    steps = int(CONFIG['T'])
    logs = []

    optimizer.zero_grad()
    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        epoch_start = time.time()

        for batch_idx, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])

            # ターゲット生成
            targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)

            # 入力スパイク生成
            imgs_rate = torch.clamp(imgs, 0, 1)

            # 1. バッチ開始時に勾配をリセット
            optimizer.zero_grad()
            
            # 状態のリセット
            model.reset_state(imgs.size(0), CONFIG['device'])
            model.feedfoward_initialize(imgs_rate, targets)
            # 時間ステップごとの処理
            for t in range(steps):
                # 順伝播ダイナミクス
                model.forward_dynamics(t, is_Training=True) # 引数修正: x_data等は内部で処理される構造ならtのみ、または適宜引数を合わせる
                
                # 勾配の手動計算と蓄積
                # この関数内で self.W.grad += ... が行われる
                # 定義コード[1]に合わせてメソッド名を weight_update と仮定
            model.weight_update(x_data=imgs, y_target=targets) 

            # 2. 全タイムステップ終了後、蓄積された勾配を使ってパラメータ更新
            optimizer.step()

            # ログ表示など
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}")
    
        # --- Testing ---

        model.eval()
        test_correct = 0
        test_samples = 0
        
        for imgs, lbls in test_l:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            imgs_rate = torch.clamp(imgs, 0, 1)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            model.feedfoward_initialize(imgs_rate)
            
            for t in range(steps):
                model.forward_dynamics(t, is_Training=False)
            
            _, pred = torch.max(model.layers[-1].x_T, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            print(model.layers[-1].x_T)
            
        test_acc = 100 * test_correct / test_samples
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch} DONE | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")

        logs.append({
            'dataset': dataset_name,
            'epoch': epoch,
            'test_acc': test_acc,
            'time': epoch_time
        })
        
        # --- ログ保存 (ファイル名はループの外で決定済み) ---
        df = pd.DataFrame(logs)
        df.to_csv(save_file_path, index=False)

if __name__ == "__main__":
    run_experiment('MNIST')