import torch
import torch.nn as nn
from torch import optim
import numpy as np
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 ---
# 前回の発火抑制設定(閾値2.0, 学習率0.005等)を維持しています
CONFIG = {
    'dt' : 0.25, # 微小時間 (ms)
    'T_r' : 1.0, # 絶対不応期 (ms) -> 1.0ms / 0.25ms = 4 steps 間は再発火不可
    'T_st' : 25,       # 短時間設定 (20ms)
    'tau_j' : 1.0,    # 時定数（電流）
    'tau_m' : 2.0,    # 時定数（電圧）
    'tau_tr' : 3.0,   # 時定数（トレースフィルタ）
    'kappa_j': 0.25,   # 漏れ係数（電流）
    'gamma_m': 1.0,    # 漏れ係数（電圧）: 発火抑制のため強め(1.0->1.5)
    'R_m' : 1.0,       # 抵抗
    'thresh': 0.4,     # 発火閾値（電圧）: 発火抑制のため高め(0.4->2.0)
    'alpha_u' : 0.005, # 学習率: 発火抑制のため低め(0.055->0.005)
    'alpha_gen' : 1e-4,   # 生成的予測誤差係数
    'alpha_disc' : 1.0,   # 識別的予測誤差係数
    'max_freq' : 320.0,   # 高周波数設定 (Hz)
    'batch_size': 64,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Comparison'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# --- ポアソンエンコーディング関数 ---
def generate_poisson_spikes(data, num_steps, config):
    """
    入力データ(0~1)を受け取り、ポアソンスパイク列(Time, Batch, Dim)を生成する
    """
    device = data.device
    batch_size, dim = data.shape
    
    dt = config['dt']
    max_freq = config['max_freq']
    
    # 確率 p = freq(Hz) * dt(ms) / 1000
    firing_probs = data * max_freq * (dt / 1000.0)
    firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
    
    firing_probs_expanded = firing_probs.unsqueeze(0).expand(num_steps, batch_size, dim)
    spikes = torch.bernoulli(firing_probs_expanded)
    
    return spikes


class bPC_SNNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, data=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = data
        
        # 内部状態
        self.v = None
        self.s = None
        self.x = None # Trace
        self.j = None # Input Current State
        self.ref_count = None # 不応期カウンタ (New)
        
        self.e_gen = None
        self.e_disc = None
        self.ref_count = None # 不応期カウンタ

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device) # 不応期カウンタ初期化
        
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device) # 不応期カウンタの初期化

    def update_state(self, total_input_current):
        if self.is_data_layer or self.is_label_layer:
            # データ層・ラベル層は外部スパイクをそのまま使用
            self.s = total_input_current
        else:
            # 隠れ層: LIF Dynamics with Refractory Period
            dt = self.cfg['dt']
            
            # 1. 不応期判定: ref_count > 0 のニューロンは不応期中
            is_refractory = (self.ref_count > 0)

            # 2. 電流ダイナミクス (不応期中でも電流は積分されるのが一般的だが、影響は膜電位リセットで打ち消される)
            d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
            self.j = self.j + (dt / self.cfg['tau_j']) * d_j
            
            # 3. 膜電位ダイナミクス
            d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
            self.v = self.v + (dt / self.cfg['tau_m']) * d_v
            
            # --- 不応期処理 ---
            # 不応期中のニューロンの膜電位を 0 (リセット電位) に固定
            self.v = torch.where(is_refractory, torch.zeros_like(self.v), self.v)
            
            # 4. 発火判定 (不応期でないニューロンのみ発火可能)
            spikes = (self.v > self.cfg['thresh']).float()
            # 念のためマスク(上の行でv=0にしてれば不要だが安全のため)
            spikes = torch.where(is_refractory, torch.zeros_like(spikes), spikes)
            
            self.s = spikes
            
            # 5. 発火後の処理
            # 発火したニューロンは膜電位リセット
            self.v = self.v * (1 - spikes)
            
            # 発火したニューロンの不応期カウンタを設定
            # ref_steps = T_r / dt
            ref_steps = int(self.cfg['T_r'] / dt)
            self.ref_count = torch.where(spikes > 0, torch.tensor(float(ref_steps), device=self.ref_count.device), self.ref_count)
            
            # 6. 不応期カウンタの更新 (カウントダウン)
            self.ref_count = self.ref_count - 1
            self.ref_count = torch.clamp(self.ref_count, min=0)

        # Trace Update
        self.x = self.x - (1 / self.cfg['tau_tr']) * self.x + self.s


class bPC_SNN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes

        # レイヤー生成
        for i, size in enumerate(layer_sizes):
            is_data = (i == 0)
            self.layers.append(bPC_SNNLayer(idx=i, output_dim=size, config=config, data=is_data))
            
        # --- 重み定義 ---
        self.W = nn.ParameterList() # Top-down
        self.V = nn.ParameterList() # Bottom-up
                    
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.5))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.5))

        self.total_synops = 0.0

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=20.0):
        with torch.no_grad():
            for w_list in [self.W, self.V]:
                for w in w_list:
                    norms = w.norm(p=2, dim=1, keepdim=True)
                    mask = norms > max_norm
                    w.data.copy_(torch.where(mask, w * (max_norm / (norms + 1e-8)), w))

    def forward_dynamics(self, x_data_t, y_target_t=None, training_mode=True):
        # === 1. Update Phase ===
        for i, layer in enumerate(self.layers):
            total_input = 0
            
            if i == 0: # Data Layer
                total_input = x_data_t
            
            elif i == len(self.layers) - 1 and layer.is_label_layer and training_mode: # Label Layer (Train)
                total_input = y_target_t

            else: # Hidden or Label (Test)
                total_input += (- layer.e_gen)
                total_input += (- layer.e_disc)

                if i < len(self.layers) - 1:
                    total_input += (- layer.e_gen)

                    e_disc_upper = self.layers[i+1].e_disc
                    total_input += torch.matmul(e_disc_upper, self.V[i]) 
                
                if i > 0:
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

            layer.update_state(total_input)

        # === 2. Prediction & Error Phase ===
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                s_upper = self.layers[i+1].s
                z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                layer.e_gen = alpha_gen * (layer.x - z_gen_pred)

                if training_mode: 
                    self.total_synops += s_upper.sum().item() * layer.dim
            
            if i > 0:
                s_lower = self.layers[i-1].s
                z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                if training_mode:
                    self.total_synops += s_lower.sum().item() * layer.dim

    def manual_weight_update(self):
        alpha_u = self.config['alpha_u']
        with torch.no_grad():
            for i in range(len(self.layers) - 1):
                # W update
                s_upper = self.layers[i+1].s
                e_gen_lower = self.layers[i].e_gen
                grad_W = torch.matmul(e_gen_lower.t(), s_upper)
                self.W[i] += alpha_u * grad_W
                
                # V update
                s_lower = self.layers[i].s
                e_disc_upper = self.layers[i+1].e_disc
                grad_V = torch.matmul(e_disc_upper.t(), s_lower)
                self.V[i] += alpha_u * grad_V


def run_experiment(dataset_name='MNIST'):
    print(f"\n=== Running bPC-SNN on {dataset_name} ===")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        train_d = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    layer_sizes = [784, 500, 500, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=CONFIG).to(CONFIG['device'])
    
    # タイムステップ数の計算 (intキャスト)
    num_steps = int(CONFIG['T_st'] / CONFIG['dt'])
    print(f"Time steps per sample: {num_steps}")

    logs = []

    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_correct = 0
        train_samples = 0
        epoch_start = time.time()
        model.total_synops = 0 
        
        for batch_idx, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            
            targets_one_hot = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
            targets_one_hot.scatter_(1, lbls.view(-1, 1), 1.0)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, num_steps, CONFIG)
            target_spikes = generate_poisson_spikes(targets_one_hot, num_steps, CONFIG)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            sum_out_spikes = 0
            
            for t in range(num_steps):
                x_t = spike_in[t]
                y_t = target_spikes[t]
                
                model.forward_dynamics(x_data_t=x_t, y_target_t=y_t, training_mode=True)
                model.manual_weight_update()
                model.clip_weights(20.0)
                
                if batch_idx % 100 == 0 and t == num_steps - 1:
                    hidden_fire_rate = model.layers[1].s.mean().item()
                    print(f"  Batch {batch_idx} Hidden Fire Rate: {hidden_fire_rate:.4f}")
                
                sum_out_spikes += model.layers[-1].s
            
            _, pred = torch.max(sum_out_spikes, 1)
            train_correct += (pred == lbls).sum().item()
            train_samples += lbls.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Train Acc: {100*train_correct/train_samples:.2f}%")

        train_acc = 100 * train_correct / train_samples
        epoch_synops = model.total_synops
        
        # --- Testing ---
        print("Switching label layer to Inference Mode (LIF)...")
        model.layers[-1].switch_label_mode()

        model.eval()
        test_correct = 0
        test_samples = 0
        
        with torch.no_grad():
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, num_steps, CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                sum_out_spikes = 0
                
                for t in range(num_steps):
                    x_t = spike_in[t]
                    model.forward_dynamics(x_data_t=x_t, y_target_t=None, training_mode=False)
                    sum_out_spikes += model.layers[-1].s
                
                _, pred = torch.max(sum_out_spikes, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)

        test_acc = 100 * test_correct / test_samples
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch} DONE | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s | G-SynOps: {epoch_synops/1e9:.3f}")
        
        print("Switching label layer back to Training Mode (Clamp)...")
        model.layers[-1].switch_label_mode()

        logs.append({
            'dataset': dataset_name,
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': epoch_time,
            'synops': epoch_synops
        })
        
        df = pd.DataFrame(logs)
        df.to_csv(f"{CONFIG['save_dir']}/log_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    run_experiment('MNIST')