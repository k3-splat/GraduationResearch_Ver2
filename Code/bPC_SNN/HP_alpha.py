import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import time
import pandas as pd
import os
from datetime import datetime
import optuna  # 追加

# --- 基本設定 ---
# 最適化対象以外の固定パラメータやデフォルト設定
BASE_CONFIG = {
    'dt' : 0.25,
    'T_st' : 100.0,
    'tau_j' : 10.0,
    'tau_m' : 20.0,
    'tau_tr' : 30.0,
    'tau_data' : 30.0,
    'kappa_j': 0.25,
    'gamma_m': 1.0,
    'R_m' : 1.0,
    # 'alpha_u', 'alpha_gen', 'alpha_disc' はOptunaによって決定されるためここでは除外
    'thresh': 0.4,
    'batch_size': 64,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/bPC_SNN_Optuna',
    'max_freq': 2000.0
}

os.makedirs(BASE_CONFIG['save_dir'], exist_ok=True)

def generate_poisson_spikes(data, num_steps, config):
    batch_size, dim = data.shape
    dt = config['dt']
    max_freq = config['max_freq']
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
        
        self.v = None
        self.s = None
        self.x = None 
        self.j = None 
        self.e_gen = None
        self.e_disc = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        dt = self.cfg['dt']
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        self.v = self.v + (dt / self.cfg['tau_m']) * d_v
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        self.v = self.v * (1 - spikes) 
        if not self.is_data_layer:
            self.x = self.x + (-self.x / self.cfg['tau_tr'] + spikes)

class label_layer(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.dim = output_dim
        self.cfg = config
        self.is_training = True

        self.v_tmp = None
        self.s_tmp = None
        self.j_tmp = None
        self.v = None
        self.s = None
        self.x = None 
        self.j = None 
        self.e_disc = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

        if not self.is_training:
            self.v_tmp = torch.zeros(batch_size, self.dim, device=device)
            self.s_tmp = torch.zeros(batch_size, self.dim, device=device)
            self.j_tmp = torch.zeros(batch_size, self.dim, device=device)

    def switch2training_mode(self):
        self.is_training = True

    def switch2testing_mode(self):
        self.is_training = False

    def update_state(self, input2main, input2tmp=None):
        dt = self.cfg['dt']

        if not self.is_training:
            d_j_tmp = (-self.cfg['kappa_j'] * self.j_tmp + input2tmp)
            self.j_tmp = self.j_tmp + (dt / self.cfg['tau_j']) * d_j_tmp
            d_v_tmp = (-self.cfg['gamma_m'] * self.v_tmp + self.cfg['R_m'] * self.j_tmp)
            self.v_tmp = self.v_tmp + (dt / self.cfg['tau_m']) * d_v_tmp
            spikes_tmp = (self.v_tmp > self.cfg['thresh']).float()
            self.s_tmp = spikes_tmp
            self.v_tmp = self.v_tmp * (1 - spikes_tmp)
        
        d_j = (-self.cfg['kappa_j'] * self.j + input2main)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        self.v = self.v + (dt / self.cfg['tau_m']) * d_v
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        self.v = self.v * (1 - spikes)

class bPC_SNN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes

        for i, size in enumerate(layer_sizes):
            if i == len(layer_sizes) - 1:
                self.layers.append(label_layer(output_dim=size, config=config))
            else:
                is_data = (i == 0)
                self.layers.append(bPC_SNNLayer(idx=i, output_dim=size, config=config, data=is_data))
            
        self.W = nn.ParameterList() 
        self.V = nn.ParameterList() 
                    
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.5))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.5))

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=20.0):
        for w_list in [self.W, self.V]:
            for w in w_list:
                norms = w.norm(p=2, dim=1, keepdim=True)
                mask = norms > max_norm
                w.data.copy_(torch.where(mask, w * (max_norm / (norms + 1e-8)), w))

    def forward_dynamics(self, x_data, y_target=None):
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        if y_target is not None:
            for i, layer in enumerate(self.layers):
                total_input = 0
                if i == 0:
                    s_upper = self.layers[i+1].s
                    total_input += torch.matmul(s_upper, self.W[i].t())
                elif i < len(self.layers) - 1:
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])
                    total_input += (- layer.e_gen)
                    e_disc_upper = self.layers[i+1].e_disc
                    total_input += torch.matmul(e_disc_upper, self.V[i])
                else:
                    s_lower = self.layers[i-1].s
                    total_input += torch.matmul(s_lower, self.V[i-1].t())
                layer.update_state(total_input)

            for i, layer in enumerate(self.layers):
                if i == 0:
                    s_own = self.layers[i].s
                    layer.e_gen = alpha_gen * (x_data - s_own)
                elif i < len(self.layers) - 1:
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        s_lower = self.layers[i-1].s
                        z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)
                    if i < len(self.layers) - 2:
                        s_upper = self.layers[i+1].s
                        z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                    else:
                        z_gen_label = torch.matmul(y_target, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_label)
                else:
                    s_own = self.layers[i].s
                    layer.e_disc = alpha_disc * (y_target - s_own)
        else:
            for i, layer in enumerate(self.layers):
                total_input = 0
                if i == 0:
                    s_upper = self.layers[i+1].s
                    total_input += torch.matmul(s_upper, self.W[i].t())
                    layer.update_state(total_input)
                elif i < len(self.layers) - 1:
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])
                    total_input += (- layer.e_gen)
                    e_disc_upper = self.layers[i+1].e_disc
                    total_input += torch.matmul(e_disc_upper, self.V[i])
                    layer.update_state(total_input)
                else:
                    input2main = 0
                    input2tmp = 0
                    input2main += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    input2main += torch.matmul(e_gen_lower, self.W[i-1])
                    s_lower = self.layers[i-1].s
                    input2tmp += torch.matmul(s_lower, self.V[i-1].t())
                    layer.update_state(input2main, input2tmp)

            for i, layer in enumerate(self.layers):
                if i == 0:
                    s_own = self.layers[i].s
                    layer.e_gen = alpha_gen * (x_data - s_own)
                else:
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    elif i < len(self.layers) - 1:
                        s_lower = self.layers[i-1].s
                        z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)
                        s_upper = self.layers[i+1].s
                        z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                    else:
                        s_tmp = layer.s_tmp
                        s_own = layer.s
                        layer.e_disc = alpha_disc * (s_own - s_tmp)

    def manual_weight_update(self, x_data, y_target=None):
        alpha_u = self.config['alpha_u']
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                e_disc_upper = self.layers[i+1].e_disc
                if i == 0:
                    grad_V = torch.matmul(e_disc_upper.t(), x_data)
                else:
                    s_own = self.layers[i].s
                    grad_V = torch.matmul(e_disc_upper.t(), s_own)
                self.V[i] += alpha_u * grad_V

            if i > 0:
                e_gen_lower = self.layers[i-1].e_gen
                if i == len(self.layers) - 1:
                    if y_target is not None:
                        grad_W = torch.matmul(e_gen_lower.t(), y_target)
                        self.W[i-1] += alpha_u * grad_W
                else:
                    s_own = self.layers[i].s
                    grad_W = torch.matmul(e_gen_lower.t(), s_own)
                    self.W[i-1] += alpha_u * grad_W

# --- Optuna Objective Function ---
def objective(trial):
    # 1. ハイパーパラメータのサンプリング
    # alpha_gen / alpha_disc の比率を[0.01, 100]の一様分布で探索
    ratio = trial.suggest_float('ratio', 0.01, 100.0)
    
    # alpha_gen + alpha_disc = 1 となるように計算
    # alpha_gen = ratio * alpha_disc
    # ratio * alpha_disc + alpha_disc = 1 => alpha_disc = 1 / (1 + ratio)
    alpha_disc = 1.0 / (1.0 + ratio)
    alpha_gen = 1.0 - alpha_disc
    
    # alpha_u を [1e-5, 1e-2] で探索 (対数スケール推奨)
    alpha_u = trial.suggest_float('alpha_u', 1e-5, 1e-2, log=True)
    
    # 既存の設定をコピーして更新
    config = BASE_CONFIG.copy()
    config['alpha_gen'] = alpha_gen
    config['alpha_disc'] = alpha_disc
    config['alpha_u'] = alpha_u
    
    dataset_name = 'MNIST' # 固定あるいは引数化
    print(f"\nTrial {trial.number}: ratio={ratio:.4f} (gen={alpha_gen:.4f}, disc={alpha_disc:.4f}), alpha_u={alpha_u:.2e}")

    # 2. データセット準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    train_l = DataLoader(train_d, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    
    # 3. モデル構築
    layer_sizes = [784, 500, 500, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=config).to(config['device'])
    
    steps = int(config['T_st'] / config['dt'])
    
    best_acc = 0.0

    # 4. 学習ループ
    for epoch in range(config['epochs']):
        # --- Training ---
        model.train()
        for batch_idx, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
            
            targets = torch.zeros(imgs.size(0), 10).to(config['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = spikegen.rate(imgs_rate, steps)
            
            model.reset_state(imgs.size(0), config['device'])
            
            for t in range(steps):
                x_t = spike_in[t]
                model.forward_dynamics(x_data=x_t, y_target=targets)
                model.manual_weight_update(x_data=x_t, y_target=targets)
                model.clip_weights(20.0)
    
        # --- Testing ---
        model.layers[-1].switch2testing_mode()
        model.eval()
        test_correct = 0
        test_samples = 0
        
        with torch.no_grad():
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = spikegen.rate(imgs_rate, steps)
                
                model.reset_state(imgs.size(0), config['device'])
                sum_out_spikes = 0
                
                for t in range(steps):
                    x_t = spike_in[t]
                    model.forward_dynamics(x_data=x_t, y_target=None)
                    sum_out_spikes += model.layers[-1].s
                
                _, pred = torch.max(sum_out_spikes, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        
        model.layers[-1].switch2training_mode()
        
        # 途中経過報告 (Pruning用)
        trial.report(test_acc, epoch)
        
        # 枝刈り判定 (有望でない試行はここで打ち切り)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc

def run_optimization():
    # OptunaのStudy作成 (最大化問題)
    study = optuna.create_study(direction='maximize')
    
    # 最適化実行 (n_trialsで試行回数を指定)
    print("Starting optimization...")
    study.optimize(objective, n_trials=20) # 必要に応じて回数を調整してください

    print("\n=== Optimization Finished ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.2f}%")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # 最適解から実際のパラメータを復元して表示
    best_ratio = study.best_params['ratio']
    best_alpha_disc = 1.0 / (1.0 + best_ratio)
    best_alpha_gen = 1.0 - best_alpha_disc
    print(f"  -> alpha_gen: {best_alpha_gen:.5f}")
    print(f"  -> alpha_disc: {best_alpha_disc:.5f}")

    # 結果の保存
    df = study.trials_dataframe()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"{BASE_CONFIG['save_dir']}/optuna_results_{timestamp}.csv"
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    run_optimization()