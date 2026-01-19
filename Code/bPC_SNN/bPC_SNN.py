import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
from snntorch import spikegen
import pandas as pd
import os
import optuna

# --- 固定パラメータ設定 ---
FIXED_CONFIG = {
    'dt' : 0.25,          # 微小時間 (ms)
    'T_r' : 1.0,          # 絶対不応期 (ms)
    'T_st' : 200,          # 1データ提示時間 (ms)
    'max_freq' : 320.0,   # 最大周波数 (Hz)
    'R_m' : 1.0,          # 抵抗 (固定)
    'alpha_disc' : 1.0,   # 識別係数 (固定または探索対象に含めても良いが今回は固定)
    'batch_size': 64,
    'epochs_per_trial': 3, # 1トライアルあたりのエポック数 (高速化のため少なめに)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Optimization'
}

os.makedirs(FIXED_CONFIG['save_dir'], exist_ok=True)

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
    def __init__(self, idx, output_dim, config, data=False, label=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = data
        self.is_label_layer = label
        
        # 内部状態
        self.v = None
        self.s = None
        self.x = None # Trace
        self.j = None # Input Current State
        # self.ref_count = None # 不応期カウンタ
        
        self.e_gen = None
        self.e_disc = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)
        
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def switch_label_mode(self):
        self.is_label_layer = not self.is_label_layer

    def update_state(self, total_input_current):
        if self.is_data_layer or self.is_label_layer:
            # データ層・ラベル層は外部スパイクをそのまま使用
            self.s = total_input_current
        else:
            # 隠れ層: LIF Dynamics
            dt = self.cfg['dt']
            
            # is_refractory = (self.ref_count > 0)

            # 電流ダイナミクス
            d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
            self.j = self.j + (dt / self.cfg['tau_j']) * d_j
            
            # 膜電位ダイナミクス
            d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
            self.v = self.v + (dt / self.cfg['tau_m']) * d_v
            
            # 不応期リセット
            # self.v = torch.where(is_refractory, torch.zeros_like(self.v), self.v)
            
            # 発火判定
            spikes = (self.v > self.cfg['thresh']).float()
            # spikes = torch.where(is_refractory, torch.zeros_like(spikes), spikes)
            self.s = spikes
            
            # 発火後リセット & 不応期設定
            self.v = self.v * (1 - spikes)
            # ref_steps = int(self.cfg['T_r'] / dt)
            # self.ref_count = torch.where(spikes > 0, torch.tensor(float(ref_steps), device=self.ref_count.device), self.ref_count)
            # self.ref_count = torch.clamp(self.ref_count - 1, min=0)

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
            is_label = (i == len(layer_sizes) - 1)
            self.layers.append(bPC_SNNLayer(idx=i, output_dim=size, config=config, data=is_data, label=is_label))
            
        # --- 重み定義 ---
        self.W = nn.ParameterList() # Top-down
        self.V = nn.ParameterList() # Bottom-up
        
        # 重み初期値は小さめに設定 (爆発防止)
        init_scale = 0.05
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * init_scale))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * init_scale))

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=1.0):
        with torch.no_grad():
            for w_list in [self.W, self.V]:
                for w in w_list:
                    w.data.clamp_(-max_norm, max_norm)

    def forward_dynamics(self, x_data_t, y_target_t=None, training_mode=True):
        # === 1. Update Phase ===
        for i, layer in enumerate(self.layers):
            total_input = 0
            
            if i == 0: # Data
                total_input = x_data_t
            elif i == len(self.layers) - 1 and layer.is_label_layer and training_mode: # Label (Train)
                total_input = y_target_t
            else: # Hidden or Label (Test)
                total_input += (- layer.e_gen)
                total_input += (- layer.e_disc)

                if i < len(self.layers) - 1:
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
            
            if i > 0:
                s_lower = self.layers[i-1].s
                z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

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

# --- データセット準備 (一度だけ読み込み) ---
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 探索高速化のためにデータ数を減らしても良いが、ここでは全データを使用
    # 必要なら Subset を使用してください
    # train_d = Subset(train_d, range(5000)) 
    # test_d = Subset(test_d, range(1000))

    train_l = DataLoader(train_d, batch_size=FIXED_CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=FIXED_CONFIG['batch_size'], shuffle=False, drop_last=True)
    return train_l, test_l

TRAIN_LOADER, TEST_LOADER = get_dataloaders()

# --- Optuna Objective Function ---
def objective(trial):
    # 1. ハイパーパラメータのサンプリング
    # T_st=25msと短いため、時定数や閾値は非常に敏感です
    
    tau_m = trial.suggest_float('tau_m', 2.0, 50.0)      # 膜時定数
    tau_j = trial.suggest_float('tau_j', 1.0, 30.0)      # 電流時定数
    tau_tr = trial.suggest_float('tau_tr', 5.0, 100.0)   # トレース時定数 (tau_mより長めが良い)
    
    thresh = trial.suggest_float('thresh', 0.5, 5.0)     # 閾値 (発火率制御の要)
    
    kappa_j = trial.suggest_float('kappa_j', 0.1, 2.0)   # 漏れ係数 (電流)
    gamma_m = trial.suggest_float('gamma_m', 0.5, 2.0)   # 漏れ係数 (電圧)
    
    alpha_u = trial.suggest_float('alpha_u', 1e-4, 0.05, log=True)   # 学習率
    alpha_gen = trial.suggest_float('alpha_gen', 1e-4, 1.0, log=True) # 生成誤差係数
    
    # 設定の統合
    config = FIXED_CONFIG.copy()
    config.update({
        'tau_m': tau_m,
        'tau_j': tau_j,
        'tau_tr': tau_tr,
        'thresh': thresh,
        'kappa_j': kappa_j,
        'gamma_m': gamma_m,
        'alpha_u': alpha_u,
        'alpha_gen': alpha_gen
    })
    
    # 2. モデルの構築
    layer_sizes = [784, 500, 500, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=config).to(config['device'])
    
    num_steps = int(config['T_st'] / config['dt'])
    
    # 3. 学習ループ (トライアル用)
    for epoch in range(config['epochs_per_trial']):
        # --- Training ---
        model.train()
        for batch_idx, (imgs, lbls) in enumerate(TRAIN_LOADER):
            imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
            
            targets_one_hot = torch.zeros(imgs.size(0), 10).to(config['device'])
            targets_one_hot.scatter_(1, lbls.view(-1, 1), 1.0)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, num_steps, config)
            target_spikes = generate_poisson_spikes(targets_one_hot, num_steps, config)
            
            model.reset_state(imgs.size(0), config['device'])
            
            for t in range(num_steps):
                x_t = spike_in[t]
                y_t = target_spikes[t]
                model.forward_dynamics(x_data_t=x_t, y_target_t=y_t, training_mode=True)
                model.manual_weight_update()
                model.clip_weights(1.0)
        
        # --- Testing & Pruning ---
        # Label層を推論モードへ
        model.layers[-1].switch_label_mode()
        model.eval()
        
        test_correct = 0
        test_samples = 0
        
        with torch.no_grad():
            for imgs, lbls in TEST_LOADER:
                imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, num_steps, config)
                
                model.reset_state(imgs.size(0), config['device'])
                sum_out_spikes = 0
                
                for t in range(num_steps):
                    x_t = spike_in[t]
                    model.forward_dynamics(x_data_t=x_t, y_target_t=None, training_mode=False)
                    sum_out_spikes += model.layers[-1].s
                
                _, pred = torch.max(sum_out_spikes, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
        
        test_acc = 100 * test_correct / test_samples
        
        # Label層を戻す
        model.layers[-1].switch_label_mode()
        
        # Optunaへの報告と枝刈り判定
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return test_acc

if __name__ == "__main__":
    print("=== Starting Bayesian Optimization with Optuna ===")
    print(f"Fixed Config: T_st={FIXED_CONFIG['T_st']}ms, Max Freq={FIXED_CONFIG['max_freq']}Hz")
    
    # 最適化の実行
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # トライアル数は時間に応じて調整してください

    print("\n=== Optimization Finished ===")
    print("Best params:", study.best_params)
    print("Best test accuracy:", study.best_value)

    # 結果の保存
    df = study.trials_dataframe()
    df.to_csv(f"{FIXED_CONFIG['save_dir']}/optuna_results.csv", index=False)
    
    # 最良のパラメータで再学習を行う場合は、ここから実装を追加