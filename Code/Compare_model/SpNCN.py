import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os
import optuna

# --- 基本設定 ---
BASE_CONFIG = {
    # --- 固定パラメータ ---
    'dt': 0.25,          
    'T_r' : 1.0,         
    'T_st' : 25,        
    'R_m': 1.0,          
    'beta': 1.0,        
    'max_freq' : 320.0, 
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Optuna',
    
    # --- モード別設定 ---
    # 探索(Search)用
    'search_epochs': 3,
    'train_subset': 10000,  # 探索時はデータを間引く
    
    # 本番(Full)用
    'full_epochs': 10,      # 本番はしっかり回す
}

os.makedirs(BASE_CONFIG['save_dir'], exist_ok=True)

# --- 共通関数 (変更なし) ---
def generate_poisson_spikes(data, num_steps, config):
    device = data.device
    batch_size, dim = data.shape
    dt = config['dt']
    max_freq = config['max_freq']
    firing_probs = data * max_freq * (dt / 1000.0)
    firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
    firing_probs_expanded = firing_probs.unsqueeze(0).expand(num_steps, batch_size, dim)
    spikes = torch.bernoulli(firing_probs_expanded)
    return spikes

class SpNCNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, is_data_layer=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = is_data_layer
        
    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.e = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        dt = self.cfg['dt']
        T_r = self.cfg['T_r']
        
        self.ref_count = (self.ref_count - dt).clamp(min=0)
        is_active = (self.ref_count <= 0).float()

        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        v_next = self.v + (dt / self.cfg['tau_m']) * d_v
        
        self.v = is_active * v_next 
        
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        
        self.v = self.v * (1 - spikes)
        self.ref_count = torch.where(spikes.bool(), torch.tensor(T_r, device=self.v.device), self.ref_count)
        
        if self.is_data_layer:
            self.x = spikes
        else:
            self.x = self.x - self.x / self.cfg['tau_tr'] + spikes

class SpNCN(nn.Module):
    def __init__(self, hidden_sizes, input_size, label_size, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        self.input_layer = SpNCNLayer('input', input_size, config, is_data_layer=True)
        self.label_layer = SpNCNLayer('label', label_size, config, is_data_layer=True)
        self.hidden_layers = nn.ModuleList()
        for i, size in enumerate(hidden_sizes):
            self.hidden_layers.append(SpNCNLayer(i, size, config, is_data_layer=False))
            
        self.W = nn.ParameterList() 
        self.E = nn.ParameterList() 
        
        h0_dim = hidden_sizes[0]
        self.W_x = nn.Parameter(torch.randn(input_size, h0_dim) * 0.05) 
        self.E_x = nn.Parameter(torch.randn(h0_dim, input_size) * 0.05) 
        self.W_y = nn.Parameter(torch.randn(label_size, h0_dim) * 0.05) 
            
        for i in range(len(hidden_sizes) - 1):
            dim_lower = hidden_sizes[i]
            dim_upper = hidden_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            self.E.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))

    def reset_state(self, batch_size, device):
        self.input_layer.init_state(batch_size, device)
        self.label_layer.init_state(batch_size, device)
        for layer in self.hidden_layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=20.0):
        with torch.no_grad():
            for w_list in [self.W, self.E]:
                for w in w_list:
                    self._clip(w, max_norm)
            self._clip(self.W_x, max_norm)
            self._clip(self.E_x, max_norm)
            self._clip(self.W_y, max_norm)

    def _clip(self, w, max_norm):
        norms = w.norm(p=2, dim=1, keepdim=True)
        mask = norms > max_norm
        w.data = torch.where(mask, w * (max_norm / (norms + 1e-8)), w)

    def forward_dynamics(self, x_data, y_target=None):
        s_h0 = self.hidden_layers[0].s
        z_pred_x = s_h0 @ self.W_x.t()
        self.input_layer.update_state(z_pred_x)
        z_pred_y = s_h0 @ self.W_y.t()
        self.label_layer.update_state(z_pred_y)

        for i, layer in enumerate(self.hidden_layers):
            total_input = (-layer.e) 
            if i == 0:
                feedback = self.input_layer.e @ self.E_x.t()
                total_input += feedback
            else:
                e_lower = self.hidden_layers[i-1].e
                feedback = e_lower @ self.E[i-1].t()
                total_input += feedback
            layer.update_state(total_input)

        self.input_layer.e = x_data - self.input_layer.s
        if y_target is not None:
            self.label_layer.e = y_target - self.label_layer.s
        else:
            self.label_layer.e = torch.zeros_like(self.label_layer.s)

        for i, layer in enumerate(self.hidden_layers):
            if i < len(self.hidden_layers) - 1:
                s_upper = self.hidden_layers[i+1].s
                z_pred = s_upper @ self.W[i].t()
                layer.e = layer.x - z_pred
            else:
                layer.e = torch.zeros_like(layer.x)

    def manual_weight_update(self):
        lr = self.config['alpha_u']
        beta = self.config['beta']
        with torch.no_grad():
            s_h0 = self.hidden_layers[0].s
            e_x = self.input_layer.e
            self.W_x += lr * (e_x.t() @ s_h0)
            self.E_x += lr * beta * (s_h0.t() @ e_x)
            e_y = self.label_layer.e
            self.W_y += lr * (e_y.t() @ s_h0)
            for i in range(len(self.hidden_layers) - 1):
                s_upper = self.hidden_layers[i+1].s
                e_lower = self.hidden_layers[i].e
                self.W[i] += lr * (e_lower.t() @ s_upper)
                self.E[i] += lr * beta * (s_upper.t() @ e_lower)

# --- 汎用学習実行関数 ---
def run_training(config, trial=None):
    """
    trial=None の場合は「本番学習モード」として動作
    trial!=None の場合は「探索モード」として動作（枝刈り・データ間引きあり）
    """
    is_search_mode = (trial is not None)
    mode_name = "SEARCH" if is_search_mode else "FINAL RUN"
    print(f"\n=== Starting {mode_name} Mode ===")
    
    # 1. データセット設定
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    # ※ 実装簡略化のため毎回ロードしていますが、大規模データなら外でロードすべきです
    train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 探索モードかつ train_subset が指定されていればデータを間引く
    if is_search_mode and config['train_subset'] is not None:
        indices = torch.randperm(len(train_d))[:config['train_subset']]
        train_d = Subset(train_d, indices)
    else:
        # 本番モードなら全データ使用
        pass

    train_l = DataLoader(train_d, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    # 2. モデル構築
    model = SpNCN(
        hidden_sizes=[500, 500], 
        input_size=784, 
        label_size=10, 
        config=config
    ).to(config['device'])
    
    steps = int(config['T_st'] / config['dt'])
    
    # エポック数の決定
    epochs = config['search_epochs'] if is_search_mode else config['full_epochs']
    best_acc = 0.0

    # 3. 学習ループ
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        # tqdmなどは使わずシンプルに表示
        for i, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
            targets = torch.zeros(imgs.size(0), 10).to(config['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, config)
            
            model.reset_state(imgs.size(0), config['device'])
            
            # 訓練中の予測精度も参考用にとる
            out_spikes_train = 0
            for t in range(steps):
                model.forward_dynamics(x_data=spike_in[t], y_target=targets)
                model.manual_weight_update()
                model.clip_weights(20.0)
                out_spikes_train += model.label_layer.s
            
            _, pred = torch.max(out_spikes_train, 1)
            train_correct += (pred == lbls).sum().item()
            train_total += lbls.size(0)

        train_acc = 100 * train_correct / train_total

        # --- Test ---
        model.eval()
        test_correct = 0
        test_samples = 0
        
        for imgs, lbls in test_l:
            imgs, lbls = imgs.to(config['device']), lbls.to(config['device'])
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, config)
            
            model.reset_state(imgs.size(0), config['device'])
            out_spikes = 0
            
            for t in range(steps):
                with torch.no_grad():
                    model.forward_dynamics(x_data=spike_in[t], y_target=None)
                out_spikes += model.label_layer.s
            
            _, pred = torch.max(out_spikes, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        best_acc = max(best_acc, test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # 探索モード時の枝刈り
        if is_search_mode:
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_acc

# --- Optunaの目的関数 ---
def objective(trial):
    # ベースの設定をコピー
    cfg = BASE_CONFIG.copy()
    
    # 探索パラメータ
    cfg['tau_m'] = trial.suggest_float('tau_m', 1.0, 10.0)
    cfg['tau_j'] = trial.suggest_float('tau_j', 1.0, 10.0)
    cfg['tau_tr'] = trial.suggest_float('tau_tr', 2.0, 15.0)
    cfg['kappa_j'] = trial.suggest_float('kappa_j', 0.1, 2.0)
    cfg['gamma_m'] = trial.suggest_float('gamma_m', 0.1, 2.0)
    cfg['alpha_u'] = trial.suggest_float('alpha_u', 1e-4, 5e-2, log=True)
    cfg['thresh'] = trial.suggest_float('thresh', 0.2, 1.2)
    
    # 学習実行（探索モード）
    accuracy = run_training(cfg, trial=trial)
    
    return accuracy

if __name__ == "__main__":
    print("=== Step 1: Starting Bayesian Optimization ===")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 探索回数は時間に応じて調整してください (例: 20回)
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*30)
    print("Optimization Finished.")
    print(f"Best Accuracy in search: {study.best_value:.2f}%")
    print("Best Params:", study.best_params)
    print("="*30 + "\n")
    
    # --- 自動で本番学習へ移行 ---
    print("=== Step 2: Starting Final Training with Best Params ===")
    
    # 1. ベース設定に、見つかったベストパラメータを上書き統合
    final_config = BASE_CONFIG.copy()
    final_config.update(study.best_params)
    
    # 2. 本番学習の実行 (trial=Noneを渡すことでFull Modeになる)
    final_acc = run_training(final_config, trial=None)
    
    print(f"\nFinal Training Finished. Final Test Accuracy: {final_acc:.2f}%")
    
    # 結果保存
    df = pd.DataFrame([final_config])
    df['final_test_acc'] = final_acc
    df.to_csv(f"{BASE_CONFIG['save_dir']}/final_result.csv", index=False)
