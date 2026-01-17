import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os
import optuna

# --- ハイパーパラメータ設定 (最適化対象のデフォルト値) ---
CONFIG = {
    'dt': 1.0,          # 微小時間
    'T_r' : 1.0,        # 絶対不応期 (time units)
    'T_st' : 200,       # データ提示時間
    'tau_m': 20.0,      # 時定数（膜電位）
    'tau_j': 10.0,      # 時定数（入力電流）
    'tau_tr': 30.0,     # 時定数（ローパスフィルタ）
    'kappa_j': 0.25,    # 漏れ係数（入力電流）
    'gamma_m': 1.0,     # 漏れ係数（膜電位）
    'R_m': 1.0,         # 抵抗
    'thresh': 0.4,      # 発火閾値
    'max_freq' : 63.75, # 最大周波数
    'batch_size': 64,
    'max_epochs': 5,    # ベイズ最適化用にエポック数を短縮 (本番は10~20推奨)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Optimization'
}

# --- ポアソンエンコーディング関数 ---
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

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# --- コスト計測クラス (今回は最適化が主目的なので簡略化して利用) ---
class CostMonitor:
    def __init__(self):
        self.reset()
    def reset(self):
        pass # 最適化中は詳細なログは取らないがクラス構造は維持
    def add_synops(self, count):
        pass
    def add_dense_ops(self, muls, adds):
        pass

# --- モデル定義 ---
class SpNCNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, monitor, is_data_layer=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.monitor = monitor
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
    def __init__(self, hidden_sizes, input_size, label_size=None, config=None, monitor=None):
        super().__init__()
        self.config = config
        self.monitor = monitor
        self.layers = nn.ModuleList()
        
        self.input_layer = SpNCNLayer(idx='input', output_dim=input_size, config=config, monitor=monitor, is_data_layer=True)
        
        self.label_size = label_size
        if label_size is not None:
            self.label_layer = SpNCNLayer(idx='label', output_dim=label_size, config=config, monitor=monitor, is_data_layer=True)
        else:
            self.label_layer = None
            
        self.hidden_layers = nn.ModuleList()
        for i, size in enumerate(hidden_sizes):
            self.hidden_layers.append(SpNCNLayer(idx=i, output_dim=size, config=config, monitor=monitor, is_data_layer=False))
            
        self.W = nn.ParameterList() 
        self.E = nn.ParameterList() 
        
        h0_dim = hidden_sizes[0]
        self.W_x = nn.Parameter(torch.randn(input_size, h0_dim) * 0.05) 
        self.E_x = nn.Parameter(torch.randn(h0_dim, input_size) * 0.05) 
        
        if self.label_layer is not None:
            self.W_y = nn.Parameter(torch.randn(label_size, h0_dim) * 0.05) 
            
        for i in range(len(hidden_sizes) - 1):
            dim_lower = hidden_sizes[i]
            dim_upper = hidden_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            self.E.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))

    def reset_state(self, batch_size, device):
        self.input_layer.init_state(batch_size, device)
        if self.label_layer is not None:
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
            if self.label_layer is not None:
                self._clip(self.W_y, max_norm)

    def _clip(self, w, max_norm):
        norms = w.norm(p=2, dim=1, keepdim=True)
        mask = norms > max_norm
        w.data = torch.where(mask, w * (max_norm / (norms + 1e-8)), w)

    def forward_dynamics(self, x_data, y_target=None):
        # 1. Update Phase
        s_h0 = self.hidden_layers[0].s
        z_pred_x = s_h0 @ self.W_x.t()
        self.input_layer.update_state(z_pred_x)
        
        if self.label_layer is not None:
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

        # 2. Prediction & Error Phase
        self.input_layer.e = x_data - self.input_layer.s
        if self.label_layer is not None:
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

    def set_gradients_for_optimizer(self):
        """
        Optunaで探索する 'beta' を使用して勾配をセットする
        """
        beta = self.config['beta'] # 最適化されたbeta値を使用
        
        with torch.no_grad():
            s_h0 = self.hidden_layers[0].s
            e_x = self.input_layer.e
            
            # W_x
            grad_Wx = -(e_x.t() @ s_h0)
            if self.W_x.grad is None: self.W_x.grad = grad_Wx
            else: self.W_x.grad += grad_Wx
            
            # E_x (beta倍)
            grad_Ex = -beta * (s_h0.t() @ e_x)
            if self.E_x.grad is None: self.E_x.grad = grad_Ex
            else: self.E_x.grad += grad_Ex
            
            # W_y
            if self.label_layer is not None:
                e_y = self.label_layer.e
                grad_Wy = -(e_y.t() @ s_h0)
                if self.W_y.grad is None: self.W_y.grad = grad_Wy
                else: self.W_y.grad += grad_Wy

            # Hidden layers
            for i in range(len(self.hidden_layers) - 1):
                s_upper = self.hidden_layers[i+1].s
                e_lower = self.hidden_layers[i].e
                
                grad_W = -(e_lower.t() @ s_upper)
                if self.W[i].grad is None: self.W[i].grad = grad_W
                else: self.W[i].grad += grad_W
                
                grad_E = -beta * (s_upper.t() @ e_lower)
                if self.E[i].grad is None: self.E[i].grad = grad_E
                else: self.E[i].grad += grad_E


# --- データローダー準備（共通化） ---
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 高速化のため、ベイズ最適化中はDropLast=Trueにしてサイズを固定
    train_l = DataLoader(train_d, batch_size=batch_size, shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_l, test_l

# --- Optuna Objective Function ---
def objective(trial, train_loader, test_loader):
    # 1. 探索空間の定義
    alpha_u = trial.suggest_float('alpha_u', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    beta = trial.suggest_float('beta', 0.1, 3.0) # 0.1~3.0程度を探索
    
    # Configの更新
    current_config = CONFIG.copy()
    current_config.update({
        'alpha_u': alpha_u,
        'weight_decay': weight_decay,
        'beta': beta
    })
    
    # 2. モデル構築
    monitor = CostMonitor()
    model = SpNCN(
        hidden_sizes=[500, 500], 
        input_size=784, 
        label_size=10, 
        config=current_config,
        monitor=monitor
    ).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=alpha_u, weight_decay=weight_decay)
    
    steps = int(current_config['T_st'] / current_config['dt'])
    
    # 3. 学習ループ
    for epoch in range(current_config['max_epochs']):
        model.train()
        
        # 学習 (全バッチは時間がかかる場合、Optuna用に一部だけ使う手もあるが、ここでは全データ使用)
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, current_config)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            
            for t in range(steps):
                optimizer.zero_grad()
                x_t = spike_in[t]
                model.forward_dynamics(x_data=x_t, y_target=targets)
                model.set_gradients_for_optimizer() # ここでbetaが使われる
                optimizer.step()
                model.clip_weights(20.0)

        # 検証 (Pruning用)
        model.eval()
        test_correct = 0
        test_samples = 0
        
        # テストデータの20%程度だけ使って高速に評価するのも定石だが、今回は全件やる
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, current_config)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            out_spikes = 0
            
            for t in range(steps):
                x_t = spike_in[t]
                with torch.no_grad():
                    model.forward_dynamics(x_data=x_t, y_target=None)
                out_spikes += model.label_layer.s
            
            _, pred = torch.max(out_spikes, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        
        # Optunaに報告
        trial.report(test_acc, epoch)
        
        # Pruning (見込みがないなら打ち切り)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return test_acc

# --- メイン実行部 ---
if __name__ == "__main__":
    print("=== SpNCN Hyperparameter Optimization with Optuna (AdamW) ===")
    
    # データを事前にロード
    train_loader, test_loader = get_dataloaders(CONFIG['batch_size'])
    
    # Studyの作成 (最大化問題)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # 最適化実行 (n_trialsで試行回数を指定)
    print("Starting optimization...")
    study.optimize(lambda trial: objective(trial, train_loader, test_loader), n_trials=20)
    
    print("\n" + "="*50)
    print("Optimization Finished.")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Acc): {trial.value:.2f}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
    
    # 結果の保存
    df = study.trials_dataframe()
    df.to_csv(f"{CONFIG['save_dir']}/optuna_results.csv", index=False)
    print(f"Results saved to {CONFIG['save_dir']}/optuna_results.csv")
    
    # (Optional) 最適パラメータで再度学習を行うコードをここに続けても良い