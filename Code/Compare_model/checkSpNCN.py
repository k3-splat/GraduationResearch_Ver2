import torch
import torch.nn as nn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset # Subsetを追加
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 ---
# Fashion MNISTで安定した設定を採用
CONFIG = {
    'dt': 0.25,
    'tau_m': 20.0,
    'tau_j': 10.0,
    'tau_tr': 30.0,
    'kappa_j': 0.25,
    'gamma_m': 1.0,
    'R_m': 1.0,
    'alpha_gen': 1.0,    # 生成（画像再構成）の誤差係数
    'alpha_u': 0.055,   # 重み学習率 (安定化のため低減済み)
    'beta': 1.0,        # 誤差重み学習率 (安定化のため低減済み)
    'thresh': 0.4,       # 発火閾値
    'num_steps': 800,    # 推論ステップ数
    'batch_size': 1,
    'epochs': 10,        # 学習エポック数
    'max_freq' : 63.75,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Comparison' # 結果保存先
}

# 結果保存用ディレクトリ作成
os.makedirs(CONFIG['save_dir'], exist_ok=True)

def generate_poisson_spikes(data, num_steps, config):
    """
    入力データ(0~1)を受け取り、ポアソンスパイク列(Time, Batch, Dim)を生成する
    """
    batch_size, dim = data.shape
    
    dt = config['dt']
    max_freq = config['max_freq']
    
    # 確率 p = freq(Hz) * dt(ms) / 1000
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

    def update_state(self, total_input_current):
        dt = self.cfg['dt']
        
        # LIF Dynamics
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        self.v = self.v + (dt / self.cfg['tau_m']) * d_v
        
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        self.v = self.v * (1 - spikes) 
        
        if self.is_data_layer:
            self.x = spikes
        else:
            self.x = self.x + (dt / self.cfg['tau_tr']) * (-self.x) + spikes

class SpNCN(nn.Module):
    def __init__(self, hidden_sizes, input_size, label_size=None, config=None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # --- 1. 入力層 (Layer 0) ---
        self.input_layer = SpNCNLayer(idx='input', output_dim=input_size, config=config, is_data_layer=True)
        
        # --- 2. ラベル層 (Layer Label) ---
        self.label_size = label_size
        if label_size is not None:
            self.label_layer = SpNCNLayer(idx='label', output_dim=label_size, config=config, is_data_layer=True)
        else:
            self.label_layer = None
            
        # --- 3. 隠れ層 (Hidden Layers) ---
        self.hidden_layers = nn.ModuleList()
        for i, size in enumerate(hidden_sizes):
            self.hidden_layers.append(SpNCNLayer(idx=i, output_dim=size, config=config, is_data_layer=False))
            
        # --- 重み定義 ---
        self.W = nn.ParameterList() # Top-down (Upper -> Lower)
        self.E = nn.ParameterList() # Feedback (Lower_error -> Upper)
        
        # (A) Input <-> Hidden[0] (Bidirectional)
        h0_dim = hidden_sizes[0]
        self.W_x = nn.Parameter(torch.randn(input_size, h0_dim) * 0.05) # H0 -> Input
        self.E_x = nn.Parameter(torch.randn(h0_dim, input_size) * 0.05) # Input_err -> H0
        
        # (B) Label <- Hidden[0] (Unidirectional / No Feedback)
        if self.label_layer is not None:
            self.W_y = nn.Parameter(torch.randn(label_size, h0_dim) * 0.05) # H0 -> Label
            
        # (C) Hidden[i] <-> Hidden[i+1]
        for i in range(len(hidden_sizes) - 1):
            dim_lower = hidden_sizes[i]
            dim_upper = hidden_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            self.E.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))

        self.total_synops = 0.0

    def reset_state(self, batch_size, device):
        self.input_layer.init_state(batch_size, device)
        if self.label_layer is not None:
            self.label_layer.init_state(batch_size, device)
        for layer in self.hidden_layers:
            layer.init_state(batch_size, device)
        # total_synopsは累積させるためリセットしない（必要なら外部で管理）

    def clip_weights(self, max_norm=20.0):
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

    def forward_dynamics(self, x_data, y_target=None, training_mode=True):
        """
        SpNCN Algorithm 1
        """
        alpha = self.config['alpha_gen']

        # === 1. Update Phase ===
        # (A) Input & Label Layers
        s_h0 = self.hidden_layers[0].s
        
        z_pred_x = s_h0 @ self.W_x.t()
        self.input_layer.update_state(z_pred_x)
        if training_mode: self.total_synops += s_h0.sum().item() * self.input_layer.dim
        
        if self.label_layer is not None:
            z_pred_y = s_h0 @ self.W_y.t()
            self.label_layer.update_state(z_pred_y)
            if training_mode: self.total_synops += s_h0.sum().item() * self.label_layer.dim

        # (B) Hidden Layers
        for i, layer in enumerate(self.hidden_layers):
            total_input = 0
            total_input += (-layer.e) # Local Error
            
            if i == 0:
                total_input += self.input_layer.e @ self.E_x.t()
            else:
                e_lower = self.hidden_layers[i-1].e
                total_input += e_lower @ self.E[i-1].t()
                
            layer.update_state(total_input)

        # === 2. Prediction & Error Phase ===
        self.input_layer.e = alpha * (x_data - self.input_layer.s)
        
        if self.label_layer is not None:
            if y_target is not None:
                # 学習時: 教師信号あり -> 誤差を計算
                self.label_layer.e = alpha * (y_target - self.label_layer.s)
            else:
                # テスト時: 教師信号なし -> 誤差ゼロ (またはTop-down予測のみで動作)
                self.label_layer.e = torch.zeros_like(self.label_layer.s)

        for i, layer in enumerate(self.hidden_layers):
            if i < len(self.hidden_layers) - 1:
                s_upper = self.hidden_layers[i+1].s
                z_pred = s_upper @ self.W[i].t()
                layer.e = alpha * (layer.x - z_pred)
                if training_mode: self.total_synops += s_upper.sum().item() * layer.dim
            else:
                layer.e = torch.zeros_like(layer.x)

    def manual_weight_update(self):
        lr = self.config['alpha_u']
        beta = self.config['beta']
        
        s_h0 = self.hidden_layers[0].s
        e_x = self.input_layer.e
        
        self.W_x += lr * (e_x.t() @ s_h0)
        self.E_x += lr * beta * (s_h0.t() @ e_x)
        
        if self.label_layer is not None:
            e_y = self.label_layer.e
            self.W_y += lr * (e_y.t() @ s_h0)
            
        for i in range(len(self.hidden_layers) - 1):
            s_upper = self.hidden_layers[i+1].s
            e_lower = self.hidden_layers[i].e
            
            self.W[i] += lr * (e_lower.t() @ s_upper)
            self.E[i] += lr * beta * (s_upper.t() @ e_lower)

def run_experiment(dataset_name='MNIST'):
    """
    共通の実験実行関数
    """
    print(f"\n=== Running SpNCN on {dataset_name} (Supervised Classification) ===")
    
    # データの準備 (Normalizeなし、[0,1]範囲)
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

    # ========================================================
    # 修正箇所: データを10個だけに制限する処理
    # ========================================================
    print(f"!!! {dataset_name}: 使用データを先頭10個に制限します !!!")
    train_d = Subset(train_d, range(10))
    test_d = Subset(test_d, range(10))
    # ========================================================

    # DataLoader
    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    # モデル構築
    model = SpNCN(
        hidden_sizes=[3000, 3000, 3000], 
        input_size=784, 
        label_size=10, 
        config=CONFIG
    ).to(CONFIG['device'])
    
    logs = []

    with torch.no_grad():
        for epoch in range(CONFIG['epochs']):
            # --- Training ---
            model.train()
            train_correct = 0
            train_samples = 0
            epoch_start = time.time()
            start_synops = model.total_synops
            
            for batch_idx, (imgs, lbls) in enumerate(train_l):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                # ターゲット作成 (One-hot)
                targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
                targets.scatter_(1, lbls.view(-1, 1), 1)
                
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, num_steps=CONFIG['num_steps'], config=CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                out_spikes = 0
                
                for t in range(CONFIG['num_steps']):
                    x_t = spike_in[t]
                    # 学習時: y_targetあり
                    model.forward_dynamics(x_data=x_t, y_target=targets, training_mode=True)
                    model.manual_weight_update()
                    model.clip_weights(20.0)
                    out_spikes += model.label_layer.s
                    
                _, pred = torch.max(out_spikes, 1)
                train_correct += (pred == lbls).sum().item()
                train_samples += lbls.size(0)
                
                # ログ出力頻度調整（データが少ないので毎回出ても良いが、念のため残す）
                if batch_idx % 1 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Train Acc: {100*train_correct/train_samples:.2f}%")
                    print(out_spikes) # 冗長な出力は抑制してもよい

            train_acc = 100 * train_correct / train_samples
            epoch_synops = model.total_synops - start_synops
            
            # --- Testing ---
            model.eval() # 評価モード (本実装では挙動変わらずだが明示)
            test_correct = 0
            test_samples = 0
            
            # テスト時は勾配計算不要だが、状態更新は必要なので with torch.no_grad() は manual_weight_update 内のみ
            # ただし全体を no_grad にしても forward_dynamics の計算（in-place更新）は可能
            # ここでは重み更新がないことを保証するため手動更新を呼ばないことで対応
            
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, num_steps=CONFIG['num_steps'], config=CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                out_spikes = 0
                
                for t in range(CONFIG['num_steps']):
                    x_t = spike_in[t]
                    # テスト時: y_target=None (ターゲットなし、予測のみ)
                    # training_mode=False (SynOpsカウントしない、学習もしない)
                    model.forward_dynamics(x_data=x_t, y_target=None, training_mode=False)
                    out_spikes += model.label_layer.s
                
                _, pred = torch.max(out_spikes, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
                
            test_acc = 100 * test_correct / test_samples
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch} Finished | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s | SynOps: {epoch_synops:.2e}")
            
            logs.append({
                'dataset': dataset_name,
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'time': epoch_time,
                'synops': epoch_synops,
                'total_synops': model.total_synops
            })
            
            # CSV保存
            df = pd.DataFrame(logs)
            df.to_csv(f"{CONFIG['save_dir']}/log_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    # MNISTの実験を実行
    run_experiment('MNIST')
    
    # 必要であれば以下をコメントアウト解除して実行
    # run_experiment('FashionMNIST')