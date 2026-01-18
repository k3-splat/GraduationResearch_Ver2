import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt': 0.25,          # 微小時間
    'T_r' : 1.0,         # 絶対不応期 (time units)
    'T_st' : 200,        # データ提示時間
    'tau_m': 20.0,       # 時定数（膜電位）
    'tau_j': 10.0,       # 時定数（入力電流）
    'tau_tr': 30.0,      # 時定数（ローパスフィルタ）
    'kappa_j': 0.25,     # 漏れ係数（入力電流）
    'gamma_m': 1.0,      # 漏れ係数（膜電位）
    'R_m': 1.0,          # 抵抗
    'alpha_u': 0.055,   # 重み学習率
    'beta': 1.0,        # 誤差重み学習率
    'thresh': 0.4,       # 発火閾値
    'max_freq' : 63.75, # 最大周波数
    'batch_size': 64,
    'target_acc': 95.0,  # 目標テスト精度 (%) (target_accモードで使用)
    'max_epochs': 10,   # 最大エポック数 (fixed_epochsモードではこの回数実行)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_CostMeasured'
}

# --- ポアソンエンコーディング関数 ---
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

# 結果保存用ディレクトリ作成
os.makedirs(CONFIG['save_dir'], exist_ok=True)

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
        # 不応期カウンタの初期化 (0なら活動可能)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        """
        ニューロン内部状態の更新 (Dense Operations)
        絶対不応期を実装
        """
        dt = self.cfg['dt']
        T_r = self.cfg['T_r']

        # --- 1. 不応期タイマーの更新 ---
        # カウンタを時間経過分減らす (最小値は0)
        self.ref_count = (self.ref_count - dt).clamp(min=0)
        
        # 活動可能なニューロンのマスク (ref_count == 0 なら 1.0, それ以外は 0.0)
        is_active = (self.ref_count <= 0).float()

        # --- 2. LIF Dynamics Calculation Cost ---
        # 入力電流 j の更新 (不応期に関わらず積分を行うのが一般的)
        # d_j = (-kappa_j * j + total_input_current)
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        # 膜電位 v の更新
        # d_v = (-gamma_m * v + R_m * j)
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        v_next = self.v + (dt / self.cfg['tau_m']) * d_v
        
        # 不応期中のニューロンは膜電位をリセット値(0)に固定、それ以外は更新
        self.v = is_active * v_next # + (1 - is_active) * 0.0

        # --- 3. Spike Generation ---
        # 閾値を超え、かつ不応期でない(activeな)ニューロンのみ発火
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        
        # Reset: 発火したニューロンの膜電位をリセット (v = 0)
        self.v = self.v * (1 - spikes)
        
        # 不応期タイマーの設定: 発火したニューロンの ref_count を T_r に設定
        # spikeが発生した箇所のref_countをT_rで上書き
        self.ref_count = torch.where(spikes.bool(), torch.tensor(T_r, device=self.v.device), self.ref_count)
        
        # --- 4. Trace Update ---
        if self.is_data_layer:
            self.x = spikes
        else:
            # x = x - x/tau + spikes
            self.x = self.x - self.x / self.cfg['tau_tr'] + spikes
        

class SpNCN(nn.Module):
    def __init__(self, hidden_sizes, input_size, label_size=None, config=None, monitor=None):
        super().__init__()
        self.config = config
        self.monitor = monitor
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
        self.W = nn.ParameterList() 
        self.E = nn.ParameterList() 
        
        # (A) Input <-> Hidden[0]
        h0_dim = hidden_sizes[0]
        self.W_x = nn.Parameter(torch.randn(input_size, h0_dim) * 0.05) 
        self.E_x = nn.Parameter(torch.randn(h0_dim, input_size) * 0.05) 
        
        # (B) Label <- Hidden[0]
        if self.label_layer is not None:
            self.W_y = nn.Parameter(torch.randn(label_size, h0_dim) * 0.05) 
            
        # (C) Hidden[i] <-> Hidden[i+1]
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
        """
        SpNCN Algorithm 1 (With Cost Monitoring)
        """
        # === 1. Update Phase ===
        # (A) Input & Label Layers
        s_h0 = self.hidden_layers[0].s
        
        # SynOps: s_h0 (Spikes) @ W_x
        # Cost: Active Spikes * Fan_out (Input Dim)
        z_pred_x = s_h0 @ self.W_x.t()
        self.input_layer.update_state(z_pred_x)
        
        if self.label_layer is not None:
            z_pred_y = s_h0 @ self.W_y.t()
            self.label_layer.update_state(z_pred_y)

        # (B) Hidden Layers
        for i, layer in enumerate(self.hidden_layers):
            total_input = 0
            
            # Local Error (-layer.e) is element-wise simple sub/add
            total_input += (-layer.e) 
            
            if i == 0:
                # Feedback: input_layer.e @ E_x.t()
                # e is (-1, 0, 1). This is a Signed SynOp.
                # Cost: Non-zero Errors * Fan_out (Hidden Dim)
                feedback = self.input_layer.e @ self.E_x.t()
                total_input += feedback
            else:
                e_lower = self.hidden_layers[i-1].e
                feedback = e_lower @ self.E[i-1].t()
                total_input += feedback
                
            layer.update_state(total_input)

        # === 2. Prediction & Error Phase ===
        # e = x - s (Sparse/Element-wise ops)
        self.input_layer.e = x_data - self.input_layer.x
        
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

    def manual_weight_update(self):
        """
        重み更新 (Dense MACs + Sparse Updates)
        """
        lr = self.config['alpha_u']
        beta = self.config['beta']
        
        with torch.no_grad():
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

def run_experiment(dataset_name='MNIST', mode='fixed_epochs'):
    """
    mode: 'target_acc' -> target_accに到達するまで学習 (max_epochs上限)
          'fixed_epochs' -> max_epochsまで指定回数学習 (target_accは無視)
    """
    print(f"\n=== Running SpNCN Cost Measurement on {dataset_name} ===")
    print(f"Mode: {mode}")
    if mode == 'target_acc':
        print(f"Goal: Reach Test Accuracy >= {CONFIG['target_acc']}%")
    else:
        print(f"Goal: Run for {CONFIG['max_epochs']} epochs")
    
    # データの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported in this script")

    # DataLoader
    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    # モデル構築
    model = SpNCN(
        hidden_sizes=[500, 500], 
        input_size=784, 
        label_size=10, 
        config=CONFIG
    ).to(CONFIG['device'])
    
    logs = []
    epoch = 0
    max_test_acc = 0.0
    steps = int(CONFIG['T_st'] / CONFIG['dt'])
    
    # ループ条件を max_epochs に統一し、内部で mode による分岐を行う
    while epoch < CONFIG['max_epochs']:
        # --- Training ---
        model.train()
        epoch_start = time.time()
        train_correct = 0
        train_samples = 0
        
        for batch_idx, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            
            targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)
            
            imgs_rate = torch.clamp(imgs, 0, 1)
            # CONFIG['num_steps'] does not exist in CONFIG. Using 'steps' calculated from T_st/dt.
            spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            out_spikes = 0
            
            for t in range(steps):
                x_t = spike_in[t]
                # 学習モード
                model.forward_dynamics(x_data=x_t, y_target=targets)
                model.manual_weight_update()
                model.clip_weights(20.0)
                out_spikes += model.label_layer.s
                
            _, pred = torch.max(out_spikes, 1)
            train_correct += (pred == lbls).sum().item()
            train_samples += lbls.size(0)

            # バッチごとの進捗表示
            print(f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_l)}] | Running Train Acc: {100 * train_correct / train_samples:.2f}%", end="")
        
        print() # エポック終了後に改行
        train_acc = 100 * train_correct / train_samples
        
        # --- Testing (Every Epoch) ---
        model.eval()
        test_correct = 0
        test_samples = 0
        
        for imgs, lbls in test_l:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
            
            model.reset_state(imgs.size(0), CONFIG['device'])
            out_spikes = 0
            
            for t in range(steps):
                x_t = spike_in[t]
                with torch.no_grad():
                    # テスト中のSynOpsもコストとして計上
                    model.forward_dynamics(x_data=x_t, y_target=None)
                out_spikes += model.label_layer.s
            
            _, pred = torch.max(out_spikes, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        max_test_acc = max(max_test_acc, test_acc)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}%")
        
        logs.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        epoch += 1
        
        # Target Accuracy Modeの場合の早期終了判定
        if mode == 'target_acc' and max_test_acc >= CONFIG['target_acc']:
            print(f"\nTarget Accuracy {CONFIG['target_acc']}% Reached at Epoch {epoch}.")
            break

    # 最終結果保存
    df = pd.DataFrame(logs)
    csv_path = f"{CONFIG['save_dir']}/cost_log_{dataset_name}_{mode}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nExperiment Finished.")
    print(f"Total Epochs: {epoch}")
    print(f"Log saved to {csv_path}")

if __name__ == "__main__":
    # モードを選択して実行
    # mode='target_acc': 精度到達まで
    # mode='fixed_epochs': CONFIG['max_epochs']回まで
    
    # ユーザーリクエスト: 指定したエポック数学習させてテストをする実験
    run_experiment('MNIST', mode='fixed_epochs')