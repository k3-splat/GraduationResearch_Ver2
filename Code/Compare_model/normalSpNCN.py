import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt': 1.0,          # 微小時間
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

# 結果保存用ディレクトリ作成
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# --- エネルギー定数 (45nm CMOS process, 32-bit float) ---
# 参照: 進捗報告8___Gemini用.pdf
E_MUL = 3.7e-12  # 3.7 pJ
E_ADD = 0.9e-12  # 0.9 pJ

class CostMonitor:
    """計算コストを集計するクラス"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_synops = 0.0      # 疎な加算操作 (Spike * Weight)
        self.total_dense_muls = 0.0  # 密な乗算 (Decay, Weight Update scaling)
        self.total_dense_adds = 0.0  # 密な加算 (Neuron integration, Weight Update accumulation)
        self.start_time = time.time()
        self.end_time = 0.0

    def add_synops(self, count):
        self.total_synops += count

    def add_dense_ops(self, muls, adds):
        self.total_dense_muls += muls
        self.total_dense_adds += adds

    def get_energy(self):
        """総エネルギー (J) を計算"""
        energy_synops = self.total_synops * E_ADD
        energy_dense = (self.total_dense_muls * E_MUL) + (self.total_dense_adds * E_ADD)
        return energy_synops + energy_dense

    def get_time(self):
        return time.time() - self.start_time

class SpNCNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, monitor, is_data_layer=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.monitor = monitor # CostMonitorへの参照
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
        batch_size = self.j.size(0)

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
        
        # コスト計上: マスク処理を行っても並列計算上は全ニューロン分の演算が走るため、Worst caseで計上
        n_neurons = batch_size * self.dim
        self.monitor.add_dense_ops(muls=5 * n_neurons, adds=4 * n_neurons)

        # --- 3. Spike Generation ---
        # 閾値を超え、かつ不応期でない(activeな)ニューロンのみ発火
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        
        # Reset: 発火したニューロンの膜電位をリセット (v = 0)
        self.v = self.v * (1 - spikes)
        
        # 不応期タイマーの設定: 発火したニューロンの ref_count を T_r に設定
        # spikeが発生した箇所のref_countをT_rで上書き
        self.ref_count = torch.where(spikes.bool(), torch.tensor(T_r, device=self.v.device), self.ref_count)
        
        self.monitor.add_dense_ops(muls=1 * n_neurons, adds=1 * n_neurons) # sub is add-like
        
        # --- 4. Trace Update ---
        if self.is_data_layer:
            self.x = spikes
        else:
            # x = x - x/tau + spikes
            self.x = self.x - self.x / self.cfg['tau_tr'] + spikes
            self.monitor.add_dense_ops(muls=1 * n_neurons, adds=2 * n_neurons)
        

class SpNCN(nn.Module):
    def __init__(self, hidden_sizes, input_size, label_size=None, config=None, monitor=None):
        super().__init__()
        self.config = config
        self.monitor = monitor
        self.layers = nn.ModuleList()
        
        # --- 1. 入力層 (Layer 0) ---
        self.input_layer = SpNCNLayer(idx='input', output_dim=input_size, config=config, monitor=monitor, is_data_layer=True)
        
        # --- 2. ラベル層 (Layer Label) ---
        self.label_size = label_size
        if label_size is not None:
            self.label_layer = SpNCNLayer(idx='label', output_dim=label_size, config=config, monitor=monitor, is_data_layer=True)
        else:
            self.label_layer = None
            
        # --- 3. 隠れ層 (Hidden Layers) ---
        self.hidden_layers = nn.ModuleList()
        for i, size in enumerate(hidden_sizes):
            self.hidden_layers.append(SpNCNLayer(idx=i, output_dim=size, config=config, monitor=monitor, is_data_layer=False))
            
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

    def forward_dynamics(self, x_data, y_target=None, training_mode=True):
        """
        SpNCN Algorithm 1 (With Cost Monitoring)
        """
        # === 1. Update Phase ===
        # (A) Input & Label Layers
        s_h0 = self.hidden_layers[0].s
        
        # SynOps: s_h0 (Spikes) @ W_x
        # Cost: Active Spikes * Fan_out (Input Dim)
        z_pred_x = s_h0 @ self.W_x.t()
        if training_mode:
            self.monitor.add_synops(s_h0.sum().item() * self.input_layer.dim)
            
        self.input_layer.update_state(z_pred_x)
        
        if self.label_layer is not None:
            z_pred_y = s_h0 @ self.W_y.t()
            if training_mode:
                self.monitor.add_synops(s_h0.sum().item() * self.label_layer.dim)
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
                if training_mode:
                    # eはfloatだが値は-1,0,1なので非ゼロ要素数をカウント
                    n_errors = (self.input_layer.e.abs() > 0.01).sum().item()
                    self.monitor.add_synops(n_errors * layer.dim)
            else:
                e_lower = self.hidden_layers[i-1].e
                feedback = e_lower @ self.E[i-1].t()
                total_input += feedback
                if training_mode:
                    n_errors = (e_lower.abs() > 0.01).sum().item()
                    self.monitor.add_synops(n_errors * layer.dim)
                
            layer.update_state(total_input)

        # === 2. Prediction & Error Phase ===
        # e = x - s (Sparse/Element-wise ops)
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
                
                # SynOps: Upper Spikes @ W
                if training_mode:
                    self.monitor.add_synops(s_upper.sum().item() * layer.dim)
                    
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
            
            # W_x update
            batch_size = s_h0.size(0)
            
            n_macs_wx = (batch_size * self.input_layer.dim * self.hidden_layers[0].dim) + \
                        (self.input_layer.dim * self.hidden_layers[0].dim)
            self.monitor.add_dense_ops(muls=n_macs_wx, adds=n_macs_wx)
            
            self.W_x += lr * (e_x.t() @ s_h0)
            self.E_x += lr * beta * (s_h0.t() @ e_x)
            self.monitor.add_dense_ops(muls=n_macs_wx, adds=n_macs_wx)
            
            if self.label_layer is not None:
                e_y = self.label_layer.e
                n_macs_wy = (batch_size * self.label_layer.dim * self.hidden_layers[0].dim) + \
                            (self.label_layer.dim * self.hidden_layers[0].dim)
                self.monitor.add_dense_ops(muls=n_macs_wy, adds=n_macs_wy)
                self.W_y += lr * (e_y.t() @ s_h0)
                
            for i in range(len(self.hidden_layers) - 1):
                s_upper = self.hidden_layers[i+1].s
                e_lower = self.hidden_layers[i].e
                
                dim_lower = self.hidden_layers[i].dim
                dim_upper = self.hidden_layers[i+1].dim
                n_macs = (batch_size * dim_lower * dim_upper) + (dim_lower * dim_upper)
                
                self.monitor.add_dense_ops(muls=n_macs, adds=n_macs)
                self.W[i] += lr * (e_lower.t() @ s_upper)
                
                self.monitor.add_dense_ops(muls=n_macs, adds=n_macs)
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
    
    monitor = CostMonitor()
    
    # モデル構築
    model = SpNCN(
        hidden_sizes=[500, 500], 
        input_size=784, 
        label_size=10, 
        config=CONFIG,
        monitor=monitor
    ).to(CONFIG['device'])
    
    logs = []
    epoch = 0
    max_test_acc = 0.0
    steps = int(CONFIG['T_st'] / CONFIG['dt'])
    
    # ループ条件を max_epochs に統一し、内部で mode による分岐を行う
    while epoch < CONFIG['max_epochs']:
        # --- Training ---
        model.train()
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
                # 学習モード: コスト加算あり
                model.forward_dynamics(x_data=x_t, y_target=targets, training_mode=True)
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
                    model.forward_dynamics(x_data=x_t, y_target=None, training_mode=True)
                out_spikes += model.label_layer.s
            
            _, pred = torch.max(out_spikes, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        max_test_acc = max(max_test_acc, test_acc)
        
        # 累積コスト取得
        current_time = monitor.get_time()
        current_energy = monitor.get_energy()
        current_synops = monitor.total_synops
        current_macs = monitor.total_dense_muls + monitor.total_dense_adds # MACs近似
        
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        print(f"  Cumul Time: {current_time:.1f}s | Energy: {current_energy:.2e}J | SynOps: {current_synops:.2e} | MACs: {current_macs:.2e}")
        
        logs.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cumul_time': current_time,
            'cumul_energy': current_energy,
            'cumul_synops': current_synops,
            'cumul_macs': current_macs
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
    print(f"Total Time: {monitor.get_time():.2f} s")
    print(f"Total Energy: {monitor.get_energy():.4e} J")
    print(f"Total SynOps: {monitor.total_synops:.4e}")
    print(f"Total MACs (Dense Ops): {(monitor.total_dense_muls + monitor.total_dense_adds):.4e}")
    print(f"Log saved to {csv_path}")

if __name__ == "__main__":
    # モードを選択して実行
    # mode='target_acc': 精度到達まで
    # mode='fixed_epochs': CONFIG['max_epochs']回まで
    
    # ユーザーリクエスト: 指定したエポック数学習させてテストをする実験
    run_experiment('MNIST', mode='fixed_epochs')