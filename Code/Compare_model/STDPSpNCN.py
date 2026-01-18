import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 (Diehl & Cook 2015 の要素を追加) ---
CONFIG = {
    'dt': 1.0,          # 微小時間 (ms)
    'T_r' : 1.0,         # 絶対不応期 (ms)
    'T_st' : 200,        # データ提示時間 (ms) - 論文に合わせて長めに設定
    'tau_m': 20.0,      # 時定数（膜電位） - 論文に合わせて長く設定 (積分効果向上)
    'tau_j': 10.0,       # 時定数（入力電流）
    'tau_tr': 30.0,      # 時定数（ローパスフィルタ）
    'tau_dt' : 100,      # 時定数（入力データ用）
    'tau_theta': 1e4,    # 時定数（適応的閾値） - 非常にゆっくり減衰
    'theta_plus': 0.05,  # 適応的閾値の増加分
    'inhib_str': 1.5,    # 側抑制の強さ (Lateral Inhibition Strength)
    
    'kappa_j': 0.25,     # 漏れ係数（入力電流）
    'gamma_m': 1.0,      # 漏れ係数（膜電位）
    'R_m': 1.0,          # 抵抗
    'alpha_u': 0.001,    # 重み学習率 (抑制が入るため少し下げて調整)
    'beta': 1.0,         # 誤差重み学習率
    'thresh': 0.4,       # ベース発火閾値
    'max_freq' : 63.75,  # 最大周波数 (Hz)
    
    # 再提示設定
    'min_spikes': 5,     # 学習に必要な最小スパイク数
    'gain_boost': 1.5,   # 再提示時のゲイン倍率 (Hz増加の代わりに入力倍率として実装)
    'max_attempts': 3,   # 最大再提示回数
    
    'batch_size': 64,
    'target_acc': 95.0,
    'max_epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_DiehlCook'
}

# --- ポアソンエンコーディング関数 (ゲイン調整対応) ---
def generate_poisson_spikes(data, num_steps, config, gain=1.0):
    """
    入力データ(0~1)を受け取り、ポアソンスパイク列(Time, Batch, Dim)を生成する
    gain: 再提示時に強度を上げるための係数
    """
    device = data.device
    batch_size, dim = data.shape
    
    dt = config['dt']
    max_freq = config['max_freq']
    
    # 確率 p = (Data * Gain) * freq(Hz) * dt(ms) / 1000
    # Gainをかけることで擬似的に輝度(周波数)を上げる
    firing_probs = data * gain * max_freq * (dt / 1000.0)
    firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
    
    firing_probs_expanded = firing_probs.unsqueeze(0).expand(num_steps, batch_size, dim)
    spikes = torch.bernoulli(firing_probs_expanded)
    
    return spikes

# 結果保存用ディレクトリ作成
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# --- エネルギー定数 ---
E_MUL = 3.7e-12  # 3.7 pJ
E_ADD = 0.9e-12  # 0.9 pJ

class CostMonitor:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_synops = 0.0
        self.total_dense_muls = 0.0
        self.total_dense_adds = 0.0
        self.start_time = time.time()
    def add_synops(self, count):
        self.total_synops += count
    def add_dense_ops(self, muls, adds):
        self.total_dense_muls += muls
        self.total_dense_adds += adds
    def get_energy(self):
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
        self.monitor = monitor 
        self.is_data_layer = is_data_layer
        
    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.e = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)
        
        # [New] 適応的閾値 (Homeostasis)
        # 初期値は0。発火しすぎるとプラスになり、発火しにくくなる。
        self.theta = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current, lateral_inhibition=None):
        """
        lateral_inhibition: Tensor (batch, dim) 
                            正の値を受け取り、電流から減算する
        """
        dt = self.cfg['dt']
        batch_size = self.j.size(0)
        n_neurons = batch_size * self.dim

        # 1. 不応期更新
        self.ref_count = (self.ref_count - dt).clamp(min=0)
        is_active = (self.ref_count <= 0).float()

        # 2. 電流積分 (LIF)
        # 側抑制 (Lateral Inhibition) の適用
        effective_input = total_input_current
        if lateral_inhibition is not None:
            effective_input = effective_input - lateral_inhibition
            # コスト: 引き算 (Add)
            self.monitor.add_dense_ops(muls=0, adds=n_neurons)

        d_j = (-self.cfg['kappa_j'] * self.j + effective_input)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        v_next = self.v + (dt / self.cfg['tau_m']) * d_v
        
        # 3. 適応的閾値の減衰 (Theta Decay)
        # theta = theta - theta/tau_theta
        self.theta = self.theta - (dt / self.cfg['tau_theta']) * self.theta
        
        self.v = is_active * v_next
        
        # コスト: LIF + Theta Decay
        # Theta decay needs 1 mul, 1 sub per neuron
        self.monitor.add_dense_ops(muls=5 * n_neurons + n_neurons, adds=4 * n_neurons + n_neurons)

        # 4. Spike Generation (Adaptive Threshold)
        # 実際の閾値 = ベース閾値 + theta
        effective_thresh = self.cfg['thresh'] + self.theta
        spikes = (self.v > effective_thresh).float()
        self.s = spikes
        
        # Reset & Refractory
        self.v = self.v * (1 - spikes)
        self.ref_count = torch.where(spikes.bool(), torch.tensor(self.cfg['T_r'], device=self.v.device), self.ref_count)
        
        # [New] Theta Update (Increase on spike)
        # スパイクしたニューロンの閾値を上げる
        self.theta = self.theta + spikes * self.cfg['theta_plus']
        
        self.monitor.add_dense_ops(muls=1 * n_neurons, adds=2 * n_neurons) # thresh calc + update

        # 5. Trace Update
        if self.is_data_layer:
            self.x = self.x - self.x / self.cfg['tau_dt'] + spikes
        else:
            self.x = self.x - self.x / self.cfg['tau_tr'] + spikes
            self.monitor.add_dense_ops(muls=1 * n_neurons, adds=2 * n_neurons)

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
        
        # self.z_data はここで定義すると self.input_layer.x が未定義のためエラーになります。
        # reset_state で定義します。
        
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

    def reset_state(self, batch_size, device, reset_theta=False):
        """
        reset_theta: Trueなら適応的閾値もリセット（エポックの最初など）。
                     再提示(Re-presentation)のときはFalseにして、閾値上昇を維持する場合と、
                     リセットする場合があるが、Diehl&Cookでは閾値は維持したまま画像を再提示する学習に近い。
                     ここではニューロン状態(v, j)はリセットするが、学習された構造やthetaの扱いは文脈による。
                     今回は「入力再提示」なので、v, j はリセットし、thetaは維持するのが一般的（ホメオスタシスの継続）。
        """
        self.input_layer.init_state(batch_size, device)
        
        # [修正] ここで z_data を初期化 (Batch, Input_Dim)
        self.z_data = torch.zeros(batch_size, self.input_layer.dim, device=device)

        if self.label_layer is not None:
            self.label_layer.init_state(batch_size, device)
        for layer in self.hidden_layers:
            # thetaを保持するために個別にリセット処理
            old_theta = layer.theta if hasattr(layer, 'theta') else None
            layer.init_state(batch_size, device)
            if not reset_theta and old_theta is not None:
                layer.theta = old_theta # 閾値履歴を復元

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
        # === 1. Update Phase ===
        s_h0 = self.hidden_layers[0].s
        z_pred_x = s_h0 @ self.W_x.t()
        if training_mode: self.monitor.add_synops(s_h0.sum().item() * self.input_layer.dim)
        self.input_layer.update_state(z_pred_x) # Input layer has no inhibition
        
        if self.label_layer is not None:
            z_pred_y = s_h0 @ self.W_y.t()
            if training_mode: self.monitor.add_synops(s_h0.sum().item() * self.label_layer.dim)
            self.label_layer.update_state(z_pred_y)

        # Hidden Layers Update
        for i, layer in enumerate(self.hidden_layers):
            total_input = (-layer.e) 
            
            # Feedback calculation
            if i == 0:
                feedback = self.input_layer.e @ self.E_x.t()
                if training_mode:
                    n_errors = (self.input_layer.e.abs() > 0.01).sum().item()
                    self.monitor.add_synops(n_errors * layer.dim)
            else:
                e_lower = self.hidden_layers[i-1].e
                feedback = e_lower @ self.E[i-1].t()
                if training_mode:
                    n_errors = (e_lower.abs() > 0.01).sum().item()
                    self.monitor.add_synops(n_errors * layer.dim)
            
            total_input += feedback
            
            # [New] Calculate Lateral Inhibition
            # 簡略化: 全結合抑制 (自分以外)
            # inhib = (sum(spikes) - my_spike) * strength
            #       = sum(spikes) * strength - my_spike * strength
            total_spikes_in_layer = layer.s.sum(dim=1, keepdim=True) # (Batch, 1)
            lateral_signal = (total_spikes_in_layer - layer.s) * self.config['inhib_str']
            
            # コスト: Sum (N adds) + Sub/Mul (2N ops)
            if training_mode:
                self.monitor.add_dense_ops(muls=layer.dim * layer.j.size(0), adds=layer.dim * layer.j.size(0))

            layer.update_state(total_input, lateral_inhibition=lateral_signal)

        # === 2. Prediction & Error Phase ===
        self.z_data += - self.z_data / self.config['tau_dt'] + x_data
        self.input_layer.e = self.z_data - self.input_layer.x
        if self.label_layer is not None:
            if y_target is not None:
                self.label_layer.e = y_target - self.label_layer.s
            else:
                self.label_layer.e = torch.zeros_like(self.label_layer.s)

        for i, layer in enumerate(self.hidden_layers):
            if i < len(self.hidden_layers) - 1:
                s_upper = self.hidden_layers[i+1].s
                z_pred = s_upper @ self.W[i].t()
                if training_mode: self.monitor.add_synops(s_upper.sum().item() * layer.dim)
                layer.e = layer.x - z_pred
            else:
                layer.e = torch.zeros_like(layer.x)

    def manual_weight_update(self):
        # 変更なし（前述のコードと同じ）
        lr = self.config['alpha_u']
        beta = self.config['beta']
        with torch.no_grad():
            s_h0 = self.hidden_layers[0].s
            e_x = self.input_layer.e
            batch_size = s_h0.size(0)
            n_macs_wx = (batch_size * self.input_layer.dim * self.hidden_layers[0].dim) + \
                        (self.input_layer.dim * self.hidden_layers[0].dim)
            self.monitor.add_dense_ops(muls=n_macs_wx, adds=n_macs_wx)
            self.W_x += lr * (e_x.t() @ s_h0)
            self.E_x += lr * beta * (s_h0.t() @ e_x)
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
                self.E[i] += lr * beta * (s_upper.t() @ e_lower)

def run_experiment(dataset_name='MNIST', mode='fixed_epochs'):
    print(f"\n=== Running SpNCN with Diehl&Cook Mechanisms ===")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported")

    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    monitor = CostMonitor()
    
    model = SpNCN(
        hidden_sizes=[500,500], # 実験のため1層に簡略化
        input_size=784, 
        label_size=10, 
        config=CONFIG,
        monitor=monitor
    ).to(CONFIG['device'])
    
    logs = []
    epoch = 0
    max_test_acc = 0.0
    steps = int(CONFIG['T_st'] / CONFIG['dt'])
    
    while epoch < CONFIG['max_epochs']:
        # --- Training ---
        model.train()
        train_correct = 0
        train_samples = 0
        re_presentations = 0 # 再提示回数カウンタ
        
        for batch_idx, (imgs, lbls) in enumerate(train_l):
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
            targets.scatter_(1, lbls.view(-1, 1), 1)
            imgs_rate = torch.clamp(imgs, 0, 1)

            # [New] 再提示ループ (Re-presentation Loop)
            # バッチ内のニューロン活動が十分かチェックし、不足ならゲインを上げてやり直す
            current_gain = 1.0
            attempt = 0
            
            while attempt < CONFIG['max_attempts']:
                spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG, gain=current_gain)
                
                # 状態リセット (Thetaは維持)
                model.reset_state(imgs.size(0), CONFIG['device'], reset_theta=False)
                out_spikes = 0
                
                # スパイク数監視用のテンソル
                # shape: (Batch, HiddenDim)
                hidden_spikes_count = torch.zeros(imgs.size(0), model.hidden_layers[0].dim, device=CONFIG['device'])
                
                # 時間発展
                for t in range(steps):
                    x_t = spike_in[t]
                    model.forward_dynamics(x_data=x_t, y_target=targets, training_mode=True)
                    model.manual_weight_update()
                    model.clip_weights(20.0)
                    
                    out_spikes += model.label_layer.s
                    hidden_spikes_count += model.hidden_layers[0].s
                
                # チェック: バッチ内の各サンプルについて、少なくとも min_spikes 発火したか？
                # Diehl論文では「5スパイク未満なら」再提示
                # ここでは簡易化のため、バッチ平均ではなく「バッチ内の誰かが発火不足なら全体やり直し」
                # あるいは「発火不足のサンプルだけ」やり直すのが理想だが、バッチ処理だと難しい。
                # ここでは、「隠れ層全体での平均発火が著しく低い場合」に再提示を行うロジックにします。
                
                avg_spikes = hidden_spikes_count.mean().item() # ニューロン1個あたりの平均スパイク数
                
                if avg_spikes >= CONFIG['min_spikes']:
                    break # 十分発火したので終了
                
                # 発火不足 -> ゲインを上げて再トライ
                current_gain *= CONFIG['gain_boost']
                attempt += 1
                re_presentations += 1
            
            _, pred = torch.max(out_spikes, 1)
            train_correct += (pred == lbls).sum().item()
            train_samples += lbls.size(0)

            print(f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_l)}] | Train Acc: {100 * train_correct / train_samples:.2f}% | Re-tries: {re_presentations}", end="")
        
        print()
        train_acc = 100 * train_correct / train_samples
        
        # --- Testing ---
        model.eval()
        test_correct = 0
        test_samples = 0
        
        for imgs, lbls in test_l:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            imgs_rate = torch.clamp(imgs, 0, 1)
            spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG) # テスト時はゲイン1.0固定
            
            model.reset_state(imgs.size(0), CONFIG['device'], reset_theta=False) # テスト時もTheta履歴は使う
            out_spikes = 0
            
            for t in range(steps):
                x_t = spike_in[t]
                with torch.no_grad():
                    model.forward_dynamics(x_data=x_t, y_target=None, training_mode=True)
                out_spikes += model.label_layer.s
            
            _, pred = torch.max(out_spikes, 1)
            test_correct += (pred == lbls).sum().item()
            test_samples += lbls.size(0)
            
        test_acc = 100 * test_correct / test_samples
        max_test_acc = max(max_test_acc, test_acc)
        
        current_energy = monitor.get_energy()
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        print(f"  Energy: {current_energy:.2e}J")
        
        logs.append({'epoch': epoch+1, 'train_acc': train_acc, 'test_acc': test_acc, 'energy': current_energy})
        epoch += 1

    df = pd.DataFrame(logs)
    csv_path = f"{CONFIG['save_dir']}/log_diehl_cook.csv"
    df.to_csv(csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    run_experiment('MNIST', mode='fixed_epochs')