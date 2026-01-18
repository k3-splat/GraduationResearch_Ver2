import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt' : 1.0,
    'T_st' : 200.0, # データ提示時間
    'T_r' : 1.0,    # 不応期
    'tau_j' : 10.0,
    'tau_m' : 20.0,
    'tau_tr' : 30.0,
    'tau_data' : 60.0,
    'kappa_j': 0.25,
    'gamma_m': 1.0,
    'R_m' : 1.0,
    'alpha_u' : 0.055,   # 学習率
    'alpha_gen' : 1e-4,  # 予測誤差の重み
    'alpha_disc' : 1.0,
    'thresh': 0.4,
    'batch_size': 64,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/SpNCN_Comparison',
    'max_freq': 63.75   # 【修正】追加: ポアソン生成用の最大周波数(Hz)
}

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
        self.x = None # Trace or Input
        self.j = None # Input Current State
        self.e_gen = None
        self.e_disc = None
        self.refrac_timer = None # 【追加】不応期タイマー

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)
        # 【追加】不応期タイマー初期化
        self.refrac_timer = torch.zeros(batch_size, self.dim, device=device)

    def switch_label_mode(self):
        self.is_label_layer = not self.is_label_layer

    def update_state(self, total_input_current):
        # ラベル層もテスト時(Inference Mode)はupdate_stateで動かす必要があるため条件を調整
        if not (self.is_data_layer or self.is_label_layer): 
            dt = self.cfg['dt']
            T_r = self.cfg['T_r']
            
            # 電流ダイナミクス (不応期に関わらず入力は積分されるモデルとする)
            d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
            self.j = self.j + (dt / self.cfg['tau_j']) * d_j
            
            # 【追加】不応期判定
            is_refractory = (self.refrac_timer > 0)

            # 膜電位ダイナミクス
            d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
            v_next = self.v + (dt / self.cfg['tau_m']) * d_v
            
            # 【修正】不応期中のニューロンは膜電位更新をスキップ (0に固定)
            # ※ 前のステップでリセットされていれば0のまま
            self.v = torch.where(is_refractory, self.v, v_next)
            
            # スパイク生成 (閾値超え かつ 不応期でない)
            spikes = (self.v > self.cfg['thresh']).float()
            spikes = spikes * (1.0 - is_refractory.float()) # 念のためマスク
            
            self.s = spikes
            
            # リセット
            self.v = self.v * (1 - spikes)
            
            # 【追加】不応期タイマーの更新
            # 1. 経過時間を引く
            self.refrac_timer = torch.clamp(self.refrac_timer - dt, min=0)
            # 2. スパイクしたニューロンのタイマーをセット
            self.refrac_timer = torch.where(spikes > 0, torch.tensor(T_r, device=self.v.device), self.refrac_timer)

            # トレース更新
            self.x = self.x - (1 / self.cfg['tau_tr']) * self.x + spikes
        

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
        self.W = nn.ParameterList() # Top-down (Upper -> Lower) [Generative]
        self.V = nn.ParameterList() # Bottom-up (Lower -> Upper) [Discriminative]
                    
        # Layer[i] <-> Layer[i+1]
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            # Xavier Uniform or Small Random
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.5))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.5))

        self.total_synops = 0.0
        self.z_data = None # データ層のトレース用
        self.z_label = None # ラベル層のトレース用

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)
            
            if layer.is_data_layer:
                # データ層用のトレース変数を初期化
                self.z_data = torch.zeros(batch_size, layer.dim, device=device)
            
            if layer.is_label_layer:
                self.z_label = torch.zeros(batch_size, layer.dim, device=device)

    def clip_weights(self, max_norm=20.0):
        with torch.no_grad():
            for w_list in [self.W, self.V]:
                for w in w_list:
                    norms = w.norm(p=2, dim=1, keepdim=True)
                    mask = norms > max_norm
                    w.data.copy_(torch.where(mask, w * (max_norm / (norms + 1e-8)), w))

    def forward_dynamics(self, x_data, y_target=None, training_mode=True):
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        # === 1. Update Phase ===
        # 学習時
        if training_mode and y_target is not None:
            # ニューロン活動更新
            for i, layer in enumerate(self.layers):
                total_input = 0

                # 学習時は，データ層とラベル層を除いたところでLIFニューロンとして活動する
                # ラベル層は教師信号で誤差計算するため、ここでの自発活動更新はスキップ
                if i > 0 and i < len(self.layers) - 1:
                    # 上からの予測/誤差フィードバック
                    if i < len(self.layers) - 1:
                        total_input += (- layer.e_gen)
                        e_disc_upper = self.layers[i+1].e_disc
                        total_input += torch.matmul(e_disc_upper, self.V[i]) 
                    
                    # 下からの予測/誤差フィードバック
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                layer.update_state(total_input)

            # 誤差計算
            for i, layer in enumerate(self.layers):
                if i == 0:
                    self.z_data += - self.z_data / self.config['tau_data'] + x_data
                    s_upper = self.layers[i+1].s
                    z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                    layer.e_gen = alpha_gen * (self.z_data - z_gen_pred)

                elif i < len(self.layers) - 1:
                    # Discriminative Error (Bottom-up Error)
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        s_lower = self.layers[i-1].s
                        z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                    # Generative Error (Top-down Error)
                    if i < len(self.layers) - 2:
                        s_upper = self.layers[i+1].s
                        z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                    else:
                        # 最後の隠れ層: ラベル層(Target)からの予測を受ける
                        z_gen_label = torch.matmul(y_target, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_label)
                
                else:
                    # ラベル層 (Output)
                    self.z_label += (-self.z_label / self.config['tau_data'] + y_target)
                    s_lower = self.layers[i-1].s
                    z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                    # 教師信号(Trace)との誤差
                    layer.e_disc = alpha_disc * (self.z_label - z_disc_pred)

        # テスト時
        else:
            # ニューロン活動更新
            for i, layer in enumerate(self.layers):
                total_input = 0

                if i > 0:
                    # 上からの予測/誤差フィードバック
                    if i < len(self.layers) - 1:
                        total_input += (- layer.e_gen)
                        e_disc_upper = self.layers[i+1].e_disc
                        total_input += torch.matmul(e_disc_upper, self.V[i]) 
                    
                    # 下からの予測/誤差フィードバック
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                layer.update_state(total_input)

            # 誤差計算
            for i, layer in enumerate(self.layers):
                if i == 0:
                    self.z_data += (- self.z_data / self.config['tau_data'] + x_data)
                    s_upper = self.layers[i+1].s
                    z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                    layer.e_gen = alpha_gen * (self.z_data - z_gen_pred)

                elif i < len(self.layers):
                    if i == 1:
                        z_disc_data = torch.matmul(self.z_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        s_lower = self.layers[i-1].s
                        z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                    if i < len(self.layers) - 1:
                        s_upper = self.layers[i+1].s
                        z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)

    def manual_weight_update(self, x_data, y_target=None):
        """
        ST-LRA Update Rule
        """
        alpha_u = self.config['alpha_u']

        for i in range(len(self.layers)):
            # Vの更新 (Discriminative weights)
            if i < len(self.layers) - 1:
                e_disc_upper = self.layers[i+1].e_disc
                if i == 0:
                    grad_V = torch.matmul(e_disc_upper.t(), x_data)
                else:
                    s_own = self.layers[i].s
                    grad_V = torch.matmul(e_disc_upper.t(), s_own)

                self.V[i] += alpha_u * grad_V

            # Wの更新 (Generative weights)
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
    
    # モデル構築
    layer_sizes = [784, 500, 500, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=CONFIG).to(CONFIG['device'])
    
    steps = int(CONFIG['T_st'] / CONFIG['dt'])
    logs = []

    with torch.no_grad():
        for epoch in range(CONFIG['epochs']):
            # --- Training ---
            model.train()
            epoch_start = time.time()
            model.total_synops = 0 
            
            for batch_idx, (imgs, lbls) in enumerate(train_l):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                # One-hotターゲット
                targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
                targets.scatter_(1, lbls.view(-1, 1), 1)
                
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
                
                # ターゲットをポアソンスパイクに変換
                spike_targets = generate_poisson_spikes(targets, steps, CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                sum_out_spikes = 0
                
                for t in range(steps):
                    x_t = spike_in[t]
                    # 時刻tのターゲットスパイク
                    y_t = spike_targets[t]
                    
                    model.forward_dynamics(x_data=x_t, y_target=y_t, training_mode=True)
                    
                    model.manual_weight_update(x_data=x_t, y_target=y_t)
                    model.clip_weights(20.0)
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx}")
        
            # --- Testing ---
            print("Switching label layer to Inference Mode (LIF)...")
            model.layers[-1].switch_label_mode()

            model.eval()
            test_correct = 0
            test_samples = 0
            
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                sum_out_spikes = 0
                
                for t in range(steps):
                    x_t = spike_in[t]
                    with torch.no_grad():
                        model.forward_dynamics(x_data=x_t, y_target=None, training_mode=False)
                    
                    sum_out_spikes += model.layers[-1].s
                
                _, pred = torch.max(sum_out_spikes, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
                
            test_acc = 100 * test_correct / test_samples
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch} DONE | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            print("Switching label layer back to Training Mode (Clamp)...")
            model.layers[-1].switch_label_mode()

            logs.append({
                'dataset': dataset_name,
                'epoch': epoch,
                'test_acc': test_acc,
                'time': epoch_time
            })
            
            df = pd.DataFrame(logs)
            df.to_csv(f"{CONFIG['save_dir']}/log_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    run_experiment('MNIST')