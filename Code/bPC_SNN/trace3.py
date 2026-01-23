import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import time
import pandas as pd
import os
from datetime import datetime

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt' : 0.25,
    'T_st' : 100.0, # データ提示時間
    'tau_j' : 2.0,
    'tau_m' : 5.0,
    'tau_tr' : 10.0,
    'tau_data' : 30.0,
    'kappa_j': 0.25,
    'gamma_m': 1.0,
    'R_m' : 1.0,
    'alpha_u' : 0.0005,   # 学習率
    'alpha_gen' : 1.0,  # 予測誤差の重み
    'alpha_disc' : 1.0,
    'thresh': 0.4,
    'batch_size': 64,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/bPC_SNN',
    'max_freq': 3000.0   # 【修正】追加: ポアソン生成用の最大周波数(Hz)
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
    def __init__(self, idx, output_dim, config, data=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = data
        
        # 内部状態
        self.v = None
        self.s = None
        self.x = None # Trace or Input
        self.j = None # Input Current State
        self.e_gen = None
        self.e_disc = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def switch_mode(self):
        self.is_data_layer = not self.is_data_layer

    def update_state(self, total_input_current):
        if not self.is_data_layer:
            dt = self.cfg['dt']
            
            # LIF Dynamics
            d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
            self.j = self.j + (dt / self.cfg['tau_j']) * d_j
            
            d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
            self.v = self.v + (dt / self.cfg['tau_m']) * d_v
            
            spikes = (self.v > self.cfg['thresh']).float()
            self.s = spikes
            self.v = self.v * (1 - spikes) 
            
            self.x = self.x + (-self.x / self.cfg['tau_tr'] + spikes)


class bPC_SNN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes

        # レイヤー生成
        for i, size in enumerate(layer_sizes):
            is_data = (i == 0) or (i == len(self.layer_sizes) - 1)
            self.layers.append(bPC_SNNLayer(idx=i, output_dim=size, config=config, data=is_data))
            
        # --- 重み定義 ---
        self.W = nn.ParameterList() # Top-down (Upper -> Lower) [Generative]
        self.V = nn.ParameterList() # Bottom-up (Lower -> Upper) [Discriminative]
                    
        # Layer[i] <-> Layer[i+1]
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            # Xavier Uniform or Small Random
            w = nn.Parameter(torch.empty(dim_lower, dim_upper))
            v = nn.Parameter(torch.empty(dim_upper, dim_lower))
            
            # Xavier Initialization
            nn.init.xavier_uniform_(w)
            nn.init.xavier_uniform_(v)
            
            self.W.append(w)
            self.V.append(v)

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=20.0):
        for w_list in [self.W, self.V]:
            for w in w_list:
                norms = w.norm(p=2, dim=1, keepdim=True)
                mask = norms > max_norm
                w.data.copy_(torch.where(mask, w * (max_norm / (norms + 1e-8)), w))
    
    def run_feedforward_initialization(self, x_data, is_Testing=False):
        """
        bPCの初期化フェーズを再現するメソッド。
        下位層からのフィードフォワードスイープにより、各層の初期状態(v, j, s)を設定する。
        f(x_l)として生成されたスパイク(s)を次の層への入力として使用する。
        """
        # 入力データ(Rate/Intensity)を最初のスパイク入力とみなしてスタート
        # Layer 0はData Layerなので内部状態の更新対象ではないが、sweepの起点となる
        current_input = x_data 

        # 学習時
        if not is_Testing:
            for i in range(1, len(self.layers) - 1):
                layer = self.layers[i]
                
                # Bottom-up入力の計算: 前の層の出力(current_input) x V
                # self.V[i-1] は Layer i と Layer i-1 を結ぶ Bottom-up 重み
                # Vのshapeは (dim_upper, dim_lower) なので、 input @ V.t() で (Batch, dim_upper)
                weight = self.V[i-1]
                input_signal = torch.matmul(current_input, weight.t())
                
                # 状態の初期化 (瞬時に充電されると仮定)
                layer.j = input_signal
                layer.v = self.config['R_m'] * input_signal 
                
                # スパイク生成 (f(x_l) = s)
                spikes = (layer.v > self.config['thresh']).float()
                layer.s = spikes
                
                # スパイクしたニューロンの膜電位をリセット (Soft Reset)
                layer.v = layer.v * (1 - spikes)
                
                # トレースの初期化 (オプション: スパイクが発生したならトレースも反応させる)
                if hasattr(layer, 'x') and layer.x is not None:
                    # 1ステップ分の更新を適用、あるいは単にスパイク値をセット
                    layer.x = spikes

                # 次の層への入力として、今生成されたスパイクを使用する
                current_input = spikes

        else:
            for i in range(1, len(self.layers)):
                layer = self.layers[i]
                
                # Bottom-up入力の計算: 前の層の出力(current_input) x V
                # self.V[i-1] は Layer i と Layer i-1 を結ぶ Bottom-up 重み
                # Vのshapeは (dim_upper, dim_lower) なので、 input @ V.t() で (Batch, dim_upper)
                weight = self.V[i-1]
                input_signal = torch.matmul(current_input, weight.t())
                
                # 状態の初期化 (瞬時に充電されると仮定)
                layer.j = input_signal
                layer.v = self.config['R_m'] * input_signal 
                
                # スパイク生成 (f(x_l) = s)
                spikes = (layer.v > self.config['thresh']).float()
                layer.s = spikes
                
                # スパイクしたニューロンの膜電位をリセット (Soft Reset)
                layer.v = layer.v * (1 - spikes)
                
                # トレースの初期化 (オプション: スパイクが発生したならトレースも反応させる)
                if hasattr(layer, 'x') and layer.x is not None:
                    # 1ステップ分の更新を適用、あるいは単にスパイク値をセット
                    layer.x = spikes

                # 次の層への入力として、今生成されたスパイクを使用する
                current_input = spikes

    def forward_dynamics(self, x_data, y_target=None):
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        # === 1. Update Phase ===
        # 学習時
        if y_target is not None:
            # ニューロン活動更新
            for i, layer in enumerate(self.layers):
                total_input = 0

                if i > 0 and i < len(self.layers) - 1:
                    # 自層の識別的予測誤差&下からの予測誤差フィードバック
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                    # 自層の生成的予測誤差&上からの予測誤差フィードバック
                    total_input += (- layer.e_gen)
                    e_disc_upper = self.layers[i+1].e_disc
                    total_input += torch.matmul(e_disc_upper, self.V[i])

                layer.update_state(total_input)

            # 誤差計算
            for i, layer in enumerate(self.layers):
                if i == 0:
                    x_upper = self.layers[i+1].x
                    z_gen_pred = torch.matmul(x_upper, self.W[i].t())
                    layer.e_gen = alpha_gen * (x_data - z_gen_pred)

                elif i < len(self.layers) - 1:
                    # Discriminative Error (Bottom-up Error)
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        x_lower = self.layers[i-1].x
                        z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                    # Generative Error (Top-down Error)
                    if i < len(self.layers) - 2:
                        x_upper = self.layers[i+1].x
                        z_gen_pred = torch.matmul(x_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                    else:
                        # 最後の隠れ層: ラベル層(Target)からの予測を受ける
                        z_gen_label = torch.matmul(y_target, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_label)
                
                else:
                    # ラベル層 (Output)
                    x_lower = self.layers[i-1].x
                    z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (y_target - z_disc_pred)

        # テスト時
        else:
            # ニューロン活動更新
            for i, layer in enumerate(self.layers):
                total_input = 0

                if i > 0:
                    # 自層の識別的予測誤差&下からの予測誤差フィードバック
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                    if i < len(self.layers) - 1:
                        # 自層の生成的予測誤差&上からの予測誤差フィードバック
                        total_input += (- layer.e_gen)
                        e_disc_upper = self.layers[i+1].e_disc
                        total_input += torch.matmul(e_disc_upper, self.V[i])

                    layer.update_state(total_input)

            # 誤差計算
            for i, layer in enumerate(self.layers):
                if i == 0:
                    x_upper = self.layers[i+1].x
                    z_gen_pred = torch.matmul(x_upper, self.W[i].t())
                    layer.e_gen = alpha_gen * (x_data - z_gen_pred)

                else:
                    # Discriminative Error (Bottom-up Error)
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        x_lower = self.layers[i-1].x
                        z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                    # Generative Error (Top-down Error)
                    if i < len(self.layers) - 1:
                        x_upper = self.layers[i+1].x
                        z_gen_pred = torch.matmul(x_upper, self.W[i].t())
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
                    x_own = self.layers[i].x
                    grad_V = torch.matmul(e_disc_upper.t(), x_own)

                self.V[i] += alpha_u * grad_V

            # Wの更新 (Generative weights)
            if i > 0:
                e_gen_lower = self.layers[i-1].e_gen
                
                if i == len(self.layers) - 1:
                    if y_target is not None:
                        grad_W = torch.matmul(e_gen_lower.t(), y_target)
                        self.W[i-1] += alpha_u * grad_W
                else:
                    x_own = self.layers[i].x
                    grad_W = torch.matmul(e_gen_lower.t(), x_own)
                    self.W[i-1] += alpha_u * grad_W

def run_experiment(dataset_name='MNIST'):
    print(f"\n=== Running bPC-SNN on {dataset_name} ===")
    
    # --- ファイル名と実行日時設定 ---
    try:
        # スクリプトファイル名を取得 (拡張子なし)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        script_name = "notebook_execution"

    # 日時を取得
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存パスの決定: log_[スクリプト名]_[データセット]_[日時].csv
    save_file_path = f"{CONFIG['save_dir']}/log_{script_name}_{dataset_name}_{timestamp}.csv"
    print(f"Results will be saved to: {save_file_path}")
    # ----------------------------

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
            
            for batch_idx, (imgs, lbls) in enumerate(train_l):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                # One-hotターゲット
                targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
                targets.scatter_(1, lbls.view(-1, 1), 1)
                
                imgs_rate = torch.clamp(imgs, 0, 1)
                # spike_in = spikegen.rate(imgs_rate, steps)
                # spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                # --- 追加: Feedforward Initialization ---
                # 各バッチの開始時に、フィードフォワードスイープによる初期化を実行
                model.run_feedforward_initialization(imgs_rate)
                # -------------------------------------
                
                for _ in range(steps):
                    model.forward_dynamics(x_data=imgs_rate, y_target=targets)
                    model.manual_weight_update(x_data=imgs_rate, y_target=targets)
                    model.clip_weights(20.0)
                
                if batch_idx % 100 == 0:
                    # print(sum_out_spikes)
                    print(f"Epoch {epoch} | Batch {batch_idx}")
                    print(model.layers[-1].e_disc)
        
            # --- Testing ---
            print("Switching label layer to Inference Mode (LIF)...")
            model.layers[-1].switch_mode()

            model.eval()
            test_correct = 0
            test_samples = 0
            
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                # spike_in = spikegen.rate(imgs_rate, steps)
                # spike_in = generate_poisson_spikes(imgs_rate, steps, CONFIG)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                # --- 追加: Feedforward Initialization ---
                # テスト時も同様に初期化を実行
                model.run_feedforward_initialization(imgs_rate)
                # -------------------------------------
                
                for _ in range(steps):
                    model.forward_dynamics(x_data=imgs_rate, y_target=None)
                    # 最終層のトレースを取得

                out_trace = model.layers[-1].x
                
                _, pred = torch.max(out_trace, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
                
            test_acc = 100 * test_correct / test_samples
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch} DONE | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            print("Switching label layer back to Training Mode (Clamp)...")
            model.layers[-1].switch_mode()

            logs.append({
                'dataset': dataset_name,
                'epoch': epoch,
                'test_acc': test_acc,
                'time': epoch_time
            })
            
            # --- ログ保存 (ファイル名はループの外で決定済み) ---
            df = pd.DataFrame(logs)
            df.to_csv(save_file_path, index=False)

if __name__ == "__main__":
    run_experiment('MNIST')