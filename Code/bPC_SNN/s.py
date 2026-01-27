import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import time
import pandas as pd
import os
from datetime import datetime

# ==========================================
#  Hyperparameters & Configuration
# ==========================================
CONFIG = {
    # --- Time & Simulation Settings ---
    'dt' : 1.0,           # タイムステップ (ms)
    'T_st' : 40.0,        # データ提示時間 (ms) - 短時間提示
    
    # --- Optimized Time Constants for Fast Convergence ---
    'tau_j' : 4.0,        # シナプス時定数 (高速追従)
    'tau_m' : 8.0,        # 膜時定数 (高速積分・収束)
    'tau_tr' : 15.0,      # トレース時定数 (学習履歴の早期更新)
    't_ref' : 2.0,        # 不応期 (ms) - 過剰発火防止
    
    # --- Neuron Parameters ---
    'kappa_j': 0.25,      # Leak factor for current (optional use)
    'gamma_m': 1.0,       # Leak factor for voltage
    'R_m' : 1.0,          # Resistance
    'thresh': 0.4,        # 発火閾値
    
    # --- Learning & Network ---
    'alpha_u' : 0.001,    # 学習率
    'alpha_gen' : 1.0,    # 生成誤差の重み
    'alpha_disc' : 1.0,   # 識別誤差の重み
    'batch_size': 64,
    'hidden_size': 500,   # 小型化: 隠れ層のニューロン数
    'epochs': 10,
    
    # --- Input Control ---
    'input_gain': 0.25,   # 入力確率のスケーリング (最大発火率を25%に制限)
    
    # --- Early Stopping (Settling) ---
    'settle_threshold': 0.01, # 誤差変動率が1%を切ったら安定とみなす
    'min_steps': 20,          # 最低回すステップ数 (初期過渡応答を無視)
    'patience': 3,            # 安定状態が何ステップ続いたら打ち切るか
    
    # --- System ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/bPC_SNN_Fast'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ==========================================
#  Layer Definitions
# ==========================================

class bPC_SNNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, data=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = data
        
        # State Variables
        self.v = None         # Membrane Potential
        self.s = None         # Spike Output
        self.x = None         # Trace (Filtered Spikes)
        self.j = None         # Input Current
        self.e_gen = None     # Generative Error
        self.e_disc = None    # Discriminative Error
        self.ref_count = None # Refractory Counter

    def init_state(self, batch_size, device):
        """全状態変数のリセット (画像切り替え時に呼ばれる)"""
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        dt = self.cfg['dt']
        
        # --- 不応期管理 ---
        is_refractory = self.ref_count > 0
        
        # --- LIF Dynamics ---
        # 1. Current Update (j)
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        new_j = self.j + (dt / self.cfg['tau_j']) * d_j
        self.j = new_j 

        # 2. Voltage Update (v)
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        new_v = self.v + (dt / self.cfg['tau_m']) * d_v
        self.v = torch.where(is_refractory, self.v, new_v)
        
        # 3. Spike Generation
        spikes = (self.v > self.cfg['thresh']).float()
        
        # 4. Refractory Timer Update
        self.ref_count = torch.clamp(self.ref_count - dt, min=0)
        self.ref_count = torch.where(spikes.bool(), torch.tensor(self.cfg['t_ref'], device=self.v.device), self.ref_count)
        
        # 5. Reset & Output
        self.v = self.v * (1 - spikes) 
        self.s = spikes
        
        # 6. Trace Update (for Learning)
        if not self.is_data_layer:
            self.x = self.x + (dt / self.cfg['tau_tr']) * (-self.x) + spikes

class label_layer(nn.Module):
    """
    出力層（ラベル層）。
    Training時は教師信号(Target)をクランプし、Testing時は推論モード(LIF)で動作する。
    """
    def __init__(self, output_dim, config):
        super().__init__()
        self.dim = output_dim
        self.cfg = config
        self.is_training = True

        self.v = None; self.s = None; self.x = None; self.j = None
        self.e_disc = None # label_layerには e_gen は無い（最上層のため）
        self.ref_count = None
        
        # Inference用バッファ
        self.v_tmp = None; self.s_tmp = None; self.j_tmp = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)
        self.ref_count = torch.zeros(batch_size, self.dim, device=device)

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
        
        # --- Test Mode Logic (Free Running) ---
        if not self.is_training:
            d_j_tmp = (-self.cfg['kappa_j'] * self.j_tmp + input2tmp)
            self.j_tmp = self.j_tmp + (dt / self.cfg['tau_j']) * d_j_tmp
            
            d_v_tmp = (-self.cfg['gamma_m'] * self.v_tmp + self.cfg['R_m'] * self.j_tmp)
            self.v_tmp = self.v_tmp + (dt / self.cfg['tau_m']) * d_v_tmp
            
            spikes_tmp = (self.v_tmp > self.cfg['thresh']).float()
            self.s_tmp = spikes_tmp
            self.v_tmp = self.v_tmp * (1 - spikes_tmp)
        
        # --- Main Logic ---
        is_refractory = self.ref_count > 0
        
        d_j = (-self.cfg['kappa_j'] * self.j + input2main)
        new_j = self.j + (dt / self.cfg['tau_j']) * d_j
        self.j = torch.where(is_refractory, self.j, new_j)
        
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        new_v = self.v + (dt / self.cfg['tau_m']) * d_v
        self.v = torch.where(is_refractory, self.v, new_v)
        
        spikes = (self.v > self.cfg['thresh']).float()
        
        self.ref_count = torch.clamp(self.ref_count - dt, min=0)
        self.ref_count = torch.where(spikes.bool(), torch.tensor(self.cfg['t_ref'], device=self.v.device), self.ref_count)
        
        self.s = spikes
        self.v = self.v * (1 - spikes)


# ==========================================
#  Network Definition
# ==========================================

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
            
        # --- Weights Initialization ---
        self.W = nn.ParameterList() 
        self.V = nn.ParameterList() 
                    
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))

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

        # 1. Update Neural Activities
        if y_target is not None: # Training
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

        else: # Testing (Inference)
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
                    input2main = 0; input2tmp = 0
                    input2main += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    input2main += torch.matmul(e_gen_lower, self.W[i-1])
                    s_lower = self.layers[i-1].s
                    input2tmp += torch.matmul(s_lower, self.V[i-1].t())
                    layer.update_state(input2main, input2tmp)

        # 2. Calculate Prediction Errors
        for i, layer in enumerate(self.layers):
            if i == 0:
                s_own = self.layers[i].s
                layer.e_gen = alpha_gen * (x_data - s_own)
            elif i < len(self.layers) - 1:
                # Discriminative Error
                if i == 1:
                    z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                else:
                    s_lower = self.layers[i-1].s
                    z_disc_pred = torch.matmul(s_lower, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                # Generative Error
                if i < len(self.layers) - 2:
                    s_upper = self.layers[i+1].s
                    z_gen_pred = torch.matmul(s_upper, self.W[i].t())
                    layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                else:
                    if y_target is not None:
                        z_gen_label = torch.matmul(y_target, self.W[i].t())
                    else:
                         z_gen_label = 0 
                    layer.e_gen = alpha_gen * (layer.x - z_gen_label)
            else:
                s_own = self.layers[i].s
                if y_target is not None:
                    layer.e_disc = alpha_disc * (y_target - s_own)
                else:
                    s_tmp = layer.s_tmp
                    layer.e_disc = alpha_disc * (s_own - s_tmp)

    def manual_weight_update(self, x_data, y_target=None):
        alpha_u = self.config['alpha_u']
        
        for i in range(len(self.layers)):
            # Update V
            if i < len(self.layers) - 1:
                e_disc_upper = self.layers[i+1].e_disc
                if i == 0:
                    grad_V = torch.matmul(e_disc_upper.t(), x_data)
                else:
                    s_own = self.layers[i].s
                    grad_V = torch.matmul(e_disc_upper.t(), s_own)
                self.V[i] += alpha_u * grad_V

            # Update W
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

# ==========================================
#  Main Experiment Loop
# ==========================================

def run_experiment(dataset_name='MNIST'):
    print(f"\n=== Running bPC-SNN | Fast Mode (T_st={CONFIG['T_st']}ms) ===")
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_file_path = f"{CONFIG['save_dir']}/log_{dataset_name}_{timestamp}.csv"
    print(f"Saving to: {save_file_path}")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    # Dataset
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    # Model
    h_size = CONFIG['hidden_size']
    layer_sizes = [784, h_size, h_size, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=CONFIG).to(CONFIG['device'])
    
    max_steps = int(CONFIG['T_st'] / CONFIG['dt'])
    logs = []

    with torch.no_grad():
        for epoch in range(CONFIG['epochs']):
            model.train()
            epoch_start = time.time()
            total_spikes_epoch = 0
            
            # --- Training Loop ---
            for batch_idx, (imgs, lbls) in enumerate(train_l):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
                targets.scatter_(1, lbls.view(-1, 1), 1)
                
                # Input Gain
                imgs_rate = torch.clamp(imgs, 0, 1) * CONFIG['input_gain']
                spike_in = spikegen.rate(imgs_rate, max_steps)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                # Early Stopping Variables
                prev_error_energy = float('inf')
                settle_count = 0
                active_steps = 0
                
                for t in range(max_steps):
                    x_t = spike_in[t]
                    
                    model.forward_dynamics(x_data=x_t, y_target=targets)
                    model.manual_weight_update(x_data=x_t, y_target=targets)
                    model.clip_weights(20.0)
                    
                    total_spikes_epoch += model.layers[-1].s.sum().item()
                    active_steps += 1
                    
                    # --- Early Stopping Logic (Fix Applied) ---
                    if t >= CONFIG['min_steps']:
                        current_energy = 0
                        for layer in model.layers:
                            # 1. e_disc (All layers have this)
                            if layer.e_disc is not None:
                                current_energy += layer.e_disc.abs().mean().item()
                            
                            # 2. e_gen (Check if layer has it)
                            if hasattr(layer, 'e_gen') and layer.e_gen is not None:
                                current_energy += layer.e_gen.abs().mean().item()
                        
                        delta = abs(prev_error_energy - current_energy)
                        ratio = delta / (prev_error_energy + 1e-9)
                        
                        if ratio < CONFIG['settle_threshold']:
                            settle_count += 1
                        else:
                            settle_count = 0
                        
                        if settle_count >= CONFIG['patience']:
                            break
                        
                        prev_error_energy = current_energy
                
                if batch_idx % 100 == 0:
                    print(f"Ep {epoch} | Bt {batch_idx} | Steps: {active_steps}/{max_steps} | Spikes(Out): {total_spikes_epoch}")
                    total_spikes_epoch = 0
        
            # --- Testing Loop ---
            print(">> Inference Mode...")
            model.layers[-1].switch2testing_mode()
            model.eval()
            test_correct = 0
            test_samples = 0
            
            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                imgs_rate = torch.clamp(imgs, 0, 1) * CONFIG['input_gain']
                spike_in = spikegen.rate(imgs_rate, max_steps)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                sum_out_spikes = 0
                
                for t in range(max_steps):
                    x_t = spike_in[t]
                    model.forward_dynamics(x_data=x_t, y_target=None)
                    sum_out_spikes += model.layers[-1].s
                
                _, pred = torch.max(sum_out_spikes, 1)
                # print(sum_out_spikes)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
                
            test_acc = 100 * test_correct / test_samples
            epoch_time = time.time() - epoch_start
            
            print(f"Ep {epoch} END | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            model.layers[-1].switch2training_mode()

            logs.append({
                'epoch': epoch,
                'test_acc': test_acc,
                'time': epoch_time
            })
            
            df = pd.DataFrame(logs)
            df.to_csv(save_file_path, index=False)

if __name__ == "__main__":
    run_experiment('MNIST')
