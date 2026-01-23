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
    'T_st' : 200.0,
    'tau_j' : 10.0,
    'tau_m' : 20.0,
    'tau_tr' : 20.0,
    'tau_data' : 100.0,
    'kappa_j': 0.25,
    'gamma_m': 1.0,
    'R_m' : 1.0,
    'alpha_u' : 0.0005,
    'alpha_gen' : 1.0,
    'alpha_disc' : 1.0,
    'thresh': 0.4,
    'batch_size': 64,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './results/bPC_SNN',
    'max_freq': 1000.0
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

class bPC_SNNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, data=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = data
        
        self.v = None
        self.s = None
        self.x = None 
        self.j = None 
        self.e_gen = None
        self.e_disc = None

    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device) 
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def switch_Testing_mode(self):
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
            
            # 【修正】蓄積型トレース (Accumulating Trace)
            # スパイク(1.0)をそのまま足し込み、時定数で減衰させる
            # これにより、適度な発火率でトレース値が 0.5~1.0 程度まで育つ
            decay = dt / self.cfg['tau_tr']
            self.x = self.x * (1 - decay) + spikes


class bPC_SNN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes

        for i, size in enumerate(layer_sizes):
            is_data = (i == 0) or (i == len(layer_sizes) - 1)
            self.layers.append(bPC_SNNLayer(idx=i, output_dim=size, config=config, data=is_data))
            
        self.W = nn.ParameterList()
        self.V = nn.ParameterList()
                    
        for i in range(len(layer_sizes) - 1):
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            
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

    def clip_weights(self, max_norm=5.0): # 【緩和】1.0だと厳しすぎる可能性があるので緩和
        for w_list in [self.W, self.V]:
            for w in w_list:
                w.data.clamp_(-max_norm, max_norm)

    def forward_dynamics(self, x_data, y_target=None):
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        # === Update Phase ===
        if y_target is not None:
            # Training
            for i, layer in enumerate(self.layers):
                total_input = 0
                if i > 0 and i < len(self.layers) - 1:
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                    total_input += (- layer.e_gen)
                    e_disc_upper = self.layers[i+1].e_disc
                    total_input += torch.matmul(e_disc_upper, self.V[i])
                layer.update_state(total_input)

            # Error Calculation
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_gen_pred = torch.matmul(self.layers[i+1].x, self.W[i].t())
                    layer.e_gen = alpha_gen * (x_data - z_gen_pred)

                elif i < len(self.layers) - 1:
                    if i == 1:
                        z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                    else:
                        x_lower = self.layers[i-1].x
                        z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                        layer.e_disc = alpha_disc * (layer.x - z_disc_pred)

                    if i < len(self.layers) - 2:
                        x_upper = self.layers[i+1].x
                        z_gen_pred = torch.matmul(x_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)
                    else:
                        # Target is clamped
                        z_gen_label = torch.matmul(y_target, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_label)
                else:
                    # Last Layer Error
                    x_lower = self.layers[i-1].x
                    z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (y_target - z_disc_pred)

        else:
            # Testing
            for i, layer in enumerate(self.layers):
                total_input = 0
                if i > 0:
                    total_input += (- layer.e_disc)
                    e_gen_lower = self.layers[i-1].e_gen
                    total_input += torch.matmul(e_gen_lower, self.W[i-1])

                    if i < len(self.layer_sizes) - 1:
                        total_input += (- layer.e_gen)
                        e_disc_upper = self.layers[i+1].e_disc
                        total_input += torch.matmul(e_disc_upper, self.V[i])
                    
                    layer.update_state(total_input)

            # Error Calculation (Test)
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_gen_pred = torch.matmul(self.layers[i+1].x, self.W[i].t())
                    layer.e_gen = alpha_gen * (x_data - z_gen_pred)
                elif i == 1:
                    z_disc_data = torch.matmul(x_data, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (layer.x - z_disc_data)
                else:
                    x_lower = self.layers[i-1].x
                    z_disc_pred = torch.matmul(x_lower, self.V[i-1].t())
                    layer.e_disc = alpha_disc * (layer.x - z_disc_pred)
                    
                    if i < len(self.layer_sizes) - 1:
                        x_upper = self.layers[i+1].x
                        z_gen_pred = torch.matmul(x_upper, self.W[i].t())
                        layer.e_gen = alpha_gen * (layer.x - z_gen_pred)


    def manual_weight_update(self, x_data, y_target=None):
        alpha_u = self.config['alpha_u']
        for i in range(len(self.layers)):
            # V update
            if i < len(self.layers) - 1:
                e_disc_upper = self.layers[i+1].e_disc
                if i == 0:
                    grad_V = torch.matmul(e_disc_upper.t(), x_data)
                else:
                    x_own = self.layers[i].x
                    grad_V = torch.matmul(e_disc_upper.t(), x_own)
                self.V[i] += alpha_u * grad_V

            # W update
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
    print(f"\n=== Running bPC-SNN on {dataset_name} (Final Fix) ===")
    
    try:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        script_name = "notebook_execution"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_file_path = f"{CONFIG['save_dir']}/log_{script_name}_{dataset_name}_{timestamp}.csv"
    print(f"Results will be saved to: {save_file_path}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    if dataset_name == 'MNIST':
        train_d = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_d = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    train_l = DataLoader(train_d, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_l = DataLoader(test_d, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)
    
    layer_sizes = [784, 500, 500, 10]
    model = bPC_SNN(layer_sizes=layer_sizes, config=CONFIG).to(CONFIG['device'])
    
    steps = int(CONFIG['T_st'] / CONFIG['dt'])
    logs = []

    with torch.no_grad():
        for epoch in range(CONFIG['epochs']):
            model.train()
            epoch_start = time.time()
            
            for batch_idx, (imgs, lbls) in enumerate(train_l):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                
                targets = torch.zeros(imgs.size(0), 10).to(CONFIG['device'])
                targets.scatter_(1, lbls.view(-1, 1), 1)
                
                imgs_rate = torch.clamp(imgs, 0, 1)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                for _ in range(steps):
                    model.forward_dynamics(x_data=imgs_rate, y_target=targets)
                    model.manual_weight_update(x_data=imgs_rate, y_target=targets)
                    model.clip_weights(5.0)
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx}")
                    # デバッグ: 各層のトレース最大値を確認
                    h1_max = model.layers[1].x.max().item()
                    h2_max = model.layers[2].x.max().item()
                    err = model.layers[-1].e_disc.abs().mean().item()
                    print(f"  Trace Max -> H1: {h1_max:.3f}, H2: {h2_max:.3f}")
                    print(f"  Mean Error: {err:.4f}")
        
            print("Switching label layer to Inference Mode (LIF)...")
            model.layers[-1].switch_Testing_mode()

            model.eval()
            test_correct = 0
            test_samples = 0
            
            # テスト時の発火状況確認用フラグ
            debug_test_printed = False

            for imgs, lbls in test_l:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                imgs_rate = torch.clamp(imgs, 0, 1)
                
                model.reset_state(imgs.size(0), CONFIG['device'])
                
                for t in range(steps):
                    model.forward_dynamics(x_data=imgs_rate, y_target=None)
                
                # 最終層のトレースを取得
                out_trace = model.layers[-1].x
                
                # デバッグ: テスト時の出力層の活動状況を1回だけ表示
                if not debug_test_printed:
                    print(f"  [TEST DEBUG] Output Trace Max: {out_trace.max().item():.3f}")
                    print(f"  [TEST DEBUG] Output Trace Mean: {out_trace.mean().item():.3f}")
                    debug_test_printed = True
                
                _, pred = torch.max(out_trace, 1)
                test_correct += (pred == lbls).sum().item()
                test_samples += lbls.size(0)
                
            test_acc = 100 * test_correct / test_samples
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch} DONE | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            print("Switching label layer back to Training Mode (Clamp)...")
            model.layers[-1].switch_Testing_mode()

            logs.append({
                'dataset': dataset_name,
                'epoch': epoch,
                'test_acc': test_acc,
                'time': epoch_time
            })
            
            df = pd.DataFrame(logs)
            df.to_csv(save_file_path, index=False)

if __name__ == "__main__":
    run_experiment('MNIST')