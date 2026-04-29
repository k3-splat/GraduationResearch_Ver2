import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import math
import numpy as np
import optuna
import sys
import os
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

# =============================================================================
# [Logger] 標準出力をファイルにも同時に書き込むクラス
# =============================================================================
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

@dataclass
class CostStats:
    flops_dense: float = 0.0
    flops_sparse: float = 0.0
    dm_dense: float = 0.0
    dm_sparse: float = 0.0
    bits_dense: float = 0.0
    bits_sparse: float = 0.0

# =============================================================================
# [Phase 0] 共通設定
# =============================================================================
COMMON_CONFIG = {
    'batch_size': 256,
    'hidden_size': 400,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'run_search': False,  
    'search_trials': 20, 
    'search_epochs': 25, 
    'final_epochs': 20,    
    'fixed_params': {
        'activation': 'leaky_relu', 
        'lr_activities': 0.0034549692255545815,
        'momentum': 0.0,
        'lr_weights': 5.7652236557134764e-05,
        'weight_decay': 0.0011400984979283125,
        'alpha_gen': 0.001,   
        'alpha_disc': 1.0,    
        'T_train': 8,         
        'T_eval': 100         
    }
}

# =============================================================================
# [共通] モデル定義 (bPC)
# =============================================================================
class bPC_Layer(nn.Module):
    def __init__(self, in_features, out_features, act_name):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # --- 活性化関数の分岐 ---
        if act_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
            self.act_deriv = lambda x: (x > 0).float() + 0.1 * (x <= 0).float()
        elif act_name == 'relu':
            self.activation = nn.ReLU()
            self.act_deriv = lambda x: (x > 0).float()
        elif act_name == 'tanh':
            self.activation = nn.Tanh()
            self.act_deriv = lambda x: 1.0 - torch.tanh(x)**2
        elif act_name == 'gelu':
            self.activation = nn.GELU()
            self.act_deriv = lambda x: 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) + \
                                     0.5 * x * (1.0 - torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))**2) * \
                                     math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * torch.pow(x, 2.0))
        else:
            raise ValueError(f"Unknown activation: {act_name}")

        self.V = nn.Linear(in_features, out_features, bias=False)
        self.W = nn.Linear(out_features, in_features, bias=False)
        
    def initialize_feedforward(self, u_lower):
        return self.V(self.activation(u_lower))

    def initialize_feedback(self, u_upper):
        return self.W(self.activation(u_upper))

class bPC_Net(nn.Module):
    def __init__(self, hidden_size, act_name):
        super().__init__()
        # 隠れ層1つの3層構造 (Input -> Hidden -> Output)
        self.layers = nn.ModuleList([
            bPC_Layer(784, hidden_size, act_name),
            bPC_Layer(hidden_size, 10, act_name)
        ])
        
    def get_act_deriv(self, layer_idx, x):
        return self.layers[layer_idx].act_deriv(x)

    def forward_init(self, x_in):
        activities = [x_in]
        curr = x_in
        for layer in self.layers:
            curr = layer.initialize_feedforward(curr)
            activities.append(curr)
        return activities

    def backward_init(self, x_top):
        activities = [None] * (len(self.layers) + 1)
        activities[-1] = x_top 
        curr = x_top
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            curr = layer.initialize_feedback(curr)
            activities[i] = curr
        return activities

    def _calc_ops_cost(self, input_tensor, dim_out, stats):
        """tanomu.py standard calculation."""
        batch_size, dim_in = input_tensor.shape
        num_active = (input_tensor.abs() > 1e-6).sum().item()
        
        # FLOPs: Dense=2*N, Sparse=2*nnz
        stats.flops_dense += batch_size * dim_in * dim_out * 2.0
        stats.flops_sparse += num_active * dim_out * 2.0

        # Data Movement (Bytes):
        # Dense: Read W(all) + Read X(all) + Write Y(all)
        stats.dm_dense += (dim_in * dim_out * 4.0) + ((batch_size * dim_in + batch_size * dim_out) * 4.0)
        
        # Sparse: Read W(rows) + Read X(val+idx=8B) + Write Y(dense)
        stats.dm_sparse += (num_active * dim_out * 4.0) + ((num_active * 2.0 + batch_size * dim_out) * 4.0)

        # Communication (Bits):
        # Dense: 32bit * batch * dim_in
        stats.bits_dense += batch_size * dim_in * 32
        # Sparse: 64bit * nnz
        stats.bits_sparse += num_active * 64

    def run_inference(self, activities, x_in, y_target, steps, lr_x, momentum, alpha_gen, alpha_disc, is_training=True, stats=None):
        x = [a.clone().detach() for a in activities]
        m_buffer = [torch.zeros_like(a) for a in x]

        if y_target is not None and is_training:
            batch_size = x_in.size(0)
            y_onehot = torch.zeros(batch_size, 10, device=x_in.device)
            y_onehot.scatter_(1, y_target.view(-1, 1), 1.0)
            x[-1] = y_onehot

        last_eps_disc = []
        last_eps_gen = []

        for _ in range(steps):
            eps_gen = [] 
            eps_disc = []
            
            # --- Measurement Integration ---
            if stats is not None:
                # Measure Forward Pass (Generation) & Backward Pass (Discrimination)
                # Note: bPC updates are simultaneous, but conceptually involves matrix multiplies
                for l in range(len(self.layers)):
                    layer = self.layers[l]
                    # Forward Prediction: V(act(x[l]))
                    # Input to matrix mult is activation(x[l])
                    act_in = layer.activation(x[l])
                    self._calc_ops_cost(act_in, layer.out_features, stats)

                    # Backward Prediction: W(act(x[l+1]))
                    act_feedback = layer.activation(x[l+1])
                    self._calc_ops_cost(act_feedback, layer.in_features, stats)
            # -------------------------------

            for l in range(len(self.layers) + 1):
                # Discriminative Error (e_d)
                if l == 0:
                    e_d = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l-1]
                    pred = layer.V(layer.activation(x[l-1]))
                    e_d = alpha_disc * (x[l] - pred)
                eps_disc.append(e_d)

                # Generative Error (e_g)
                if l == len(self.layers):
                    e_g = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l]
                    pred = layer.W(layer.activation(x[l+1]))
                    e_g = alpha_gen * (x[l] - pred)
                eps_gen.append(e_g)

            # 更新処理
            start_idx = 1
            end_idx = len(self.layers)
            if is_training and y_target is not None:
                end_idx = len(self.layers) - 1

            new_x = [t.clone() for t in x]
            for l in range(start_idx, end_idx + 1):
                grad_E = eps_gen[l] + eps_disc[l]
                
                # Error Propagation (Matrix Mults for Gradient)
                # These are also matrix operations
                layer_below = self.layers[l-1]
                # prop_gen = eps_gen[l-1] @ W
                if stats is not None: self._calc_ops_cost(eps_gen[l-1], layer_below.in_features, stats)
                prop_gen = torch.matmul(eps_gen[l-1], layer_below.W.weight)
                
                if l < len(self.layers):
                    layer_above = self.layers[l]
                    # prop_disc = eps_disc[l+1] @ V
                    if stats is not None: self._calc_ops_cost(eps_disc[l+1], layer_above.out_features, stats)
                    prop_disc = torch.matmul(eps_disc[l+1], layer_above.V.weight)
                else:
                    prop_disc = 0

                f_prime = self.get_act_deriv(l-1, x[l])
                delta = -grad_E + f_prime * (prop_gen + prop_disc)
                m_buffer[l] = momentum * m_buffer[l] + delta
                new_x[l] = x[l] + lr_x * m_buffer[l]
            x = new_x
            last_eps_disc = eps_disc
            last_eps_gen = eps_gen

        return x, last_eps_disc, last_eps_gen


# =============================================================================
# [Data] Train/Val/Test Split
# =============================================================================
def get_dataloaders(batch_size, is_search=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    if is_search:
        val_dataset, test_dataset = random_split(full_test, [5000, 5000])
        print("Data Split: Train=60000, Val=5000, Test=5000 (Search Mode)")
    else:
        val_dataset = full_test
        test_dataset = full_test
        print("Data Split: Train=60000, Test=10000 (Full Mode)")
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

def compute_class_averages(loader, device):
    class_sums = torch.zeros(10, 784).to(device)
    class_counts = torch.zeros(10).to(device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(10):
            mask = (labels == i)
            if mask.sum() > 0:
                class_sums[i] += images[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    class_counts[class_counts == 0] = 1.0
    return class_sums / class_counts.unsqueeze(1)

def run_validation_acc(model, loader, device, params):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        activities = model.forward_init(images)
        with torch.no_grad():
            final_activities, _, _ = model.run_inference(
                activities, images, None, steps=params['T_eval'], 
                lr_x=params['lr_activities'], momentum=params['momentum'], 
                alpha_gen=params['alpha_gen'], alpha_disc=params['alpha_disc'], is_training=False, stats=None
            )
        _, predicted = torch.max(final_activities[-1], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_validation_rmse(model, class_averages, device, params):
    labels = torch.arange(10).to(device)
    y_onehot = torch.zeros(10, 10).to(device).scatter_(1, labels.view(-1, 1), 1.0)
    x = model.backward_init(y_onehot)
    x[-1] = y_onehot
    m_buffer = [torch.zeros_like(a) for a in x]
    
    with torch.no_grad():
        for step in range(params['T_eval']):
            eps_gen, eps_disc = [], []
            for l in range(len(model.layers) + 1):
                if l == 0: e_d = torch.zeros_like(x[l])
                else:
                    layer = model.layers[l-1]
                    e_d = params['alpha_disc'] * (x[l] - layer.V(layer.activation(x[l-1])))
                eps_disc.append(e_d)
                if l == len(model.layers): e_g = torch.zeros_like(x[l])
                else:
                    layer = model.layers[l]
                    e_g = params['alpha_gen'] * (x[l] - layer.W(layer.activation(x[l+1])))
                eps_gen.append(e_g)
                
            for l in range(len(model.layers)): 
                grad_E = eps_gen[l] + eps_disc[l]
                prop_gen = torch.matmul(eps_gen[l-1], model.layers[l-1].W.weight) if l > 0 else 0
                prop_disc = torch.matmul(eps_disc[l+1], model.layers[l].V.weight)
                delta = -grad_E + model.layers[l].act_deriv(x[l]) * (prop_gen + prop_disc)
                m_buffer[l] = params['momentum'] * m_buffer[l] + delta
                x[l] = x[l] + params['lr_activities'] * m_buffer[l]
                
    return torch.sqrt(torch.mean((x[0] - class_averages) ** 2)).item()

def compute_model_cost(model):
    cost = 0.0
    for param in model.parameters():
        cost += 0.5 * torch.norm(param, 2) ** 2
    return cost.item()

# =============================================================================
# [Phase 2] 学習 & 誤差計測
# =============================================================================
def plot_error_trends(error_history, save_path):
    epochs = range(1, len(error_history['disc_L0']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for l in range(3):
        label = f'Layer {l} (Input)' if l==0 else (f'Layer {l} (Hidden)' if l==1 else f'Layer {l} (Output)')
        plt.plot(epochs, error_history[f'disc_L{l}'], label=label, marker='o', markersize=3)
    plt.title('Discriminative Error (e_disc) per Sample')
    plt.xlabel('Epoch'); plt.ylabel('Mean Error'); 
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    for l in range(3):
        label = f'Layer {l} (Input)' if l==0 else (f'Layer {l} (Hidden)' if l==1 else f'Layer {l} (Output)')
        plt.plot(epochs, error_history[f'gen_L{l}'], label=label, marker='o', markersize=3)
    plt.title('Generative Error (e_gen) per Sample')
    plt.xlabel('Epoch'); plt.ylabel('Mean Error'); 
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_fixed_epoch_training(best_params, train_loader, test_loader, csv_path, error_fig_path):
    epochs = COMMON_CONFIG['final_epochs']
    print("\n" + "="*60 + f"\nPHASE 2: Final Training with Advanced Metrics (Epochs: {epochs})\n" + "="*60)
    device = torch.device(COMMON_CONFIG['device'])
    model = bPC_Net(COMMON_CONFIG['hidden_size'], best_params['activation']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr_weights'], weight_decay=best_params['weight_decay'])
    
    num_layers_total = len(model.layers) + 1 
    error_history = {f'{t}_L{l}': [] for t in ['disc', 'gen'] for l in range(num_layers_total)}
    
    cum_stats = CostStats()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Accuracy', 'MDL_Total', 
            'FLOPs_D(G)', 'FLOPs_S(G)',      
            'DM_D(GB)', 'DM_S(GB)',          
            'Bits_D(Gb)', 'Bits_S(Gb)',      
            'Time_s'
        ])

    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_errs = {k: 0.0 for k in error_history.keys()}
        epoch_data_cost = 0.0 
        sample_count = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            sample_count += bs
            
            activities = model.forward_init(images)
            
            # Pass stats object to accumulate costs strictly step-by-step
            final_activities, e_discs, e_gens = model.run_inference(
                activities, images, labels, steps=best_params['T_train'], 
                lr_x=best_params['lr_activities'], momentum=best_params['momentum'], 
                alpha_gen=best_params['alpha_gen'], alpha_disc=best_params['alpha_disc'],
                is_training=True, stats=cum_stats
            )
            
            for l in range(num_layers_total):
                epoch_errs[f'disc_L{l}'] += torch.norm(e_discs[l], p=2, dim=1).sum().item()
                epoch_errs[f'gen_L{l}'] += torch.norm(e_gens[l], p=2, dim=1).sum().item()

            optimizer.zero_grad()
            loss_total = 0
            for l in range(len(model.layers)):
                layer = model.layers[l]
                loss_total += 0.5 * best_params['alpha_disc'] * torch.sum((final_activities[l+1].detach() - layer.V(layer.activation(final_activities[l].detach())))**2)
                loss_total += 0.5 * best_params['alpha_gen'] * torch.sum((final_activities[l].detach() - layer.W(layer.activation(final_activities[l+1].detach())))**2)
            
            loss_total.backward()
            optimizer.step()
            epoch_data_cost += loss_total.item()

        for k in error_history.keys():
            error_history[k].append(epoch_errs[k] / sample_count)

        mdl_model = compute_model_cost(model)
        mdl_total = epoch_data_cost + mdl_model

        acc = run_validation_acc(model, test_loader, device, best_params)
        
        print(f"Epoch {epoch:02d} | Acc: {acc:.2f}% | MDL: {mdl_total:.0f} | "
              f"Bits(Sparse): {cum_stats.bits_sparse/1e9:.2f}Gb")
        
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, acc, mdl_total, 
                cum_stats.flops_dense/1e9, cum_stats.flops_sparse/1e9,
                cum_stats.dm_dense/1e9,    cum_stats.dm_sparse/1e9,
                cum_stats.bits_dense/1e9,  cum_stats.bits_sparse/1e9, 
                time.time()-start_time
            ])

    plot_error_trends(error_history, error_fig_path)
    return model

# =============================================================================
# [Phase 3] 画像生成
# =============================================================================
def generate_images_from_labels(model, device, params, class_averages, save_path):
    model.eval()
    labels = torch.arange(10).to(device)
    y_onehot = torch.zeros(10, 10).to(device).scatter_(1, labels.view(-1, 1), 1.0)
    x = model.backward_init(y_onehot)
    x[-1] = y_onehot
    m_buffer = [torch.zeros_like(a) for a in x]

    for _ in range(params['T_eval']):
        eps_gen, eps_disc = [], []
        for l in range(len(model.layers) + 1):
            if l == 0: e_d = torch.zeros_like(x[l])
            else:
                layer = model.layers[l-1]
                e_d = params['alpha_disc'] * (x[l] - layer.V(layer.activation(x[l-1])))
            eps_disc.append(e_d)
            if l == len(model.layers): e_g = torch.zeros_like(x[l])
            else:
                layer = model.layers[l]
                e_g = params['alpha_gen'] * (x[l] - layer.W(layer.activation(x[l+1])))
            eps_gen.append(e_g)
        for l in range(len(model.layers)):
            delta = -(eps_gen[l] + eps_disc[l]) + model.layers[l].act_deriv(x[l]) * ( (torch.matmul(eps_gen[l-1], model.layers[l-1].W.weight) if l>0 else 0) + torch.matmul(eps_disc[l+1], model.layers[l].V.weight) )
            m_buffer[l] = params['momentum'] * m_buffer[l] + delta
            x[l] = x[l] + params['lr_activities'] * m_buffer[l]

    x_img = x[0].detach().cpu().numpy().reshape(10, 28, 28)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(x_img[i], cmap='gray', vmin=x_img.min(), vmax=x_img.max())
        axes[i].axis('off')
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists('logs'): os.makedirs('logs')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/bPC_log_{ts}.txt"
    csv_file = f"logs/bPC_metrics_{ts}.csv"
    err_fig = f"logs/error_trends_{ts}.png"
    gen_img = f"logs/generated_digits_{ts}.png"
    
    sys.stdout = DualLogger(log_file)
    # ここで run_search フラグを渡すように変更
    train_loader, val_loader, test_loader = get_dataloaders(COMMON_CONFIG['batch_size'], is_search=COMMON_CONFIG['run_search'])
    device = torch.device(COMMON_CONFIG['device'])

    best_params = COMMON_CONFIG['fixed_params']
    
    if COMMON_CONFIG['run_search']:
        pass 
    
    trained_model = run_fixed_epoch_training(best_params, train_loader, test_loader, csv_file, err_fig)
    
    print("Computing Final RMSE and generating images...")
    test_class_avgs = compute_class_averages(test_loader, device)
    final_rmse = run_validation_rmse(trained_model, test_class_avgs, device, best_params)
    print(f"Final Generative RMSE: {final_rmse:.4f}")
    
    generate_images_from_labels(trained_model, device, best_params, test_class_avgs, gen_img)
    print(f"Results saved: {err_fig}, {gen_img}")