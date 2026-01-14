import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import math
import numpy as np
import optuna
import copy
import sys
import os
import csv
from datetime import datetime

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
        self.log.flush() # 即時書き込み

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# =============================================================================
# [Phase 0] 共通設定
# =============================================================================
COMMON_CONFIG = {
    'batch_size': 256,
    'hidden_size': 500,
    'val_interval': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 探索用設定
    'search_trials': 20,    
    'search_epochs': 5,     
    
    # 測定用設定
    'measure_target_acc': 95.0, 
    'measure_max_epochs': 50,   
}

# =============================================================================
# [共通] モデル定義 (bPC Corrected)
# =============================================================================
class bPC_Layer(nn.Module):
    def __init__(self, in_features, out_features, act_name):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if act_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
            self.act_deriv = lambda x: (x > 0).float() + 0.1 * (x <= 0).float()
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

class bPC_Net(nn.Module):
    def __init__(self, hidden_size, act_name):
        super().__init__()
        self.layers = nn.ModuleList([
            bPC_Layer(784, hidden_size, act_name),
            bPC_Layer(hidden_size, hidden_size, act_name),
            bPC_Layer(hidden_size, hidden_size, act_name),
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

    def run_inference(self, activities, x_in, y_target, steps, lr_x, momentum, alpha_gen, alpha_disc, is_training=True):
        x = [a.clone().detach() for a in activities]
        m_buffer = [torch.zeros_like(a) for a in x]

        if y_target is not None and is_training:
            batch_size = x_in.size(0)
            y_onehot = torch.zeros(batch_size, 10, device=x_in.device)
            y_onehot.scatter_(1, y_target.view(-1, 1), 1.0)
            x[-1] = y_onehot

        for _ in range(steps):
            eps_gen = [] 
            eps_disc = []
            
            for l in range(len(self.layers) + 1):
                if l == 0:
                    e_d = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l-1]
                    pred = layer.V(layer.activation(x[l-1]))
                    e_d = alpha_disc * (x[l] - pred)
                eps_disc.append(e_d)

                if l == len(self.layers):
                    e_g = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l]
                    pred = layer.W(layer.activation(x[l+1]))
                    e_g = alpha_gen * (x[l] - pred)
                eps_gen.append(e_g)

            start_idx = 1
            end_idx = len(self.layers)
            if is_training and y_target is not None:
                end_idx = len(self.layers) - 1

            new_x = [t.clone() for t in x]
            for l in range(start_idx, end_idx + 1):
                grad_E = eps_gen[l] + eps_disc[l]
                layer_below = self.layers[l-1]
                prop_gen = torch.matmul(eps_gen[l-1], layer_below.W.weight)
                
                if l < len(self.layers):
                    layer_above = self.layers[l]
                    prop_disc = torch.matmul(eps_disc[l+1], layer_above.V.weight)
                else:
                    prop_disc = 0

                f_prime = self.get_act_deriv(l-1, x[l])
                delta = -grad_E + f_prime * (prop_gen + prop_disc)
                
                m_buffer[l] = momentum * m_buffer[l] + delta
                new_x[l] = x[l] + lr_x * m_buffer[l]
            x = new_x
        return x

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

def run_validation(model, loader, device, params):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        activities = model.forward_init(images)
        with torch.no_grad():
            final_activities = model.run_inference(
                activities, images, None, 
                steps=params['T_eval'], lr_x=params['lr_activities'], 
                momentum=params['momentum'], 
                alpha_gen=params['alpha_gen'], alpha_disc=params['alpha_disc'], 
                is_training=False
            )
        output = final_activities[-1]
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

# =============================================================================
# [Phase 1] ハイパーパラメータ探索 (Optuna)
# =============================================================================
def search_hyperparameters():
    print("\n" + "="*60)
    print("PHASE 1: Hyperparameter Search (Optuna)")
    print("="*60)
    
    train_loader, val_loader, _ = get_dataloaders(COMMON_CONFIG['batch_size'])
    device = torch.device(COMMON_CONFIG['device'])

    def objective(trial):
        params = {
            'activation': trial.suggest_categorical('activation', ['leaky_relu', 'tanh']),
            'lr_activities': trial.suggest_float('lr_activities', 1e-3, 0.1, log=True),
            'momentum': trial.suggest_categorical('momentum', [0.5, 0.9]),
            'lr_weights': trial.suggest_float('lr_weights', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'alpha_gen': trial.suggest_categorical('alpha_gen', [1e-1, 1e-4]),
            'alpha_disc': 1.0,
            'T_train': 8,
            'T_eval': 20
        }

        model = bPC_Net(COMMON_CONFIG['hidden_size'], params['activation']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr_weights'], weight_decay=params['weight_decay'])
        
        for epoch in range(COMMON_CONFIG['search_epochs']):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                if i > 50: break 
                images, labels = images.to(device), labels.to(device)
                
                activities = model.forward_init(images)
                with torch.no_grad():
                    final_activities = model.run_inference(
                        activities, images, labels, 
                        steps=params['T_train'], lr_x=params['lr_activities'], 
                        momentum=params['momentum'], 
                        alpha_gen=params['alpha_gen'], alpha_disc=params['alpha_disc'], 
                        is_training=True
                    )
                
                optimizer.zero_grad()
                loss_total = 0
                x = final_activities
                for l in range(len(model.layers)):
                    layer = model.layers[l]
                    if l < len(model.layers):
                        pred_up = layer.V(layer.activation(x[l].detach()))
                        loss_total += 0.5 * params['alpha_disc'] * torch.sum((x[l+1].detach() - pred_up)**2)
                    pred_down = layer.W(layer.activation(x[l+1].detach()))
                    loss_total += 0.5 * params['alpha_gen'] * torch.sum((x[l].detach() - pred_down)**2)
                
                loss_total.backward()
                optimizer.step()
            
            val_acc = run_validation(model, val_loader, device, params)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return val_acc

    # Optunaのログは冗長なので少し抑制
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=COMMON_CONFIG['search_trials'])
    
    print(f"Best Search Accuracy: {study.best_value:.2f}%")
    print(f"Best Params: {study.best_params}")
    
    best_params = study.best_params
    best_params.update({'T_train': 8, 'T_eval': 100, 'alpha_disc': 1.0})
    return best_params

# =============================================================================
# [Phase 2] コスト計測 (Metrics Calculation)
# =============================================================================
def compute_twonn_id(data):
    N = data.shape[0]
    if N < 3: return 0.0
    data_t = torch.tensor(data).float()
    if torch.cuda.is_available(): data_t = data_t.cuda()
    
    sum_sq = torch.sum(data_t**2, dim=1, keepdim=True)
    dist_sq = sum_sq + sum_sq.t() - 2 * torch.mm(data_t, data_t.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq)
    sorted_dist, _ = torch.sort(dist, dim=1)
    
    r1 = sorted_dist[:, 1]; r2 = sorted_dist[:, 2]
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]
    if len(mu) == 0: return 0.0
    return len(mu) / torch.sum(torch.log(mu)).item()

def compute_weight_entropy(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.detach().cpu().numpy().flatten())
    all_weights = np.concatenate(all_weights)
    hist, _ = np.histogram(all_weights, bins=100, density=True)
    hist = hist[hist > 0]
    hist_norm = hist / hist.sum()
    return -np.sum(hist_norm * np.log2(hist_norm))

def calculate_pass_costs(model, batch_size):
    dm_bytes = 0; fpo = 0
    for layer in model.layers:
        n_in = layer.in_features; n_out = layer.out_features
        dm_bytes += (batch_size * n_in + n_out * n_in + batch_size * n_out) * 4
        fpo += (batch_size * n_in * n_out) * 2
    return dm_bytes, fpo

def measure_cost(best_params, csv_path):
    print("\n" + "="*60)
    print(f"PHASE 2: Cost Measurement (Target: {COMMON_CONFIG['measure_target_acc']}%)")
    print("="*60)
    
    device = torch.device(COMMON_CONFIG['device'])
    train_loader, _, test_loader = get_dataloaders(COMMON_CONFIG['batch_size'])
    model = bPC_Net(COMMON_CONFIG['hidden_size'], best_params['activation']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr_weights'], weight_decay=best_params['weight_decay'])
    
    # CSVヘッダー書き込み
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Accuracy', 'Total_G_FPO', 'Total_GB_Data', 'Weight_Entropy', 'ID_L1', 'Time_s'])

    total_fpo = 0.0
    total_dm = 0.0
    total_steps = 0
    start_time = time.time()
    reached = False
    
    print(f"{'Epoch':<6} | {'Test Acc':<10} | {'G-FPO':<10} | {'GB-Data':<10} | {'W-Ent':<8} | {'ID-L1':<8}")
    print("-" * 75)
    
    for epoch in range(1, COMMON_CONFIG['measure_max_epochs'] + 1):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            
            # 1. Init
            activities = model.forward_init(images)
            dm, fpo = calculate_pass_costs(model, bs)
            total_dm += dm; total_fpo += fpo
            
            # 2. Inference
            with torch.no_grad():
                final_activities = model.run_inference(
                    activities, images, labels, 
                    steps=best_params['T_train'], lr_x=best_params['lr_activities'], 
                    momentum=best_params['momentum'], 
                    alpha_gen=best_params['alpha_gen'], alpha_disc=best_params['alpha_disc'], 
                    is_training=True
                )
            dm, fpo = calculate_pass_costs(model, bs)
            total_dm += dm * 4 * best_params['T_train']
            total_fpo += fpo * 4 * best_params['T_train']
            
            # 3. Update
            optimizer.zero_grad()
            loss_total = 0
            x = final_activities
            for l in range(len(model.layers)):
                layer = model.layers[l]
                if l < len(model.layers):
                    pred_up = layer.V(layer.activation(x[l].detach()))
                    loss_total += 0.5 * best_params['alpha_disc'] * torch.sum((x[l+1].detach() - pred_up)**2)
                pred_down = layer.W(layer.activation(x[l+1].detach()))
                loss_total += 0.5 * best_params['alpha_gen'] * torch.sum((x[l].detach() - pred_down)**2)
            
            loss_total.backward()
            optimizer.step()
            
            dm, fpo = calculate_pass_costs(model, bs)
            total_dm += dm * 2; total_fpo += fpo * 2
            total_steps += 1
            
            if batch_idx % COMMON_CONFIG['val_interval'] == 0:
                acc = run_validation(model, test_loader, device, best_params)
                w_ent = compute_weight_entropy(model)
                id_l1 = compute_twonn_id(x[1].detach().cpu().numpy())
                elapsed = time.time() - start_time
                
                # 画面表示
                print(f"{epoch:<6} | {acc:<10.2f} | {total_fpo/1e9:<10.2f} | {total_dm/1e9:<10.2f} | {w_ent:<8.2f} | {id_l1:<8.2f}")
                
                # CSV保存
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, acc, total_fpo/1e9, total_dm/1e9, w_ent, id_l1, elapsed])
                
                if acc >= COMMON_CONFIG['measure_target_acc']:
                    reached = True
                    break
        if reached: break

    print("\n=== Final Report ===")
    if reached:
        print(f"Goal Reached in {time.time()-start_time:.2f} seconds.")
        print(f"Total Steps: {total_steps}")
        print(f"Total FPO: {total_fpo/1e9:.4f} G")
        print(f"Total Data Movement: {total_dm/1e9:.4f} GB")
        print(f"Data saved to: {csv_path}")
    else:
        print("Target accuracy not reached within max epochs.")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # ログ設定
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/bPC_log_{timestamp}.txt"
    csv_file = f"logs/bPC_metrics_{timestamp}.csv"
    
    # 標準出力をフックしてログファイルにも書き込む
    sys.stdout = DualLogger(log_file)
    
    print(f"Starting bPC Experiment. Logs will be saved to {log_file} and {csv_file}")
    
    # 1. 探索
    best_params = search_hyperparameters()
    
    # 2. 測定 (CSVパスを渡す)
    measure_cost(best_params, csv_file)