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
import matplotlib.pyplot as plt
import collections

# =============================================================================
# [Logger] Writes standard output to a file simultaneously
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

# =============================================================================
# [Phase 0] Configuration
# =============================================================================
COMMON_CONFIG = {
    'batch_size': 256,
    'hidden_size': 400,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'run_search': False,  # Set to True to run Optuna search
    'search_trials': 20, 
    'search_epochs': 25, 
    'final_epochs': 20,    
    'fixed_params': {
        'activation': 'leaky_relu',
        'lr_activities': 0.0034549692255545815,
        'momentum': 0.0,
        'lr_weights': 5.7652236557134764e-05,
        'weight_decay': 0.0011400984979283125,
        'alpha_gen': 0.0001,  
        'alpha_disc': 1.0,    
        'T_train': 8,         
        'T_eval': 100         
    }
}

# =============================================================================
# [Model] bPC Network Definition
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

    def initialize_feedback(self, u_upper):
        return self.W(self.activation(u_upper))

class bPC_Net(nn.Module):
    def __init__(self, hidden_size, act_name):
        super().__init__()
        self.layers = nn.ModuleList([
            bPC_Layer(784, hidden_size, act_name),
            # bPC_Layer(hidden_size, hidden_size, act_name),
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

    def run_inference(self, activities, x_in, y_target, steps, lr_x, momentum, alpha_gen, alpha_disc, is_training=True):
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

            # Update Activities
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
            last_eps_disc = eps_disc
            last_eps_gen = eps_gen

        return x, last_eps_disc, last_eps_gen

# =============================================================================
# [Data] Data Loaders
# =============================================================================
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_dataset, test_dataset = random_split(full_test, [5000, 5000])
    
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
                alpha_gen=params['alpha_gen'], alpha_disc=params['alpha_disc'], is_training=False
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

# =============================================================================
# [Phase 1] Hyperparameter Search
# =============================================================================
def search_hyperparameters(train_loader, val_loader):
    print("\n" + "="*60 + "\nPHASE 1: Hyperparameter Search\n" + "="*60)
    device = torch.device(COMMON_CONFIG['device'])
    val_class_averages = compute_class_averages(val_loader, device)

    def objective(trial):
        params = {
            'activation': trial.suggest_categorical('activation', ['leaky_relu', 'tanh', 'gelu']),
            'lr_activities': trial.suggest_float('lr_activities', 1e-4, 5e-2, log=True),
            'momentum': trial.suggest_categorical('momentum', [0.0, 0.5, 0.9]),
            'lr_weights': trial.suggest_float('lr_weights', 1e-6, 1e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'alpha_gen': trial.suggest_categorical('alpha_gen', [1.0, 0.1, 0.01, 0.001, 0.0001]),
            'alpha_disc': 1.0, 'T_train': 8, 'T_eval': 100
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
                    x, _, _ = model.run_inference(activities, images, labels, **{k: params[k] for k in ['steps', 'lr_x', 'momentum', 'alpha_gen', 'alpha_disc'] if k in params or k.replace('steps','T_train') in params}, steps=params['T_train'], lr_x=params['lr_activities'])
                
                optimizer.zero_grad()
                loss = 0
                for l in range(len(model.layers)):
                    layer = model.layers[l]
                    loss += 0.5 * params['alpha_disc'] * torch.sum((x[l+1].detach() - layer.V(layer.activation(x[l].detach())))**2)
                    loss += 0.5 * params['alpha_gen'] * torch.sum((x[l].detach() - layer.W(layer.activation(x[l+1].detach())))**2)
                loss.backward(); optimizer.step()
            val_acc = run_validation_acc(model, val_loader, device, params)
            val_rmse = run_validation_rmse(model, val_class_averages, device, params)
            combined_loss = 2 * (1.0 - val_acc / 100.0) + val_rmse
            trial.report(combined_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return combined_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=COMMON_CONFIG['search_trials'])
    best_params = study.best_params
    best_params.update({'T_train': 8, 'T_eval': 100, 'alpha_disc': 1.0})
    return best_params

# =============================================================================
# [Phase 2] Training & Analysis Utilities
# =============================================================================
def calculate_pass_costs(model, batch_size):
    dm, fpo = 0, 0
    for layer in model.layers:
        n_in, n_out = layer.in_features, layer.out_features
        dm += (batch_size * n_in + n_out * n_in + batch_size * n_out) * 4
        fpo += (batch_size * n_in * n_out) * 2
    return dm, fpo

def plot_error_trends(error_history, save_path):
    epochs = range(1, len(error_history['disc_L0']) + 1)
    plt.figure(figsize=(12, 5))
    
    # Discriminative Error
    plt.subplot(1, 2, 1)
    for l in range(3):
        plt.plot(epochs, error_history[f'disc_L{l}'], label=f'Layer {l}', marker='o', markersize=3)
    plt.title('Discriminative Error (e_disc)')
    plt.xlabel('Epoch'); plt.ylabel('Mean Error')
    plt.grid(True); plt.legend()

    # Generative Error
    plt.subplot(1, 2, 2)
    for l in range(3):
        plt.plot(epochs, error_history[f'gen_L{l}'], label=f'Layer {l}', marker='o', markersize=3)
    plt.title('Generative Error (e_gen)')
    plt.xlabel('Epoch'); plt.ylabel('Mean Error')
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_activity_histograms(xt_samples, epoch, save_dir):
    """Generates histograms for activities in each layer."""
    plt.figure(figsize=(18, 5))
    
    # L0 (Input reconstruction?), L1 (Hidden 1), L2 (Hidden 2), L3 (Output)
    # Note: xt_samples is a list of tensors [x_0, x_1, x_2, x_3]
    # We aggregate all batches first, so xt_samples[layer_idx] is (Total_Samples, Features)
    
    num_layers = len(xt_samples)
    for i in range(num_layers):
        plt.subplot(1, num_layers, i+1)
        data = xt_samples[i].flatten().numpy()
        plt.hist(data, bins=100, alpha=0.7, color='purple', density=True)
        plt.yscale('log')
        plt.title(f"Layer {i} Activity (Epoch {epoch})")
        plt.xlabel("Value")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"dist_epoch_{epoch:03d}.png"))
    plt.close()

def run_fixed_epoch_training(best_params, train_loader, test_loader, csv_path, error_fig_path, run_dir):
    epochs = COMMON_CONFIG['final_epochs']
    print("\n" + "="*60 + f"\nPHASE 2: Final Training (Epochs: {epochs})\n" + "="*60)
    device = torch.device(COMMON_CONFIG['device'])
    model = bPC_Net(COMMON_CONFIG['hidden_size'], best_params['activation']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr_weights'], weight_decay=best_params['weight_decay'])
    
    error_history = {f'{t}_L{l}': [] for t in ['disc', 'gen'] for l in range(3)}
    
    # Analysis History
    analysis_history = collections.defaultdict(list)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Accuracy', 'Sparsity_L1', 'MeanAct_L1', 'Total_G_FPO', 'Total_GB_Data', 'Time_s'])

    total_fpo, total_dm = 0.0, 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_errs = {k: 0.0 for k in error_history.keys()}
        sample_count = 0
        
        # Container for sampling activities (to avoid OOM, we sample periodically)
        xt_samples_buffer = [[] for _ in range(4)] # For layers 0, 1, 2, 3

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            sample_count += bs
            
            activities = model.forward_init(images)
            final_activities, e_discs, e_gens = model.run_inference(
                activities, images, labels, steps=best_params['T_train'], 
                lr_x=best_params['lr_activities'], momentum=best_params['momentum'], 
                alpha_gen=best_params['alpha_gen'], alpha_disc=best_params['alpha_disc']
            )
            
            # Collect Error
            for l in range(3):
                epoch_errs[f'disc_L{l}'] += torch.norm(e_discs[l+1], p=2, dim=1).sum().item()
                epoch_errs[f'gen_L{l}'] += torch.norm(e_gens[l], p=2, dim=1).sum().item()
            
            # Collect Activities for Analysis (Sample every 20 batches)
            if i % 20 == 0:
                for l_idx, act in enumerate(final_activities):
                    xt_samples_buffer[l_idx].append(act.detach().cpu())

            # Update Weights
            optimizer.zero_grad()
            loss_total = 0
            for l in range(len(model.layers)):
                layer = model.layers[l]
                loss_total += 0.5 * best_params['alpha_disc'] * torch.sum((final_activities[l+1].detach() - layer.V(layer.activation(final_activities[l].detach())))**2)
                loss_total += 0.5 * best_params['alpha_gen'] * torch.sum((final_activities[l].detach() - layer.W(layer.activation(final_activities[l+1].detach())))**2)
            loss_total.backward(); optimizer.step()
            
            # Update Costs
            dm, fpo = calculate_pass_costs(model, bs)
            total_dm += (dm * (1 + 4 * best_params['T_train'] + 2))
            total_fpo += (fpo * (1 + 4 * best_params['T_train'] + 2))

        # Record Mean Errors
        for k in error_history.keys():
            error_history[k].append(epoch_errs[k] / sample_count)

        # --- ANALYSIS: Compute Sparsity and Histograms ---
        # Concatenate buffered samples
        full_xt_samples = [torch.cat(buf, dim=0) for buf in xt_samples_buffer]
        
        # Calculate stats for Layer 1 (First Hidden Layer)
        l1_act = full_xt_samples[1]
        sparsity = (l1_act.abs() < 0.01).float().mean().item()
        mean_act = l1_act.abs().mean().item()
        
        # Save Histograms
        plot_activity_histograms(full_xt_samples, epoch, run_dir)
        
        # Validate
        acc = run_validation_acc(model, test_loader, device, best_params)
        
        # Update History
        analysis_history['epoch'].append(epoch)
        analysis_history['accuracy'].append(acc)
        analysis_history['sparsity_l1'].append(sparsity)
        analysis_history['mean_act_l1'].append(mean_act)

        print(f"Epoch {epoch:02d} | Acc: {acc:.2f}% | Sparse(L1): {sparsity:.3f} | MeanAct(L1): {mean_act:.3f}")
        
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, acc, sparsity, mean_act, total_fpo/1e9, total_dm/1e9, time.time()-start_time])

    # Plot Final Error Trends
    plot_error_trends(error_history, error_fig_path)
    
    # Plot Summary Metrics (Acc, Sparsity, Activity)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(analysis_history['epoch'], analysis_history['accuracy'], 'o-')
    plt.title('Test Accuracy'); plt.xlabel('Epoch'); plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(analysis_history['epoch'], analysis_history['sparsity_l1'], 'o-', color='green')
    plt.title('Sparsity L1 (|x|<0.01)'); plt.xlabel('Epoch'); plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(analysis_history['epoch'], analysis_history['mean_act_l1'], 'o-', color='orange')
    plt.title('Mean Activity L1'); plt.xlabel('Epoch'); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_summary_metrics.png"))
    plt.close()

    return model

# =============================================================================
# [Phase 3] Image Generation
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
# Main Execution
# =============================================================================
if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/bPC_{ts}"
    if not os.path.exists(run_dir): os.makedirs(run_dir)
    
    log_file = os.path.join(run_dir, "bPC_log.txt")
    csv_file = os.path.join(run_dir, "bPC_metrics.csv")
    err_fig = os.path.join(run_dir, "error_trends.png")
    gen_img = os.path.join(run_dir, "generated_digits.png")
    
    sys.stdout = DualLogger(log_file)
    train_loader, val_loader, test_loader = get_dataloaders(COMMON_CONFIG['batch_size'])
    device = torch.device(COMMON_CONFIG['device'])

    best_params = search_hyperparameters(train_loader, val_loader) if COMMON_CONFIG['run_search'] else COMMON_CONFIG['fixed_params']
    
    # Run Training with Analysis
    trained_model = run_fixed_epoch_training(best_params, train_loader, test_loader, csv_file, err_fig, run_dir)
    
    print("Computing Final RMSE and generating images...")
    test_class_avgs = compute_class_averages(test_loader, device)
    generate_images_from_labels(trained_model, device, best_params, test_class_avgs, gen_img)
    print(f"Results saved in: {run_dir}")