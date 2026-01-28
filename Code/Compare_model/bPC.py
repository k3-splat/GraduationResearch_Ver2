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

# =============================================================================
# [Phase 0] 共通設定
# =============================================================================
COMMON_CONFIG = {
    'batch_size': 256,
    'hidden_size': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # -----------------------------------------------------------
    # [制御フラグ] 
    # True : Optunaによる探索を実行 (Table 5の設定を使用)
    # False: 探索をスキップし、下記 'fixed_params' で学習・生成のみ実行
    # -----------------------------------------------------------
    'run_search': True,  # ★ここを False にすれば探索なしモードになります

    # 探索設定 (Table 5準拠)
    'search_trials': 20,    # 探索回数 (必要に応じて増やしてください)
    'search_epochs': 25,    # 論文 Table 5 では Epoch 25
    
    # 本番学習設定
    'final_epochs': 25,    
    
    # 固定パラメータ (探索スキップ時に使用するデフォルト設定)
    'fixed_params': {
        'activation': 'leaky_relu',
        'lr_activities': 0.05,
        'momentum': 0.9,
        'lr_weights': 1e-4,
        'weight_decay': 1e-3,
        'alpha_gen': 0.01,  
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
        
        # 論文で使用される活性化関数
        if act_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
            self.act_deriv = lambda x: (x > 0).float() + 0.1 * (x <= 0).float()
        elif act_name == 'tanh':
            self.activation = nn.Tanh()
            self.act_deriv = lambda x: 1.0 - torch.tanh(x)**2
        elif act_name == 'gelu':
            self.activation = nn.GELU()
            # GELUの微分近似
            self.act_deriv = lambda x: 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) + \
                                       0.5 * x * (1.0 - torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))**2) * \
                                       math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * torch.pow(x, 2.0))
        else:
            raise ValueError(f"Unknown activation: {act_name}")

        self.V = nn.Linear(in_features, out_features, bias=False)
        self.W = nn.Linear(out_features, in_features, bias=False)
        
    def initialize_feedforward(self, u_lower):
        """ボトムアップ初期化"""
        return self.V(self.activation(u_lower))

    def initialize_feedback(self, u_upper):
        """トップダウン初期化"""
        return self.W(self.activation(u_upper))

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
        """画像(x_in)からラベル層へ向かうボトムアップ初期化"""
        activities = [x_in]
        curr = x_in
        for layer in self.layers:
            curr = layer.initialize_feedforward(curr)
            activities.append(curr)
        return activities

    def backward_init(self, x_top):
        """ラベル(x_top)から画像層へ向かうトップダウン初期化"""
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

        # 教師あり学習の場合、最上層(ラベル層)をターゲットに固定
        if y_target is not None and is_training:
            batch_size = x_in.size(0)
            y_onehot = torch.zeros(batch_size, 10, device=x_in.device)
            y_onehot.scatter_(1, y_target.view(-1, 1), 1.0)
            x[-1] = y_onehot

        for _ in range(steps):
            eps_gen = [] 
            eps_disc = []
            
            # 1. 誤差計算
            for l in range(len(self.layers) + 1):
                # Bottom-up (Discriminative) Error
                if l == 0:
                    e_d = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l-1]
                    pred = layer.V(layer.activation(x[l-1]))
                    e_d = alpha_disc * (x[l] - pred)
                eps_disc.append(e_d)

                # Top-down (Generative) Error
                if l == len(self.layers):
                    e_g = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l]
                    pred = layer.W(layer.activation(x[l+1]))
                    e_g = alpha_gen * (x[l] - pred)
                eps_gen.append(e_g)

            # 2. 活動更新
            start_idx = 1
            end_idx = len(self.layers)
            if is_training and y_target is not None:
                end_idx = len(self.layers) - 1

            new_x = [t.clone() for t in x]
            for l in range(start_idx, end_idx + 1):
                grad_E = eps_gen[l] + eps_disc[l]
                
                # 下層からのトップダウン誤差の影響
                layer_below = self.layers[l-1]
                prop_gen = torch.matmul(eps_gen[l-1], layer_below.W.weight)
                
                # 上層からのボトムアップ誤差の影響
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

# =============================================================================
# [Data] Train(60k) / Val(5k) / Test(5k) Split
# =============================================================================
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    # Train set (60,000)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Original Test set (10,000) -> Split into Val (5,000) & Test (5,000)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_size = 5000
    test_size = 5000
    val_dataset, test_dataset = random_split(full_test, [val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# =============================================================================
# [Helper] クラスごとの平均画像を計算 (正規化済み空間)
# =============================================================================
def compute_class_averages(loader, device):
    class_sums = torch.zeros(10, 784).to(device)
    class_counts = torch.zeros(10).to(device)
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        for i in range(10):
            mask = (labels == i)
            if mask.sum() > 0:
                class_sums[i] += images[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    
    # 0除算防止
    class_counts[class_counts == 0] = 1.0
    class_avgs = class_sums / class_counts.unsqueeze(1)
    return class_avgs

# =============================================================================
# [Validation Utils]
# =============================================================================
def run_validation_acc(model, loader, device, params):
    """分類精度の計算"""
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

def run_validation_rmse(model, class_averages, device, params):
    """生成画像のRMSE計算 (トップダウン初期化を使用)"""
    labels = torch.arange(10).to(device)
    batch_size = 10
    y_onehot = torch.zeros(batch_size, 10).to(device)
    y_onehot.scatter_(1, labels.view(-1, 1), 1.0)
    
    # トップダウン初期化 (ラベル -> 画像)
    x = model.backward_init(y_onehot)
    x[-1] = y_onehot # ラベル固定
    
    m_buffer = [torch.zeros_like(a) for a in x]
    
    # 生成推論
    with torch.no_grad():
        for step in range(params['T_eval']):
            eps_gen = []
            eps_disc = []
            
            for l in range(len(model.layers) + 1):
                if l == 0:
                    e_d = torch.zeros_like(x[l])
                else:
                    layer = model.layers[l-1]
                    pred = layer.V(layer.activation(x[l-1]))
                    e_d = params['alpha_disc'] * (x[l] - pred)
                eps_disc.append(e_d)
                
                if l == len(model.layers):
                    e_g = torch.zeros_like(x[l])
                else:
                    layer = model.layers[l]
                    pred = layer.W(layer.activation(x[l+1]))
                    e_g = params['alpha_gen'] * (x[l] - pred)
                eps_gen.append(e_g)
                
            for l in range(len(model.layers)): 
                grad_E = eps_gen[l] + eps_disc[l]
                
                if l > 0:
                    prop_gen = torch.matmul(eps_gen[l-1], model.layers[l-1].W.weight)
                else:
                    prop_gen = 0
                
                prop_disc = torch.matmul(eps_disc[l+1], model.layers[l].V.weight)
                f_prime = model.layers[l].act_deriv(x[l])
                
                delta = -grad_E + f_prime * (prop_gen + prop_disc)
                m_buffer[l] = params['momentum'] * m_buffer[l] + delta
                x[l] = x[l] + params['lr_activities'] * m_buffer[l]

    x_gen = x[0] # 生成画像 (正規化空間)
    
    # RMSE計算: 平均画像(class_averages) と 生成画像(x_gen) の誤差
    mse = torch.mean((x_gen - class_averages) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

# =============================================================================
# [Phase 1] ハイパーパラメータ探索 (Optuna - Table 5)
# =============================================================================
def search_hyperparameters(train_loader, val_loader):
    print("\n" + "="*60)
    print("PHASE 1: Hyperparameter Search (Optuna - Table 5)")
    print("Objective: Minimize [(1 - Accuracy) + RMSE]")
    print("="*60)
    
    device = torch.device(COMMON_CONFIG['device'])
    
    # バリデーションセットを使ってクラスごとの平均画像を事前計算
    print("Computing validation class averages for RMSE optimization...")
    val_class_averages = compute_class_averages(val_loader, device)

    def objective(trial):
        # -----------------------------------------------------
        # Table 5 に基づく探索空間
        # -----------------------------------------------------
        params = {
            'activation': trial.suggest_categorical('activation', ['leaky_relu', 'tanh', 'gelu']),
            'lr_activities': trial.suggest_float('lr_activities', 1e-4, 5e-2, log=True),
            'momentum': trial.suggest_categorical('momentum', [0.0, 0.5, 0.9]),
            'lr_weights': trial.suggest_float('lr_weights', 1e-6, 1e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            # alpha_gen: 10^0 to 10^-4
            'alpha_gen': trial.suggest_categorical('alpha_gen', [1.0, 0.1, 0.01, 0.001, 0.0001]),
            'alpha_disc': 1.0,
            'T_train': 8,
            'T_eval': 100
        }

        model = bPC_Net(COMMON_CONFIG['hidden_size'], params['activation']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr_weights'], weight_decay=params['weight_decay'])
        
        n_epochs = COMMON_CONFIG['search_epochs']
        
        for epoch in range(n_epochs):
            model.train()
            # 探索時間の短縮のため、ここでは最初の50バッチのみ使用しています。
            # フルデータセットで探索したい場合は "if i > 50: break" を削除してください。
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
            
            # --- Evaluation Step ---
            val_acc = run_validation_acc(model, val_loader, device, params)
            val_rmse = run_validation_rmse(model, val_class_averages, device, params)
            
            # Combined Objective
            # Accuracyは大きいほど良いので (1 - acc/100) で最小化問題にする
            # RMSEは小さいほど良い
            combined_loss = (1.0 - val_acc / 100.0) + val_rmse
            
            trial.report(combined_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return combined_loss

    # ログ詳細度の設定
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=COMMON_CONFIG['search_trials'])
    
    # ★探索結果のCSV保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_csv_path = f"logs/optuna_search_results_{timestamp}.csv"
    study.trials_dataframe().to_csv(search_csv_path, index=False)
    print(f"\nAll hyperparameter search results saved to: {search_csv_path}")
    
    print(f"Best Loss: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    best_params = study.best_params
    best_params.update({'T_train': 8, 'T_eval': 100, 'alpha_disc': 1.0})
    return best_params

# =============================================================================
# [Phase 2] 学習
# =============================================================================
def calculate_pass_costs(model, batch_size):
    dm_bytes = 0; fpo = 0
    for layer in model.layers:
        n_in = layer.in_features; n_out = layer.out_features
        dm_bytes += (batch_size * n_in + n_out * n_in + batch_size * n_out) * 4
        fpo += (batch_size * n_in * n_out) * 2
    return dm_bytes, fpo

def run_fixed_epoch_training(best_params, train_loader, test_loader, csv_path):
    epochs = COMMON_CONFIG['final_epochs']
    print("\n" + "="*60)
    print(f"PHASE 2: Final Training (Fixed Epochs: {epochs})")
    print(f"Using Params: {best_params}")
    print("="*60)
    
    device = torch.device(COMMON_CONFIG['device'])
    model = bPC_Net(COMMON_CONFIG['hidden_size'], best_params['activation']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr_weights'], weight_decay=best_params['weight_decay'])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Accuracy', 'Total_G_FPO', 'Total_GB_Data', 'Time_s'])

    total_fpo = 0.0
    total_dm = 0.0
    start_time = time.time()
    acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            
            activities = model.forward_init(images)
            dm, fpo = calculate_pass_costs(model, bs)
            total_dm += dm; total_fpo += fpo
            
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
            
        # Test Splitを使用
        acc = run_validation_acc(model, test_loader, device, best_params)
        elapsed = time.time() - start_time
        
        print(f"{epoch:<6} | {acc:<10.2f} | {total_fpo/1e9:<10.2f} | {total_dm/1e9:<10.2f}")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc, total_fpo/1e9, total_dm/1e9, elapsed])

    print("\n=== Final Report ===")
    print(f"Final Accuracy: {acc:.2f}%")
    return model

# =============================================================================
# [Phase 3] 画像生成実験 + RMSE計算
# =============================================================================
def generate_images_from_labels(model, device, params, class_averages, save_path="generated_digits.png"):
    print("\n" + "="*60)
    print("PHASE 3: Generating Images & RMSE Calculation")
    print("="*60)
    
    model.eval()
    rmse = run_validation_rmse(model, class_averages, device, params)
    print(f"Total RMSE (Generated vs Class Average): {rmse:.4f}")

    # 画像保存用に再生成
    labels = torch.arange(10).to(device)
    y_onehot = torch.zeros(10, 10).to(device)
    y_onehot.scatter_(1, labels.view(-1, 1), 1.0)
    
    x = model.backward_init(y_onehot)
    x[-1] = y_onehot
    m_buffer = [torch.zeros_like(a) for a in x]

    for step in range(params['T_eval']):
        eps_gen = []
        eps_disc = []
        for l in range(len(model.layers) + 1):
            if l == 0: e_d = torch.zeros_like(x[l])
            else:
                layer = model.layers[l-1]
                pred = layer.V(layer.activation(x[l-1]))
                e_d = params['alpha_disc'] * (x[l] - pred)
            eps_disc.append(e_d)
            if l == len(model.layers): e_g = torch.zeros_like(x[l])
            else:
                layer = model.layers[l]
                pred = layer.W(layer.activation(x[l+1]))
                e_g = params['alpha_gen'] * (x[l] - pred)
            eps_gen.append(e_g)
            
        for l in range(len(model.layers)): 
            grad_E = eps_gen[l] + eps_disc[l]
            if l > 0: prop_gen = torch.matmul(eps_gen[l-1], model.layers[l-1].W.weight)
            else: prop_gen = 0
            prop_disc = torch.matmul(eps_disc[l+1], model.layers[l].V.weight)
            f_prime = model.layers[l].act_deriv(x[l])
            delta = -grad_E + f_prime * (prop_gen + prop_disc)
            m_buffer[l] = params['momentum'] * m_buffer[l] + delta
            x[l] = x[l] + params['lr_activities'] * m_buffer[l]

    x_img = x[0].detach().cpu().numpy().reshape(10, 28, 28)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(x_img[i], cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated images saved to: {save_path}")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists('logs'): os.makedirs('logs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/bPC_log_{timestamp}.txt"
    csv_file = f"logs/bPC_metrics_{timestamp}.csv"
    gen_img_file = f"logs/generated_digits_{timestamp}.png"
    
    sys.stdout = DualLogger(log_file)
    print(f"Starting bPC Experiment.")
    
    train_loader, val_loader, test_loader = get_dataloaders(COMMON_CONFIG['batch_size'])
    device = torch.device(COMMON_CONFIG['device'])

    if COMMON_CONFIG['run_search']:
        # Phase 1: 探索実行 (Table 5の範囲)
        best_params = search_hyperparameters(train_loader, val_loader)
    else:
        # 探索なし: 固定パラメータで通常の学習へ
        print("\nSkipping Hyperparameter Search... Using Fixed Params.")
        best_params = COMMON_CONFIG['fixed_params']
    
    # Phase 2: 学習 (探索したパラメータ or 固定パラメータを使用)
    trained_model = run_fixed_epoch_training(best_params, train_loader, test_loader, csv_file)
    
    # Phase 3: 生成とRMSE評価 (Test Loaderで平均画像を計算して評価)
    print("Computing Test Set class averages for final evaluation...")
    test_class_avgs = compute_class_averages(test_loader, device)
    
    generate_images_from_labels(
        trained_model, device, best_params,
        class_averages=test_class_avgs,
        save_path=gen_img_file
    )