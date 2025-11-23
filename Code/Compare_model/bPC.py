import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

# ---------------------------------------------------------
# 1. 設定とハイパーパラメータ (論文 Table 8 準拠)
# ---------------------------------------------------------
CONFIG = {
    'batch_size': 128,      # ユーザー指定 (論文は256)
    'lr_weights': 1e-4,     # 論文 Table 8: lr_theta = 1e-4
    'lr_activities': 0.01,  # 論文 Table 8: lr_x = 0.01
    'weight_decay': 5e-3,   # 論文 Table 8: weight_decay = 5e-3
    'target_accuracy': 97.0,
    'max_epochs': 50,
    'hidden_size': 500,     # ユーザー指定 (論文は256)
    'val_interval': 100,
    'alpha_gen': 1e-4,      # 論文 Table 8: alpha_gen = 1e-4
    'alpha_disc': 1.0,      # 論文 Table 8: alpha_disc = 1
    'T_train': 8,           # 論文 Table 8: T = 8
    'T_test': 100,          # 論文 Table 8: T eval = 100
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ---------------------------------------------------------
# 2. データセットの準備
# ---------------------------------------------------------
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------
# 3. モデル定義: bPC (Structure: 784 -> 500 -> 500 -> 500 -> 10)
# ---------------------------------------------------------
class bPC_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 論文 Table 8: Activation = Leaky ReLU
        self.activation = nn.LeakyReLU(0.1)
        
        # Bottom-up weights (Discriminative): V
        self.V = nn.Linear(in_features, out_features, bias=False)
        # Top-down weights (Generative): W
        self.W = nn.Linear(out_features, in_features, bias=False)
        
    def forward_bottom_up(self, u_lower):
        return self.activation(self.V(u_lower))

class bPC_Net(nn.Module):
    def __init__(self, hidden_size=500):
        super().__init__()
        self.layers = nn.ModuleList([
            bPC_Layer(784, hidden_size),
            bPC_Layer(hidden_size, hidden_size),
            bPC_Layer(hidden_size, hidden_size),
            bPC_Layer(hidden_size, 10)
        ])
        # Leaky ReLU derivative for manual update
        self.act_deriv = lambda x: (x > 0).float() + 0.1 * (x <= 0).float()

    def forward_init(self, x_in):
        """ボトムアップ・スイープによる初期化"""
        activities = [x_in]
        curr = x_in
        for layer in self.layers:
            curr = layer.forward_bottom_up(curr)
            activities.append(curr)
        return activities

    def run_inference(self, activities, x_in, y_target, steps, lr_x, is_training=True):
        """
        反復推論 (Neural Dynamics)
        Eq 5: dx/dt = -err_gen - err_disc + f'(x) * (errors_from_neighbors)
        """
        x = [a.clone().detach() for a in activities]
        
        # Training時は最上位層をラベルに固定
        if y_target is not None and is_training:
            batch_size = x_in.size(0)
            y_onehot = torch.zeros(batch_size, 10, device=x_in.device)
            y_onehot.scatter_(1, y_target.view(-1, 1), 1.0)
            x[-1] = y_onehot

        for _ in range(steps):
            eps_gen = [] 
            eps_disc = []
            
            # --- 1. Error Computation ---
            for l in range(len(self.layers) + 1):
                # Discriminative Error: x_l - f(V * x_{l-1})
                if l == 0:
                    e_d = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l-1]
                    pred = layer.activation(layer.V(x[l-1]))
                    e_d = CONFIG['alpha_disc'] * (x[l] - pred)
                eps_disc.append(e_d)

                # Generative Error: x_l - W * f(x_{l+1})
                # Note: 論文 Eq 4 & 5 に従い、top-down予測は W * f(x_{l+1})
                if l == len(self.layers):
                    e_g = torch.zeros_like(x[l])
                else:
                    layer = self.layers[l]
                    pred = layer.W(layer.activation(x[l+1]))
                    e_g = CONFIG['alpha_gen'] * (x[l] - pred)
                eps_gen.append(e_g)

            # --- 2. Activity Update ---
            start_idx = 1
            end_idx = len(self.layers)
            if is_training and y_target is not None:
                end_idx = len(self.layers) - 1 # Label層は更新しない

            new_x = [t.clone() for t in x]
            
            for l in range(start_idx, end_idx + 1):
                # Gradient term from local errors
                grad = -eps_gen[l] - eps_disc[l]
                
                # Propagated errors
                # From below: eps_gen[l-1] * W^T
                layer_below = self.layers[l-1]
                prop_gen = torch.matmul(eps_gen[l-1], layer_below.W.weight)
                
                # From above: eps_disc[l+1] * V^T
                if l < len(self.layers):
                    layer_above = self.layers[l]
                    prop_disc = torch.matmul(eps_disc[l+1], layer_above.V.weight)
                else:
                    prop_disc = 0

                f_prime = self.act_deriv(x[l])
                delta = grad + f_prime * (prop_gen + prop_disc)
                new_x[l] = x[l] + lr_x * delta
            
            x = new_x
            
        return x

# ---------------------------------------------------------
# 4. コスト計算 (MACs & FLOPs)
# ---------------------------------------------------------
def calculate_bpc_costs(model, batch_size, inference_steps, is_training=True):
    """
    bPCの計算コスト:
    1. 初期化 (1 pass)
    2. 反復推論 (T steps): 各ステップで V, W, V^T, W^T の4回の行列演算
    3. 重み更新 (Training): V, W の勾配計算 (2回)
    """
    layer_sizes = [784, 500, 500, 500, 10]
    
    macs_per_pass = 0
    flops_per_pass = 0
    
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        
        # Linear layer cost
        layer_macs = n_in * n_out
        layer_flops = 2 * layer_macs + 2 * n_out # + Activation
        
        macs_per_pass += layer_macs
        flops_per_pass += layer_flops

    total_macs = 0
    total_flops = 0
    
    # 1. Initialization (Bottom-up)
    total_macs += macs_per_pass * batch_size
    total_flops += flops_per_pass * batch_size
    
    # 2. Inference Loops (T steps) -> 4x Matrix Mult per layer per step
    # (V forward, W forward, V backward error, W backward error)
    inference_macs = (macs_per_pass * 4) * batch_size * inference_steps
    inference_flops = (flops_per_pass * 4) * batch_size * inference_steps
    
    total_macs += inference_macs
    total_flops += inference_flops
    
    # 3. Weight Update (Training only) -> 2x Matrix Mult (dV, dW)
    if is_training:
        update_macs = (macs_per_pass * 2) * batch_size
        update_flops = (flops_per_pass * 2) * batch_size
        total_macs += update_macs
        total_flops += update_flops
        
    return total_macs, total_flops

# ---------------------------------------------------------
# 5. メインループ
# ---------------------------------------------------------
def main():
    device = torch.device(CONFIG['device'])
    print(f"=== bPC Training (Paper Hyperparams) on {device} ===")
    print(f"Config: {CONFIG}")
    
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG['batch_size'])
    model = bPC_Net(hidden_size=CONFIG['hidden_size']).to(device)
    
    # 論文 Table 8: AdamW with weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_weights'], weight_decay=CONFIG['weight_decay'])
    
    total_macs = 0.0
    total_flops = 0.0
    start_time = time.time()
    reached_target = False
    
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Batch':<6} | {'Val Acc':<10} | {'G-MACs':<15} | {'G-FLOPs':<15} | {'Time (s)':<10}")
    print("-" * 95)

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            
            # 1. Init
            activities = model.forward_init(images)
            
            # 2. Inference (T=8)
            with torch.no_grad():
                final_activities = model.run_inference(
                    activities, images, labels, 
                    steps=CONFIG['T_train'], 
                    lr_x=CONFIG['lr_activities'], 
                    is_training=True
                )
            
            # 3. Weight Update (Loss based)
            optimizer.zero_grad()
            loss_total = 0
            x = final_activities
            
            for l in range(len(model.layers)):
                layer = model.layers[l]
                # Disc Loss
                if l < len(model.layers):
                    pred_up = layer.activation(layer.V(x[l]))
                    loss_disc = 0.5 * CONFIG['alpha_disc'] * torch.sum((x[l+1].detach() - pred_up)**2)
                    loss_total += loss_disc
                
                # Gen Loss
                pred_down = layer.W(layer.activation(x[l+1].detach()))
                loss_gen = 0.5 * CONFIG['alpha_gen'] * torch.sum((x[l].detach() - pred_down)**2)
                loss_total += loss_gen
            
            loss_total.backward()
            optimizer.step()
            
            # Cost
            b_macs, b_flops = calculate_bpc_costs(model, bs, CONFIG['T_train'], is_training=True)
            total_macs += b_macs
            total_flops += b_flops
            
            # Validation
            if batch_idx % CONFIG['val_interval'] == 0:
                val_acc = run_validation(model, val_loader, device)
                elapsed = time.time() - start_time
                print(f"{epoch:<6} | {batch_idx:<6} | {val_acc:<10.2f} | {total_macs/1e9:<15.2f} | {total_flops/1e9:<15.2f} | {elapsed:<10.2f}")
                
                if val_acc >= CONFIG['target_accuracy']:
                    print("-" * 95)
                    print("Target Accuracy Reached!")
                    reached_target = True
                    break
        
        if reached_target:
            break

    # Final Test
    print("\nRunning Final Test (T=100)...")
    test_acc = run_validation(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    inf_macs, _ = calculate_bpc_costs(model, 1, CONFIG['T_test'], is_training=False)
    print(f"Total Training Cost: {total_macs/1e9:.2f} G-MACs")
    print(f"Inference Cost per Sample (T={CONFIG['T_test']}): {inf_macs/1e6:.2f} M-MACs")

def run_validation(model, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        activities = model.forward_init(images)
        with torch.no_grad():
            # 検証時は T=100 で推論 (論文 Table 8)
            final_activities = model.run_inference(
                activities, images, y_target=None, 
                steps=CONFIG['T_test'],
                lr_x=CONFIG['lr_activities'], 
                is_training=False
            )
        output = final_activities[-1]
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    main()