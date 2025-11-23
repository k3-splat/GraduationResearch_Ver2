import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

# --- 設定 ---
CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3,  # Adam用
    'target_accuracy': 97.0,
    'max_epochs': 50,
    'hidden_size': 500,
    'val_interval': 100,    # 100バッチごとに検証
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- データセット準備 ---
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Train: 50k, Val: 10k
    train_dataset, val_dataset = random_split(full_train, [50000, 10000])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# --- MLPモデル (3 hidden layers) ---
class MLP(nn.Module):
    def __init__(self, hidden_size=500):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # 各層を定義（コスト計算でアクセスしやすくするため個別定義）
        self.fc1 = nn.Linear(784, hidden_size)
        self.act1 = nn.LeakyReLU(0.1)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.LeakyReLU(0.1)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.LeakyReLU(0.1)
        
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

# --- コスト計算 (MACs & FLOPs) ---
def calculate_costs(model, batch_size, is_training=True):
    """
    現在のバッチサイズに基づき、ForwardパスのMACsとFLOPsを計算。
    is_training=True の場合、Backwardパス（Forwardの約2倍）を考慮して3倍する。
    """
    macs = 0
    flops = 0
    
    # 隠れ層サイズ
    h_size = CONFIG['hidden_size']
    
    # --- Layer 1 (784 -> 500) ---
    # Linear
    macs += 784 * h_size
    flops += 2 * 784 * h_size  # 2 * MACs
    # Act (LeakyReLU)
    flops += 2 * h_size        # 要素数分 (比較 + 乗算)

    # --- Layer 2 (500 -> 500) ---
    # Linear
    macs += h_size * h_size
    flops += 2 * h_size * h_size
    # Act
    flops += 2 * h_size

    # --- Layer 3 (500 -> 500) ---
    # Linear
    macs += h_size * h_size
    flops += 2 * h_size * h_size
    # Act
    flops += 2 * h_size

    # --- Layer 4 (500 -> 10) ---
    # Linear
    macs += h_size * 10
    flops += 2 * h_size * 10
    # Output (通常SoftmaxなどはLossに含まれるが、モデル出力まではLinearのみ)
    
    # バッチサイズ倍
    total_macs = macs * batch_size
    total_flops = flops * batch_size
    
    if is_training:
        # 学習時は Forward(1) + Backward(2) = 3倍のコストと仮定
        return total_macs * 3, total_flops * 3
    else:
        # 推論時は Forwardのみ
        return total_macs, total_flops

# --- 実行メイン関数 ---
def main():
    device = torch.device(CONFIG['device'])
    print(f"=== MLP (ANN) Training on {device} ===")
    
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG['batch_size'])
    model = MLP(hidden_size=CONFIG['hidden_size']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    total_macs = 0.0
    total_flops = 0.0
    start_time = time.time()
    reached_target = False
    
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Batch':<6} | {'Val Acc':<10} | {'G-MACs':<15} | {'G-FLOPs':<15} | {'Time (s)':<10}")
    print("-" * 95)

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            current_batch_size = images.size(0)
            
            # Forward & Backward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # コスト加算
            batch_macs, batch_flops = calculate_costs(model, current_batch_size, is_training=True)
            total_macs += batch_macs
            total_flops += batch_flops
            
            # --- Validation ---
            if batch_idx % CONFIG['val_interval'] == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for v_img, v_lbl in val_loader:
                        v_img, v_lbl = v_img.to(device), v_lbl.to(device)
                        out = model(v_img)
                        _, pred = torch.max(out, 1)
                        correct += (pred == v_lbl).sum().item()
                        total += v_lbl.size(0)
                
                val_acc = 100 * correct / total
                elapsed_time = time.time() - start_time
                
                print(f"{epoch:<6} | {batch_idx:<6} | {val_acc:<10.2f} | {total_macs/1e9:<15.2f} | {total_flops/1e9:<15.2f} | {elapsed_time:<10.2f}")
                
                if val_acc >= CONFIG['target_accuracy']:
                    print("-" * 95)
                    print(f"Target accuracy reached!")
                    reached_target = True
                    break
                model.train()
        
        if reached_target:
            break

    # --- Final Test ---
    print("\nRunning Final Test...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # 推論時のコストも1回分計算して表示（参考用）
    # batch_size=1 のときの Forward Cost
    inference_macs, inference_flops = calculate_costs(model, 1, is_training=False)
    
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Total Training Cost (MACs):  {total_macs/1e9:.2f} G-MACs")
    print(f"Total Training Cost (FLOPs): {total_flops/1e9:.2f} G-FLOPs")
    print(f"Total Time: {time.time() - start_time:.2f} s")
    print("-" * 50)
    print(f"Inference Cost per Sample (MACs):  {inference_macs:.0f}")
    print(f"Inference Cost per Sample (FLOPs): {inference_flops:.0f}")

if __name__ == "__main__":
    main()