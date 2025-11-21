import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# FLOPs計算のためのライブラリ (インストールが必要です: pip install thop)
try:
    from thop import profile
except ImportError:
    profile = None
    print("Warning: 'thop' not found. FLOPs will be calculated using a dummy value.")

# --- 1. 基本演算クラスの定義 ---

class Conv(nn.Module):
    """通常の畳み込みブロック"""
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, affine=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """分離可能畳み込みブロック"""
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, affine=True):
        super(SepConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x):
        return self.op(x)


# --- 2. 探索結果に基づく最終モデルクラス (CIFAR-10対応) ---

class FinalNASModel(nn.Module):
    
    def __init__(self, num_classes=10, num_input_channels=3):
        super(FinalNASModel, self).__init__()
        
        layers = 17
        C = 63 # 初期チャネル数
        ops_code = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        kernel_sizes = [3] * layers
        
        # 1. ステム層 (入力チャネル数に合わせて変更)
        stem_multiplier = 2
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        # 2. ミックス層の構築
        C_prev = C_curr
        C_curr_base = C
        
        self.mixlayers = nn.ModuleList()
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr_base *= 2
                reduction = True
            else:
                reduction = False
                
            stride = 2 if reduction else 1
            kernel_size = kernel_sizes[i]
            C_out = C_curr_base

            if ops_code[i] == 0:
                mixlayer = SepConv(C_prev, C_out, kernel_size=kernel_size, stride=stride, affine=True)
            else:
                mixlayer = Conv(C_prev, C_out, kernel_size=kernel_size, stride=stride, affine=True)
            
            self.mixlayers.append(mixlayer)
            C_prev = C_out
            
        # 3. 最終層
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for mixlayer in self.mixlayers:
            x = mixlayer(x)
        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


# --- 3. データローディング関数 (CIFAR-10 & Augmentation) ---

def load_cifar10_data(batch_size=96):
    """CIFAR-10データセットをロードする"""
    
    # CIFAR-10の平均と標準偏差
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)

    # 訓練用: データ拡張 (Augmentation) を適用して過学習を防ぐ
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # ランダムに切り抜き
        transforms.RandomHorizontalFlip(),         # ランダムに左右反転
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    # テスト用: 正規化のみ
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# --- 4. トレーニング関数 ---

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}')
    
    acc = 100. * correct / total
    print(f'End of Epoch {epoch}, Train Accuracy: {acc:.2f}%')


# --- 5. テスト/評価関数とFLOPs計測 ---

def test(model, device, test_loader, criterion, input_size):
    model.eval()
    test_loss = 0
    correct = 0
    
    # 1順伝播あたりのFLOPs計測
    model_flops = 0.0
    if profile is not None:
        try:
            dummy_input = torch.randn(1, *input_size).to(device)
            total_ops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            model_flops = total_ops
        except Exception:
            model_flops = 1e8
    else:
        model_flops = 1e8

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)')
          
    return accuracy, model_flops


# --- 6. メイン実行ブロック ---

if __name__ == "__main__":
    
    # 1. ハイパーパラメータ設定
    epochs = 50 # CIFAR-10は難しいので多めに設定
    batch_size = 128 # GPUメモリに応じて調整
    learning_rate = 0.025 # SGDの初期学習率
    
    # 目標精度 (CIFAR-10の場合、90%は高い壁です。まずは85-90%を目指します)
    TARGET_ACCURACY = 90.0
    
    # 2. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. データローダーの準備
    train_loader, test_loader = load_cifar10_data(batch_size=batch_size)
    
    # 4. モデル準備 (入力3チャネル)
    num_input_channels = 3
    input_size = (3, 32, 32) # CIFAR-10 Size
    
    model = FinalNASModel(num_classes=10, num_input_channels=num_input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 最適化: SGD + Momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=3e-4)
    
    # 学習率スケジューラ: Cosine Annealing (学習率を徐々に下げて収束させる)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. トレーニングとコスト計測
    total_train_steps = len(train_loader.dataset)
    accumulated_flops = 0.0
    model_flops_per_inference = 0.0

    print("\n--- CIFAR-10 Training Start ---")
    start_time = time.time()
    
    # 初期チェック
    initial_accuracy, model_flops_per_inference = test(model, device, test_loader, criterion, input_size)
    
    if profile is not None:
        dummy_input = torch.randn(1, *input_size).to(device)
        _, total_params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"Model Parameters: {total_params / 1e6:.2f} M")

    for epoch in range(1, epochs + 1):
        
        # --- 訓練 ---
        train(model, device, train_loader, optimizer, criterion, epoch)
        
        # スケジューラの更新
        scheduler.step()
        
        # コスト累積 (Train = Forward + Backward approx 3x FLOPs)
        flops_per_epoch = model_flops_per_inference * total_train_steps * 3 
        accumulated_flops += flops_per_epoch
        
        # --- テスト ---
        accuracy, _ = test(model, device, test_loader, criterion, input_size)
        
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 目標達成チェック
        if accuracy >= TARGET_ACCURACY:
            end_time = time.time()
            wall_clock_time = end_time - start_time
            
            print("\n=========================================================")
            print(f"🎉 GOAL ACHIEVED! Accuracy {accuracy:.2f}% >= {TARGET_ACCURACY}%")
            print(f"🎯 Accumulated FLOPs Cost: {accumulated_flops:.2e}")
            print(f"⏱️ Wall-Clock Time Cost: {wall_clock_time:.2f} seconds")
            print("=========================================================")
            break
    else:
        end_time = time.time()
        wall_clock_time = end_time - start_time
        print(f"\n--- Training Finished (Goal Not Met) ---")
        print(f"Final Accuracy: {accuracy:.2f}%.")
        print(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")