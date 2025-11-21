import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from typing import List

# FLOPs計算のためのライブラリ (インストールが必要です: pip install thop)
try:
    from thop import profile
except ImportError:
    # thopがない場合のダミー設定
    profile = None
    print("Warning: 'thop' not found. FLOPs will be calculated using a dummy value.")

# --- 1. 基本演算クラスの定義 ---

class Conv(nn.Module):
    """通常の畳み込みブロック (ReLU -> Conv2d -> BatchNorm2d)"""
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
    """分離可能畳み込みブロック (ReLU -> Depthwise Conv -> Pointwise Conv -> BatchNorm2d)"""
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


# --- 2. 探索結果に基づく最終モデルクラス (MNIST用に調整) ---

class FinalNASModel(nn.Module):
    
    def __init__(self, num_classes=10, num_input_channels=1):
        super(FinalNASModel, self).__init__()
        
        layers = 17
        C = 63 # 初期チャネル数 (Model Width)
        ops_code = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        kernel_sizes = [3] * layers
        
        # 1. ステム層
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
            # NetworkMixのリダクションロジック (17層の場合、層 5, 11でチャネル倍増)
            if i in [layers // 3, 2 * layers // 3]: # i=5 と i=11
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


# --- 3. データローディング関数 ---

def load_mnist_data(batch_size=64):
    """MNISTデータセットをロードし、データローダーを返す"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# --- 4. トレーニング関数 ---

def train(model, device, train_loader, optimizer, criterion, epoch):
    """一エポック分のトレーニングを実行"""
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss

# --- 5. テスト/評価関数とFLOPs計測 ---

def test(model, device, test_loader, criterion, input_size):
    """モデルのテストとFLOPsの計測を行う"""
    model.eval()
    test_loss = 0
    correct = 0
    
    # 1順伝播あたりのFLOPs計測
    model_flops = 0.0
    if profile is not None:
        try:
            # ダミー入力 (Batch=1)
            dummy_input = torch.randn(1, *input_size).to(device)
            total_ops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            model_flops = total_ops # thopの出力値 (浮動小数点演算回数)
        except Exception as e:
            print(f"Error during FLOPs calculation with thop: {e}")
            model_flops = 1e8 # ダミー値
    else:
        model_flops = 1e8 # thopがない場合のダミー値 (100 MFLOPsを仮定)
    

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)')
          
    return accuracy, model_flops


# --- 6. メイン実行ブロック ---

if __name__ == "__main__":
    
    # 1. ハイパーパラメータ設定
    epochs = 20 # 最大エポック数
    batch_size = 128
    learning_rate = 0.01
    
    # [新規] コスト計測のための目標精度
    TARGET_ACCURACY = 99.0 # 目標分類精度 (%)
    
    # 2. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. データローダーの準備
    train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    
    # 4. モデル、損失関数、最適化アルゴリズムの準備
    num_input_channels = 1 # MNISTは1チャネル
    input_size = (num_input_channels, 28, 28) # MNISTの入力サイズ (C, H, W)
    
    model = FinalNASModel(num_classes=10, num_input_channels=num_input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # 5. トレーニングとコスト計測の実行
    
    total_train_steps = len(train_loader.dataset) # 訓練データの総数
    
    # コスト指標の初期化
    accumulated_flops = 0.0
    model_flops_per_inference = 0.0 # モデルの1順伝播あたりのFLOPs (テスト実行時に設定)

    print("\n--- Model Training Start with Cost Tracking ---")
    start_time = time.time()
    
    # 初回テスト実行でモデルのFLOPs/パラメータ数を計算
    initial_accuracy, model_flops_per_inference = test(model, device, test_loader, criterion, input_size)
    
    # パラメータ数の表示 (thopがある場合のみ)
    if profile is not None:
        dummy_input = torch.randn(1, *input_size).to(device)
        _, total_params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"Model Parameters: {total_params / 1e6:.2f} M")

    if initial_accuracy >= TARGET_ACCURACY:
        print(f"Goal met on initialization! Accuracy: {initial_accuracy:.2f}%. Cost: 0 FLOPs, 0s.")
    else:
        accuracy = initial_accuracy # 初期精度を設定
        for epoch in range(1, epochs + 1):
            
            # --- 訓練ステップ ---
            train(model, device, train_loader, optimizer, criterion, epoch)
            
            # 訓練ステップ後のFLOPsを累積
            # 訓練のFLOPsはテストの約3倍 (順伝播1 + 逆伝播2)
            flops_per_epoch = model_flops_per_inference * total_train_steps * 3 
            accumulated_flops += flops_per_epoch
            
            # --- テスト/評価ステップ ---
            accuracy, _ = test(model, device, test_loader, criterion, input_size)
            
            # 目標精度達成のチェック
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
            # 全エポックを回しきったが、目標精度に達しなかった場合
            end_time = time.time()
            wall_clock_time = end_time - start_time
            print(f"\n--- Training Finished (Goal Not Met) ---")
            print(f"Final Accuracy: {accuracy:.2f}%.")
            print(f"Max Accumulated FLOPs Cost: {accumulated_flops:.2e}")
            print(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")