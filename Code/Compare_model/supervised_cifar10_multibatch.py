import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
import time
import sys

# --- ハイパーパラメータ設定 ---
BATCH_SIZE = 128
BETA = 0.9  # ニューロンの減衰率
TIME_STEPS = 10  # シミュレーションのタイムステップ数
LEARNING_RATE = 1e-3
TARGET_ACCURACY = 0.60  # 目標テスト精度 (60%に設定。必要に応じて変更してください)
MAX_EPOCHS = 50  # 無限ループ防止用の最大エポック数

# --- デバイス設定 ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# --- データセットの準備 (CIFAR-10) ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),  # 計算量削減のためグレースケール化 (RGBのままも可能)
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --- SNNモデルの定義 ---
class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid() # 勾配近似関数

        # 畳み込み層1: 1ch -> 12ch, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        # 畳み込み層2: 12ch -> 64ch, 5x5 kernel
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        # 全結合層: Flatten後のサイズ計算 64 * 5 * 5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 10)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        # 膜電位の初期化
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 出力スパイクの記録用
        spk3_rec = []
        
        # 総スパイク数カウント用 (バッチ内の全ニューロンの発火合計)
        batch_total_spikes = 0

        # タイムステップごとの処理
        for step in range(TIME_STEPS):
            # Layer 1
            cur1 = self.conv1(x) # 入力を電流として直接印加
            spk1, mem1 = self.lif1(cur1, mem1)
            x_pool1 = self.pool1(spk1)

            # Layer 2
            cur2 = self.conv2(x_pool1)
            spk2, mem2 = self.lif2(cur2, mem2)
            x_pool2 = self.pool2(spk2)
            
            # Flatten
            x_flat = x_pool2.view(BATCH_SIZE, -1)

            # Layer 3 (Output)
            cur3 = self.fc1(x_flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # 記録
            spk3_rec.append(spk3)
            
            # 全層のスパイク数を加算 (Gradient計算からは切り離してカウントのみに使用)
            with torch.no_grad():
                batch_total_spikes += spk1.sum() + spk2.sum() + spk3.sum()

        return torch.stack(spk3_rec, dim=0), batch_total_spikes

net = SCNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# --- 計測変数の初期化 ---
total_start_time = time.time()
cumulative_samples = 0
cumulative_spikes = 0
epoch = 0
best_acc = 0.0

print("\n--- Training Started ---")
print(f"Target Accuracy: {TARGET_ACCURACY * 100}%")

# --- 学習ループ ---
while best_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS:
    epoch += 1
    net.train()
    train_loss_accum = 0
    
    # 1. 学習フェーズ
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        spk_rec, batch_spikes = net(data)
        
        # コスト計上
        cumulative_samples += BATCH_SIZE
        cumulative_spikes += batch_spikes.item()

        # Loss計算 (時間方向の平均発火率に対するCrossEntropy)
        loss_val = torch.zeros((1), dtype=torch.float, device=device)
        # Rate coding loss: スパイクの総数をクラス分類に使用
        loss_val = loss_fn(spk_rec.sum(0), targets)

        # Backward pass
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss_accum += loss_val.item()

    # 2. テストフェーズ
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data) # テスト時のスパイク数は学習コストに含めない（要件次第で変更可）

            # 最も発火回数が多かったクラスを予測とする
            _, predicted = spk_rec.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    current_acc = correct / total
    best_acc = max(best_acc, current_acc)
    
    current_time = time.time() - total_start_time
    
    # ログ出力
    print(f"Epoch {epoch}: "
          f"Acc: {current_acc*100:.2f}% | "
          f"Total Time: {current_time:.1f}s | "
          f"Total Samples: {cumulative_samples} | "
          f"Total Spikes: {int(cumulative_spikes)}")

# --- 結果サマリー ---
print("\n--- Result Summary ---")
if best_acc >= TARGET_ACCURACY:
    print(f"Target Accuracy ({TARGET_ACCURACY*100}%) Reached!")
else:
    print(f"Max Epochs ({MAX_EPOCHS}) Reached. Final Accuracy: {best_acc*100:.2f}%")

print(f"Total Execution Time : {time.time() - total_start_time:.2f} seconds")
print(f"Total Data Processed : {cumulative_samples} images")
print(f"Total Spike Count    : {int(cumulative_spikes)} spikes")