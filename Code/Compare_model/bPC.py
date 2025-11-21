import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ==================================
# 1. モデルの定義 (bPC CNN)
# ==================================
class bPC_CNN(nn.Module):
    def __init__(self, inference_steps=20, alpha_disc=0.2, alpha_gen=0.8):
        super().__init__()
        self.inference_steps = inference_steps
        self.alpha_disc = alpha_disc # 識別的エネルギーの重み
        self.alpha_gen = alpha_gen   # 生成的エネルギーの重み
        self.activation = nn.ReLU()

        # ネットワーク層の定義
        # ボトムアップ (Discriminative) パス: Conv2d
        self.conv1_v = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_v = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_v = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_v = nn.Linear(128 * 8 * 8, 10)

        # トップダウン (Generative) パス: ConvTranspose2d
        self.fc_w = nn.Linear(10, 128 * 8 * 8)
        self.conv3_w = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_w = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_w = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        # ニューロン活動 (x) を nn.Parameter のリストとして保持
        # これらが推論中に更新される
        self.xs = [nn.Parameter(torch.zeros(1, 3, 32, 32)),      # x0 (input)
                   nn.Parameter(torch.zeros(1, 32, 32, 32)),     # x1
                   nn.Parameter(torch.zeros(1, 64, 16, 16)),     # x2
                   nn.Parameter(torch.zeros(1, 128, 8, 8)),      # x3
                   nn.Parameter(torch.zeros(1, 10))]             # x4 (output)
    
    def initialize_activities(self, image, label):
        """ニューロン活動を初期化する"""
        batch_size = image.size(0)
        
        # 各層の活動の形状をバッチサイズに合わせる
        self.xs = [
            nn.Parameter(torch.zeros(batch_size, 3, 32, 32, device=image.device)),
            nn.Parameter(torch.zeros(batch_size, 32, 32, 32, device=image.device)),
            nn.Parameter(torch.zeros(batch_size, 64, 16, 16, device=image.device)),
            nn.Parameter(torch.zeros(batch_size, 128, 8, 8, device=image.device)),
            nn.Parameter(torch.zeros(batch_size, 10, device=image.device))
        ]
        
        # 入力層と出力層をクランプ（固定）
        self.xs[0].data = image
        if label is not None:
             self.xs[-1].data = label

    def forward(self, image, label):
        """推論と学習のメインプロセス"""
        self.initialize_activities(image, label)

        # --- 推論ループ ---
        # ニューロン活動 x のみを最適化
        optimizer_x = optim.SGD(self.xs[1:-1], lr=0.1, momentum=0.9) # 中間層のみ更新

        for _ in range(self.inference_steps):
            optimizer_x.zero_grad()
            
            # エネルギー関数を計算
            energy = self.calculate_energy()
            
            # エネルギーの勾配を計算し、x を更新
            energy.backward(retain_graph=True)
            optimizer_x.step()

        # --- 学習ステップ ---
        # 推論後の最終的なエネルギーを計算
        final_energy = self.calculate_energy()
        return final_energy

    def calculate_energy(self):
        """エネルギー関数を計算"""
        e_disc = 0
        e_gen = 0
        
        # 識別的エネルギー (ボトムアップ予測誤差)
        pred_x1 = self.conv1_v(self.activation(self.xs[0]))
        e_disc += 0.5 * ((self.xs[1] - pred_x1) ** 2).sum()
        
        pred_x2 = self.conv2_v(self.activation(self.xs[1]))
        e_disc += 0.5 * ((self.xs[2] - pred_x2) ** 2).sum()
        
        pred_x3 = self.conv3_v(self.activation(self.xs[2]))
        e_disc += 0.5 * ((self.xs[3] - pred_x3) ** 2).sum()
        
        x3_flat = self.xs[3].view(self.xs[3].size(0), -1)
        pred_x4 = self.fc_v(self.activation(x3_flat))
        e_disc += 0.5 * ((self.xs[4] - pred_x4) ** 2).sum()

        # 生成的エネルギー (トップダウン予測誤差)
        pred_x3_flat = self.fc_w(self.activation(self.xs[4]))
        pred_x3_gen = pred_x3_flat.view_as(self.xs[3])
        e_gen += 0.5 * ((self.xs[3] - pred_x3_gen) ** 2).sum()

        pred_x2_gen = self.conv3_w(self.activation(self.xs[3]))
        e_gen += 0.5 * ((self.xs[2] - pred_x2_gen) ** 2).sum()
        
        pred_x1_gen = self.conv2_w(self.activation(self.xs[2]))
        e_gen += 0.5 * ((self.xs[1] - pred_x1_gen) ** 2).sum()
        
        pred_x0_gen = self.conv1_w(self.activation(self.xs[1]))
        e_gen += 0.5 * ((self.xs[0] - pred_x0_gen) ** 2).sum()
        
        total_energy = self.alpha_disc * e_disc + self.alpha_gen * e_gen
        return total_energy

    def predict(self, image):
        """評価時の予測"""
        # ラベルなしで活動を初期化
        self.initialize_activities(image, label=None)

        # 推論ループを実行して最終的な出力を得る
        optimizer_x = optim.SGD(self.xs[1:], lr=0.1, momentum=0.9) # 全ての隠れ層と出力層を更新

        for _ in range(self.inference_steps):
            optimizer_x.zero_grad()
            energy = self.calculate_energy()
            energy.backward()
            optimizer_x.step()

        return self.xs[-1].data

# ==================================
# 2. データセットの準備
# ==================================
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# ==================================
# 3. 学習と評価
# ==================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = bPC_CNN(inference_steps=20).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4) # 重み更新用のオプティマイザ

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = data
        inputs = inputs.to(device)
        
        # ラベルをone-hotベクトルに変換
        labels_onehot = nn.functional.one_hot(labels, num_classes=10).float().to(device)

        optimizer.zero_grad()

        # 推論と学習
        energy = model(inputs, labels_onehot)
        
        # 重みの更新
        energy.backward()
        optimizer.step()

        running_loss += energy.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

    # --- 評価 ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model.predict(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test images: {accuracy:.2f} %')

print('Finished Training')