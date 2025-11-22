import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================

class Config:
    # バッチサイズ: VRAMに合わせて調整してください（論文は256ですが64で安定させます）
    batch_size = 64  
    epochs = 20
    
    # 学習率設定 (Reduction='mean' に合わせて調整)
    lr_x = 0.05       # 推論（ニューロン活動）の更新率
    lr_theta = 0.001  # 重みの学習率
    
    # 推論ステップ数 (論文 Table 7 準拠)
    T_train = 32     
    T_test = 100     
    
    # エネルギー項の重み付け
    # reduction='mean' を使うため、極端に小さな値ではなく 0.1 程度でバランスを取ります
    alpha_disc = 1.0 
    alpha_gen = 0.1  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

# 再現性の確保
torch.manual_seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.seed)

# ==========================================
# Data Loading (CIFAR-10)
# ==========================================

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 画像を [-1, 1] に正規化 (Tanhの出力範囲に合わせるため)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # 論文とは異なり、全ての学習データ(50k)とテストデータ(10k)を使用
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# ==========================================
# VGG-bPC Model Implementation
# ==========================================

class VGG_bPC(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_bPC, self).__init__()
        
        # Layer dimensions based on Sec 4.3 & Table 3
        # L1 (Input): [3, 32, 32]
        # L2: [128, 16, 16]
        # L3: [256, 8, 8]
        # L4: [512, 4, 4]
        # L5: [512, 2, 2]
        # L6 (Label): [10]
        
        # --- Discriminative Weights (Bottom-up: V) ---
        self.V1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.V2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.V3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.V4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.V5 = nn.Linear(512 * 2 * 2, num_classes) 
        
        # --- Generative Weights (Top-down: W) ---
        self.W6 = nn.Linear(num_classes, 512 * 2 * 2)
        self.W5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.W4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.W3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.W2 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

        # Activations
        self.relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()         # For image generation output
        self.pool = nn.MaxPool2d(2, 2)
        
    def init_layers(self, x_in):
        """
        初期化: Bottom-up feedforward sweep
        """
        batch_size = x_in.shape[0]
        
        with torch.no_grad():
            # L1 -> L2
            h1 = self.pool(self.relu(self.V1(x_in)))
            # L2 -> L3
            h2 = self.pool(self.relu(self.V2(h1)))
            # L3 -> L4
            h3 = self.pool(self.relu(self.V3(h2)))
            # L4 -> L5
            h4 = self.pool(self.relu(self.V4(h3)))
            # L5 -> L6
            h4_flat = h4.view(batch_size, -1)
            h5 = self.V5(h4_flat) 
            
        # 勾配計算用に requires_grad=True に設定した変数をリスト化
        nodes = [
            x_in.detach().clone(),                    # x1 (Image)
            h1.detach().clone().requires_grad_(True), # x2
            h2.detach().clone().requires_grad_(True), # x3
            h3.detach().clone().requires_grad_(True), # x4
            h4.detach().clone().requires_grad_(True), # x5
            h5.detach().clone().requires_grad_(True)  # x6 (Label)
        ]
        return nodes

    def compute_energy(self, nodes):
        """
        Total Energy E = E_disc + E_gen
        修正: reduction='mean' を使用して勾配爆発を防ぎ、次元差を吸収します。
        """
        x1, x2, x3, x4, x5, x6 = nodes
        
        # --- Discriminative Errors (Bottom-up) ---
        
        # x1 -> x2
        pred_x2 = self.pool(self.relu(self.V1(x1)))
        err_disc_2 = F.mse_loss(x2, pred_x2, reduction='mean')
        
        # x2 -> x3
        pred_x3 = self.pool(self.relu(self.V2(x2)))
        err_disc_3 = F.mse_loss(x3, pred_x3, reduction='mean')
        
        # x3 -> x4
        pred_x4 = self.pool(self.relu(self.V3(x3)))
        err_disc_4 = F.mse_loss(x4, pred_x4, reduction='mean')
        
        # x4 -> x5
        pred_x5 = self.pool(self.relu(self.V4(x4)))
        err_disc_5 = F.mse_loss(x5, pred_x5, reduction='mean')
        
        # x5 -> x6 (Identity mapping per paper requirement)
        pred_x6 = self.V5(x5.view(x5.size(0), -1))
        err_disc_6 = F.mse_loss(x6, pred_x6, reduction='mean')
        
        E_disc = 0.5 * Config.alpha_disc * (err_disc_2 + err_disc_3 + err_disc_4 + err_disc_5 + err_disc_6)

        # --- Generative Errors (Top-down) ---
        
        # x6 -> x5 (LeakyReLU)
        pred_gen_x5 = self.W6(x6).view_as(x5)
        err_gen_5 = F.mse_loss(x5, self.relu(pred_gen_x5), reduction='mean')

        # x5 -> x4 (LeakyReLU)
        pred_x4_gen = self.W5(self.relu(x5))
        err_gen_4 = F.mse_loss(x4, self.relu(pred_x4_gen), reduction='mean')
        
        # x4 -> x3 (LeakyReLU)
        pred_x3_gen = self.W4(self.relu(x4))
        err_gen_3 = F.mse_loss(x3, self.relu(pred_x3_gen), reduction='mean')
        
        # x3 -> x2 (LeakyReLU)
        pred_x2_gen = self.W3(self.relu(x3))
        err_gen_2 = F.mse_loss(x2, self.relu(pred_x2_gen), reduction='mean')
        
        # x2 -> x1 (Tanh per paper requirement)
        pred_x1_gen = self.W2(self.relu(x2))
        err_gen_1 = F.mse_loss(x1, self.tanh(pred_x1_gen), reduction='mean')
        
        E_gen = 0.5 * Config.alpha_gen * (err_gen_1 + err_gen_2 + err_gen_3 + err_gen_4 + err_gen_5)
        
        return E_disc + E_gen, E_disc, E_gen

# ==========================================
# Training & Evaluation Functions
# ==========================================

def run_pc_inference(model, images, labels, steps, mode='train'):
    """
    推論フェーズ: ニューロン活動 x を更新してエネルギーを最小化
    """
    nodes = model.init_layers(images)
    
    # x6 (Label層) のクランプ処理
    if mode == 'train':
        with torch.no_grad():
            one_hot = F.one_hot(labels, num_classes=10).float().to(Config.device)
            nodes[-1].data = one_hot
        
        # ラベル層以外を更新対象にする
        train_nodes = [n for i, n in enumerate(nodes) if n.requires_grad and i != 5]
        x_optim = optim.SGD(train_nodes, lr=Config.lr_x)
        
    elif mode == 'test':
        # ラベル層も含めて更新（推論）する
        x_optim = optim.SGD([n for n in nodes if n.requires_grad], lr=Config.lr_x)

    model.eval() # BatchNorm等がないので影響小だが作法として
    
    losses = []
    
    for t in range(steps):
        x_optim.zero_grad()
        
        total_energy, _, _ = model.compute_energy(nodes)
        total_energy.backward()
        
        x_optim.step()
        
        # Train時はラベルを再度固定（念のため）
        if mode == 'train':
            with torch.no_grad():
                one_hot = F.one_hot(labels, num_classes=10).float().to(Config.device)
                nodes[-1].data = one_hot
        
        losses.append(total_energy.item())
        
    return nodes, losses

def train_one_epoch(model, dataloader, optimizer_theta):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(Config.device), labels.to(Config.device)
        
        # 1. 推論フェーズ (x の更新)
        nodes, _ = run_pc_inference(model, images, labels, Config.T_train, mode='train')
        
        # 2. 学習フェーズ (パラメータ θ の更新)
        optimizer_theta.zero_grad()
        
        # 推論後の平衡状態でのエネルギーを再計算
        # (ここで計算グラフを重みパラメータに接続する)
        total_energy, _, _ = model.compute_energy(nodes)
        
        total_energy.backward()
        
        # 勾配クリッピング（安定化のため）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer_theta.step()
        
        running_loss += total_energy.item()
        pbar.set_postfix({'Energy': f"{total_energy.item():.4f}"})

    return running_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    # パラメータ更新はしないが、推論のために勾配計算(xについて)は必要
    # torch.no_grad()の外側、あるいは内側でenable_grad()を使う
    
    with torch.no_grad(): # 重み固定
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images, labels = images.to(Config.device), labels.to(Config.device)
            
            # 推論のために一時的に勾配計算を有効化
            with torch.enable_grad():
                nodes, _ = run_pc_inference(model, images, labels, Config.T_test, mode='test')
            
            # 最上位層 x6 の最大値を持つクラスを予測とする
            pred_logits = nodes[-1]
            predictions = torch.argmax(pred_logits, dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
    return 100 * correct / total

# ==========================================
# Main
# ==========================================

def main():
    print(f"=== VGG-bPC Fully Supervised Training ===")
    print(f"Device: {Config.device}")
    print(f"Config: batch={Config.batch_size}, lr_x={Config.lr_x}, reduction='mean'")
    
    print("Loading Data...")
    train_loader, test_loader = get_dataloaders()
    
    print("Initializing Model...")
    model = VGG_bPC(num_classes=10).to(Config.device)
    
    # 重み学習用オプティマイザ
    optimizer_theta = optim.AdamW(model.parameters(), lr=Config.lr_theta, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_theta, T_max=Config.epochs)

    print("Starting Training...")
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(Config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer_theta)
        acc = evaluate(model, test_loader)
        
        scheduler.step()
        
        history['loss'].append(loss)
        history['acc'].append(acc)
        
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {loss:.6f} | Test Acc: {acc:.2f}%")
        
    # 結果のプロット
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Energy Loss (Mean)', color=color)
    ax1.plot(history['loss'], color=color, marker='o', label='Train Energy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(history['acc'], color=color, marker='s', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("VGG-bPC CIFAR-10 Training Curve")
    plt.savefig("training_curve.png")
    plt.show()

    print("Training Complete.")

if __name__ == "__main__":
    main()
