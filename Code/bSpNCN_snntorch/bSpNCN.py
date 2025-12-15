import torch
import torch.nn as nn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt': 0.5,           # タイムステップ幅 (ms)
    'tau_m': 20.0,       # 膜時定数 (ms)
    'tau_j': 10.0,       # 電流時定数 (ms)
    'tau_tr': 30.0,      # トレース時定数 (ms)
    'kappa_j': 0.25,     # 電流減衰係数
    'gamma_m': 1.0,      # 電圧漏れ係数 (LIF)
    'R_m': 1.0,          # 膜抵抗
    'alpha_gen': 1.0,    # [重要] 生成誤差の係数 (SNN学習のため1.0推奨)
    'alpha_disc': 1.0,   # 識別誤差の係数
    'alpha_u': 0.005,    # 重み学習率 (発散を防ぐため少し下げて開始を推奨)
    'beta': 1.0,         # 誤差重み学習率
    'thresh': 0.4,       # 発火閾値
    'num_steps': 100,    # 推論ステップ数 (200msなら400, 動作確認なら100推奨)
    'batch_size': 64,
    'target_acc': 97.0,  # 目標精度 (%)
    'max_epochs': 50,    # 最大エポック数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class bSpNCNLayer(nn.Module):
    def __init__(self, idx, output_dim, config, is_data_layer=False):
        super().__init__()
        self.idx = idx
        self.dim = output_dim
        self.cfg = config
        self.is_data_layer = is_data_layer
        
    def init_state(self, batch_size, device):
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device)
        self.x = torch.zeros(batch_size, self.dim, device=device)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device)
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        dt = self.cfg['dt']
        
        # 1. 電流 j(t) の更新
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        # 2. 膜電位 v(t) の更新
        d_v = (-self.cfg['gamma_m'] * self.v + self.cfg['R_m'] * self.j)
        self.v = self.v + (dt / self.cfg['tau_m']) * d_v
        
        # 3. スパイク生成 & リセット
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        self.v = self.v * (1 - spikes) 
        
        # 4. トレース x(t) の更新 (分岐処理)
        if self.is_data_layer:
            # データ層: スパイクそのものをトレースとする
            self.x = spikes
        else:
            # 隠れ層: ローパスフィルタ (オイラー法)
            self.x = self.x + (dt / self.cfg['tau_tr']) * (-self.x) + spikes

class bSpNCN(nn.Module):
    def __init__(self, hidden_sizes, input_size=None, output_size=None, config=None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # --- 層構築 ---
        current_layer_idx = 0
        
        # 1. 入力予測層 (Layer 0)
        self.input_layer_idx = None
        if input_size is not None:
            self.layers.append(bSpNCNLayer(
                current_layer_idx, input_size, config, is_data_layer=True
            ))
            self.input_layer_idx = current_layer_idx
            current_layer_idx += 1
            
        # 2. 隠れ層 (Layer 1 ... L)
        self.hidden_layer_indices = []
        for size in hidden_sizes:
            self.layers.append(bSpNCNLayer(
                current_layer_idx, size, config, is_data_layer=False
            ))
            self.hidden_layer_indices.append(current_layer_idx)
            current_layer_idx += 1
            
        # 3. 出力予測層 (Layer L+1)
        self.output_layer_idx = None
        if output_size is not None:
            self.layers.append(bSpNCNLayer(
                current_layer_idx, output_size, config, is_data_layer=True
            ))
            self.output_layer_idx = current_layer_idx
            
        self.num_layers = len(self.layers)

        # --- 重み定義 ---
        self.W = nn.ParameterList()
        self.V = nn.ParameterList()
        self.E_gen = nn.ParameterList()
        self.E_disc = nn.ParameterList()

        for i in range(self.num_layers - 1):
            dim_lower = self.layers[i].dim
            dim_upper = self.layers[i+1].dim
            
            # 初期化: 0.05倍
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))
            self.E_gen.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))
            self.E_disc.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))

        # --- コスト計測用 ---
        self.total_synops = 0.0

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def clip_weights(self, max_norm=20.0):
        with torch.no_grad():
            for weight_list in [self.W, self.V, self.E_gen, self.E_disc]:
                for w in weight_list:
                    norms = w.norm(p=2, dim=1, keepdim=True)
                    mask = norms > max_norm
                    w.data = torch.where(mask, w * (max_norm / (norms + 1e-8)), w)

    def forward_dynamics(self, x_data=None, y_target=None, training_mode=True):
        """
        Algorithm 1 準拠 + SynOps計測
        """
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']

        # === 1. Update Phase ===
        
        # --- [Step 1-A] 入力層・出力層の更新 ---
        if self.input_layer_idx is not None:
            l0 = self.layers[self.input_layer_idx]
            if self.num_layers > 1:
                # W[0]: Layer 1(prev) -> Layer 0
                s_upper = self.layers[self.input_layer_idx + 1].s 
                z_gen_input = s_upper @ self.W[0].t()
                l0.update_state(total_input_current=z_gen_input)
                
                if training_mode:
                    self.total_synops += s_upper.sum().item() * l0.dim

        if self.output_layer_idx is not None:
            lL = self.layers[self.output_layer_idx]
            if self.num_layers > 1:
                # V[-1]: Layer L(prev) -> Layer L+1
                s_lower = self.layers[self.output_layer_idx - 1].s
                z_disc_output = s_lower @ self.V[-1].t()
                lL.update_state(total_input_current=z_disc_output)
                
                if training_mode:
                    self.total_synops += s_lower.sum().item() * lL.dim

        # --- [Step 1-B] 隠れ層の更新 ---
        for i in self.hidden_layer_indices:
            layer = self.layers[i]
            total_input = 0
            
            # (A) Local Error Terms
            total_input += (-layer.e_gen - layer.e_disc)
            
            # (B) Bottom-up Feedback
            if i > 0:
                e_gen_lower = self.layers[i-1].e_gen
                total_input += e_gen_lower @ self.E_gen[i-1].t()
                
            # (C) Top-down Feedback
            if i < self.num_layers - 1:
                e_disc_upper = self.layers[i+1].e_disc
                total_input += e_disc_upper @ self.E_disc[i].t()
            
            layer.update_state(total_input)

        # === 2. Prediction & Error Phase ===
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # --- Top-down Prediction (W) ---
            if i < self.num_layers - 1:
                s_upper = self.layers[i+1].s
                z_gen = s_upper @ self.W[i].t()
                
                if training_mode:
                    self.total_synops += s_upper.sum().item() * layer.dim

                if i == self.input_layer_idx:
                    if x_data is not None:
                        # データ(Spike) - 予測(Spike)
                        layer.e_gen = alpha_gen * (x_data - layer.x)
                    else:
                        layer.e_gen = torch.zeros_like(layer.x)
                else:
                    # 隠れ層: Trace - Prediction
                    layer.e_gen = alpha_gen * (layer.x - z_gen)
            else:
                layer.e_gen = torch.zeros_like(layer.x)

            # --- Bottom-up Prediction (V) ---
            if i > 0:
                s_lower = self.layers[i-1].s
                z_disc = s_lower @ self.V[i-1].t()
                
                if training_mode:
                    self.total_synops += s_lower.sum().item() * layer.dim

                if i == self.output_layer_idx:
                    if y_target is not None:
                        # ラベル(Spike) - 予測(Spike)
                        layer.e_disc = alpha_disc * (y_target - layer.x)
                    else:
                        layer.e_disc = torch.zeros_like(layer.x)
                else:
                    # 隠れ層: Trace - Prediction
                    layer.e_disc = alpha_disc * (layer.x - z_disc)
            else:
                layer.e_disc = torch.zeros_like(layer.x)

    def manual_weight_update(self):
        lr = self.config['alpha_u']
        beta = self.config['beta']
        
        with torch.no_grad():
            for i in range(self.num_layers - 1):
                s_lower = self.layers[i].s
                s_upper = self.layers[i+1].s
                e_gen_lower = self.layers[i].e_gen
                e_disc_upper = self.layers[i+1].e_disc
                
                # W Update
                self.W[i] += lr * (e_gen_lower.t() @ s_upper)
                
                # V Update
                self.V[i] += lr * (e_disc_upper.t() @ s_lower)
                
                # E Update
                self.E_gen[i] += lr * beta * (s_upper.t() @ e_gen_lower)
                self.E_disc[i] += lr * beta * (s_lower.t() @ e_disc_upper)

# --- 評価用関数 ---
def evaluate(model, dataloader, device, steps):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).view(images.size(0), -1)
            labels = labels.to(device)
            spike_in = spikegen.rate(images, num_steps=steps)
            
            model.reset_state(images.size(0), device)
            batch_output_sum = torch.zeros(images.size(0), 10).to(device)
            
            for t in range(steps):
                x_t = spike_in[t]
                model.forward_dynamics(x_data=x_t, y_target=None, training_mode=False)
                if model.output_layer_idx is not None:
                    batch_output_sum += model.layers[model.output_layer_idx].s
            
            _, predicted = torch.max(batch_output_sum, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- 共通の学習・実行関数 ---
def run_task(dataset_name='MNIST'):
    print(f"\n=== Running bSpNCN on {dataset_name} ===")
    print(f"Device: {CONFIG['device']}")
    
    # データセットごとの設定
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        input_size = 784
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        input_size = 784
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        input_size = 3072
    else:
        raise ValueError("Unknown dataset")

    # 分割 (Train:Val = 50000:10000 などを想定)
    # CIFAR-10はTrainが50000枚なので 45000:5000 などに適宜調整
    total_len = len(full_train)
    val_len = int(total_len * 0.1) # 10% for validation
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_train, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # モデル構築
    model = bSpNCN(
        hidden_sizes=[500, 500], 
        input_size=input_size, 
        output_size=10, 
        config=CONFIG
    ).to(CONFIG['device'])
    
    print(f"Start Training... (Steps: {CONFIG['num_steps']}, Batch: {CONFIG['batch_size']})")
    print(f"{'Epoch':<6} | {'Batch':<10} | {'SynOps (G)':<12} | {'Time (s)':<10}")
    
    start_time = time.time()
    reached_target = False
    
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            images = images.to(CONFIG['device']).view(CONFIG['batch_size'], -1)
            labels = labels.to(CONFIG['device'])
            
            targets = torch.zeros(CONFIG['batch_size'], 10).to(CONFIG['device'])
            targets.scatter_(1, labels.view(-1, 1), 1)
            
            # Rate Coding
            if dataset_name == 'CIFAR10':
                # [-1,1] -> [0,1]
                images = (images + 1) / 2
            
            # Clamp for safety
            images = torch.clamp(images, 0, 1)
            spike_in = spikegen.rate(images, num_steps=CONFIG['num_steps'])
            
            model.reset_state(CONFIG['batch_size'], CONFIG['device'])
            
            # Time Loop
            for t in range(CONFIG['num_steps']):
                x_t = spike_in[t]
                model.forward_dynamics(x_data=x_t, y_target=targets, training_mode=True)
                model.manual_weight_update()
                model.clip_weights(max_norm=20.0)
            
            # --- ログ出力 (10バッチごと) ---
            if batch_idx % 10 == 0:
                print(f"{epoch:<6} | {batch_idx:<10} | {model.total_synops/1e9:<12.4f} | {time.time()-batch_start:<10.2f}")

        # --- Validation ---
        print(f"Validating Epoch {epoch}...")
        val_acc = evaluate(model, val_loader, CONFIG['device'], CONFIG['num_steps'])
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Done | Val Acc: {val_acc:.2f}% | Total Time: {elapsed:.1f}s")
        
        if val_acc >= CONFIG['target_acc']:
            print("Target Accuracy Reached!")
            reached_target = True
            break
            
    # Final Test
    if reached_target:
        print("\nRunning Final Test...")
        test_acc = evaluate(model, test_loader, CONFIG['device'], CONFIG['num_steps'])
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"Total Training Cost: {model.total_synops/1e9:.4f} G-SynOps")
    else:
        print("Failed to reach target accuracy.")

if __name__ == "__main__":
    # 実行したいタスクを選択してください
    run_task('MNIST')
    # run_task('FashionMNIST')
    # run_task('CIFAR10')