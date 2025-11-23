import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import time

# --- 設定 ---
CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3, # Adam
    'target_accuracy': 97.0,
    'max_epochs': 50,
    'hidden_size': 500,
    'num_steps': 25,       # タイムステップ数
    'beta': 0.95,          # 減衰率
    'val_interval': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- データセット準備 ---
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(full_train, [50000, 10000])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# --- SNNモデル (3 hidden layers) ---
class SNN(nn.Module):
    def __init__(self, hidden_size=500, beta=0.95):
        super(SNN, self).__init__()
        # 代理勾配
        spike_grad = surrogate.fast_sigmoid()
        
        # Layer 1
        self.fc1 = nn.Linear(784, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Output
        self.fc4 = nn.Linear(hidden_size, 10)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_input):
        # spike_input: [Time, Batch, 784] (Poisson Encoded)
        
        # 膜電位初期化
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        spk1_rec, spk2_rec, spk3_rec, spk4_rec = [], [], [], []
        
        for step in range(spike_input.size(0)):
            x = spike_input[step] # [Batch, 784]
            
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Output
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
            
        return (torch.stack(spk1_rec), torch.stack(spk2_rec), torch.stack(spk3_rec), torch.stack(spk4_rec))

# --- SynOps 計算 (syops-counter Logic) ---
def count_synops(model, spk_in, spk_tuple):
    """
    SynOps = 発火回数(Spike Count) × ファンアウト数(Fan-out)
    ポアソン入力(spk_in)も含めて計算する。
    """
    spk1, spk2, spk3, spk4 = spk_tuple
    synops = 0
    total_spikes = 0
    
    # 1. Input Spikes -> Hidden1 (Fan-out: fc1.out_features)
    s_in_count = spk_in.sum().item()
    synops += s_in_count * model.fc1.out_features
    total_spikes += s_in_count
    
    # 2. Hidden1 Spikes -> Hidden2
    s1_count = spk1.sum().item()
    synops += s1_count * model.fc2.out_features
    total_spikes += s1_count
    
    # 3. Hidden2 Spikes -> Hidden3
    s2_count = spk2.sum().item()
    synops += s2_count * model.fc3.out_features
    total_spikes += s2_count
    
    # 4. Hidden3 Spikes -> Output
    s3_count = spk3.sum().item()
    synops += s3_count * model.fc4.out_features
    total_spikes += s3_count

    # Output層のスパイクは次の結合がないためSynOpsには加算しないが、
    # スパイク総数には含める
    total_spikes += spk4.sum().item()
    
    return synops, total_spikes

# --- 実行メイン関数 ---
def main():
    device = torch.device(CONFIG['device'])
    print(f"=== SNN Training (Poisson Input) on {device} ===")
    
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG['batch_size'])
    model = SNN(hidden_size=CONFIG['hidden_size'], beta=CONFIG['beta']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    total_synops = 0.0
    total_spikes = 0.0
    start_time = time.time()
    reached_target = False
    
    print(f"{'Epoch':<6} | {'Batch':<6} | {'Val Acc':<10} | {'G-SynOps':<15} | {'Spikes(M)':<10} | {'Time (s)':<10}")
    print("-" * 75)

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            
            # --- Poisson Encoding ---
            # [Batch, 1, 28, 28] -> [Time, Batch, 784]
            # rate encoding: 画素値(0-1)を確率としてベルヌーイ試行
            spike_input = spikegen.rate(images.view(images.size(0), -1), num_steps=CONFIG['num_steps'])
            
            # Forward
            spk_tuple = model(spike_input)
            spk4_out = spk_tuple[3] # [Time, Batch, 10]
            
            # Loss: 時間方向の合計発火数を予測値とする
            outputs = spk4_out.sum(dim=0)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Cost計測 (Forward SynOps)
            batch_synops, batch_spikes = count_synops(model, spike_input, spk_tuple)
            total_synops += batch_synops
            total_spikes += batch_spikes
            
            # --- Validation ---
            if batch_idx % CONFIG['val_interval'] == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for v_img, v_lbl in val_loader:
                        v_img, v_lbl = v_img.to(device), v_lbl.to(device)
                        # Validation時もエンコーディングが必要
                        v_spk_in = spikegen.rate(v_img.view(v_img.size(0), -1), num_steps=CONFIG['num_steps'])
                        v_out_tuple = model(v_spk_in)
                        v_outputs = v_out_tuple[3].sum(dim=0)
                        _, pred = torch.max(v_outputs, 1)
                        correct += (pred == v_lbl).sum().item()
                        total += v_lbl.size(0)
                
                val_acc = 100 * correct / total
                elapsed_time = time.time() - start_time
                
                print(f"{epoch:<6} | {batch_idx:<6} | {val_acc:<10.2f} | {total_synops/1e9:<15.2f} | {total_spikes/1e6:<10.2f} | {elapsed_time:<10.2f}")
                
                if val_acc >= CONFIG['target_accuracy']:
                    print("-" * 75)
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
            spike_input = spikegen.rate(images.view(images.size(0), -1), num_steps=CONFIG['num_steps'])
            spk_tuple = model(spike_input)
            outputs = spk_tuple[3].sum(dim=0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Total Forward SynOps: {total_synops/1e9:.2f} G-SynOps")
    print(f"Total Spikes Generated: {total_spikes/1e6:.2f} M")
    print(f"Total Time: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    main()