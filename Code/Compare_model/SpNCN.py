import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import snntorch as snn
from snntorch import surrogate
import math
import time

# ---------------------------------------------------------
# 1. 設定とハイパーパラメータ (論文 Appendix A.4 / Section 3.1 準拠)
# ---------------------------------------------------------
# 時間解像度と定数
DT = 0.25        # ms (Integration time step)
TAU_M = 20.0     # ms (Membrane time constant)
TAU_SYN = 10.0   # ms (Synaptic/Current time constant)
TAU_TR = 30.0    # ms (Trace time constant)

# snnTorch用の減衰率 (Decay factors) 計算
# V(t+1) = (1 - dt/tau) * V(t) + ... -> beta = 1 - dt/tau
ALPHA = 1.0 - (DT / TAU_SYN) # Current decay
BETA  = 1.0 - (DT / TAU_M)   # Membrane decay
KAPPA = 1.0 - (DT / TAU_TR)  # Trace decay

CONFIG = {
    'batch_size': 128,
    'lr': 0.055,             # 論文: step size alpha_u = 0.055
    'target_accuracy': 97.0,
    'max_epochs': 50,
    'hidden_size': 500,
    'val_interval': 100,
    'time_steps': int(200 / DT), # 論文: T_st = 200ms -> 200/0.25 = 800 steps
    'threshold': 0.4,        # 論文: v_thr = 0.4
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ---------------------------------------------------------
# 2. データセット (Poisson Spike Generation)
# ---------------------------------------------------------
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(), # [0, 1]
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
# 3. モデル定義: SpNCN Layer (Synaptic LIF)
# ---------------------------------------------------------
class SpNCN_Layer(nn.Module):
    def __init__(self, size, input_size_bottom, input_size_top=0, is_output=False):
        super().__init__()
        self.size = size
        
        # Synaptic LIF Neuron (Current based)
        # alpha: Current decay, beta: Membrane decay
        # init_hidden=True なので forward は spk のみを返す
        self.lif = snn.Synaptic(alpha=ALPHA, beta=BETA, threshold=CONFIG['threshold'], 
                                reset_mechanism='zero', init_hidden=True)
        
        # Trace (low-pass filtered spike) 用のバッファ
        self.register_buffer('trace', torch.zeros(1, size))
        
        # --- Synapses (Weights) ---
        # Bottom-up (Recognition): V (Input -> This)
        self.W_rec = nn.Linear(input_size_bottom, size, bias=False)
        
        # Top-down (Generative): W (Higher -> This)
        if input_size_top > 0:
            self.W_gen = nn.Linear(input_size_top, size, bias=False)
        else:
            self.W_gen = None
            
        # Error Feedback Weights (E): Error -> This (Current)
        self.W_err = nn.Linear(size, size, bias=False)
        
        # 重み初期化
        nn.init.xavier_uniform_(self.W_rec.weight)
        if self.W_gen: nn.init.xavier_uniform_(self.W_gen.weight)
        nn.init.eye_(self.W_err.weight) 

    def reset_state(self, batch_size, device):
        self.lif.reset_hidden()
        self.trace = torch.zeros(batch_size, self.size, device=device)

    def update_trace(self, spike):
        # z(t+1) = z(t) * kappa + s(t)
        self.trace = self.trace * KAPPA + spike
        return self.trace

    def forward_dynamic(self, input_signal, error_signal_self=None):
        """
        1ステップ分のニューロン更新
        """
        total_current = input_signal
        if error_signal_self is not None:
            # 誤差フィードバック (Negative Error Feedback)
            feedback = self.W_err(error_signal_self)
            total_current = total_current - feedback

        # LIF更新 (init_hidden=Trueなので戻り値はspkのみ)
        spk = self.lif(total_current) 
        
        # Trace更新
        tr = self.update_trace(spk)
        
        return spk, tr

# ---------------------------------------------------------
# 4. ネットワーク全体 (Case 2: Continual Learning Architecture)
# ---------------------------------------------------------
class SpNCN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = CONFIG['hidden_size']
        
        # H1 (Bottom of hierarchy, connects to X and Y)
        # 入力は [X(784), Y(10)] の結合ベクトルを受ける
        self.h1 = SpNCN_Layer(hidden_size, 784 + 10, input_size_top=0) 
        
        # Prediction Layers (LIF neurons for reconstruction)
        # 論文 Eq 14-19: 予測もニューロン活動として生成
        self.pred_x = snn.Synaptic(alpha=ALPHA, beta=BETA, threshold=CONFIG['threshold'], reset_mechanism='zero', init_hidden=True)
        self.pred_y = snn.Synaptic(alpha=ALPHA, beta=BETA, threshold=CONFIG['threshold'], reset_mechanism='zero', init_hidden=True)
        
        # Top-down weights from H1 to Prediction Layers
        self.W_gen_x = nn.Linear(hidden_size, 784, bias=False)
        self.W_gen_y = nn.Linear(hidden_size, 10, bias=False)
        
        # Trace buffers for pred layers
        self.trace_pred_x = None
        self.trace_pred_y = None

        # 重み初期化
        nn.init.xavier_uniform_(self.W_gen_x.weight)
        nn.init.xavier_uniform_(self.W_gen_y.weight)

    def reset_state(self, batch_size, device):
        self.h1.reset_state(batch_size, device)
        self.pred_x.reset_hidden()
        self.pred_y.reset_hidden()
        self.trace_pred_x = torch.zeros(batch_size, 784, device=device)
        self.trace_pred_y = torch.zeros(batch_size, 10, device=device)

# ---------------------------------------------------------
# 5. ST-LRA 学習関数 (Manual Update)
# ---------------------------------------------------------
def run_simulation_and_learn(model, images, labels_onehot, config):
    device = images.device
    batch_size = images.size(0)
    model.reset_state(batch_size, device)
    
    # H1スパイクの初期状態 (t-1)
    s_h1 = torch.zeros(batch_size, config['hidden_size'], device=device)
    
    # 重み勾配の蓄積用バッファ
    grad_W_gen_x = torch.zeros_like(model.W_gen_x.weight)
    grad_W_gen_y = torch.zeros_like(model.W_gen_y.weight)
    
    # W_rec (Input->H1) の勾配: H1には [X, Y] が入力されるので重みサイズに注意
    # model.h1.W_rec.weight の形状は [500, 794]
    grad_W_rec = torch.zeros_like(model.h1.W_rec.weight)
    
    total_synops = 0
    
    # Poisson Generation Loop
    for t in range(config['time_steps']):
        # 入力スパイク生成 (Bernoulli trial)
        spk_x = torch.bernoulli(images).to(device)
        spk_y = torch.bernoulli(labels_onehot).to(device)
        
        # --- 1. Generate Predictions (Top-down) ---
        # H1 -> X (Reconstruction)
        curr_pred_x = model.W_gen_x(s_h1)
        spk_pred_x = model.pred_x(curr_pred_x) # 戻り値は1つ
        model.trace_pred_x = model.trace_pred_x * KAPPA + spk_pred_x 
        
        # H1 -> Y (Prediction)
        curr_pred_y = model.W_gen_y(s_h1)
        spk_pred_y = model.pred_y(curr_pred_y) # 戻り値は1つ
        model.trace_pred_y = model.trace_pred_y * KAPPA + spk_pred_y 
        
        # --- 2. Compute Errors (Eq 21) ---
        # Input Trace (z^0) の更新
        if t == 0:
            trace_x = spk_x
            trace_y = spk_y
        else:
            trace_x = trace_x * KAPPA + spk_x
            trace_y = trace_y * KAPPA + spk_y
            
        err_x = trace_x - model.trace_pred_x
        err_y = trace_y - model.trace_pred_y 
        
        # --- 3. Update Hidden Layer (H1) ---
        # H1への入力: Rec(X) + Rec(Y)
        joint_input_spk = torch.cat([spk_x, spk_y], dim=1)
        in_h1_rec = model.h1.W_rec(joint_input_spk)
        
        # Case 2: ラベル誤差(err_y)はH1にフィードバックしない
        # H1自身の予測誤差(err_h1)があればフィードバックするが、今回は省略
        curr_h1 = in_h1_rec 
        
        new_spk_h1, trace_h1 = model.h1.forward_dynamic(curr_h1)
        
        # --- 4. Accumulate Gradients (ST-LRA) ---
        # ST-LRA Rule: Delta W = Error_post * Spike_pre^T
        
        # W_gen_x (H1 -> X): post=X(err), pre=H1(s)
        # grad = err_x.T @ s_h1
        g_gen_x = torch.matmul(err_x.T, s_h1) / batch_size
        grad_W_gen_x += g_gen_x
        
        # W_gen_y (H1 -> Y): post=Y(err), pre=H1(s)
        g_gen_y = torch.matmul(err_y.T, s_h1) / batch_size
        grad_W_gen_y += g_gen_y
        
        # W_rec (Input -> H1): post=H1(err?), pre=Input(s)
        # Bottom-up重みの更新には、上位層(H1)の予測誤差が必要だが、
        # 論文 Case 2 では「H1は教師なし的に学習」あるいは「再構成誤差を使用」
        # ここではシンプルに Hebbian項 (pre * post) と Error項を組み合わせる実装が一般的だが、
        # 厳密なST-LRAでは H1の予測誤差 (trace_h1 - top_down_from_H2) を使う。
        # 今回はH2を省略しているため、簡易的なOja則に近い更新、あるいは
        # "W_rec is updated to minimize prediction error of X and Y" という逆方向の解釈も可能。
        # ここでは安定学習のため、Generative Weightの更新を主とし、
        # W_rec は W_gen の転置(対称重み)に近づける制約を入れるか、
        # 単純に固定(Random Projection)にする手法もある(df-DRTP)。
        # 今回は W_gen の学習のみを有効化し、W_recは固定とする（論文のdf-DRTP設定に近い安定策）
        # ※もしW_recも学習させるなら、H1のエラー定義が必要。
        
        # Update state for next step
        s_h1 = new_spk_h1
        
        # SynOps Counting
        fan_out_h1 = 784 + 10 # H1 -> X, Y
        # H1_in = (784+10) -> H1
        total_synops += s_h1.sum().item() * fan_out_h1
        total_synops += joint_input_spk.sum().item() * CONFIG['hidden_size']

    # --- 5. Apply Updates ---
    with torch.no_grad():
        lr = config['lr']
        # 勾配上昇 (ST-LRAは相関学習なので加算)
        # 重み減衰 (Weight Decay) も手動適用
        model.W_gen_x.weight += lr * grad_W_gen_x - (lr * 0.001 * model.W_gen_x.weight)
        model.W_gen_y.weight += lr * grad_W_gen_y - (lr * 0.001 * model.W_gen_y.weight)
        
    return total_synops

# ---------------------------------------------------------
# 6. バリデーション関数
# ---------------------------------------------------------
def run_validation(model, loader, device, config):
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        model.reset_state(batch_size, device)
        
        s_h1 = torch.zeros(batch_size, config['hidden_size'], device=device)
        
        # 推論ループ
        # Test時はラベル入力なし（Yからの入力はゼロ、あるいはランダム）
        # ここではゼロ入力として扱う
        spk_y_dummy = torch.zeros(batch_size, 10, device=device)
        
        # 蓄積用
        sum_spk_y = torch.zeros(batch_size, 10, device=device)
        
        for t in range(config['time_steps']):
            spk_x = torch.bernoulli(images).to(device)
            
            # H1 -> Y Prediction
            curr_pred_y = model.W_gen_y(s_h1)
            spk_pred_y = model.pred_y(curr_pred_y) # 戻り値1つ
            
            sum_spk_y += spk_pred_y
            
            # H1 Update (Input from X only)
            joint_input = torch.cat([spk_x, spk_y_dummy], dim=1)
            in_h1 = model.h1.W_rec(joint_input)
            new_spk_h1, _ = model.h1.forward_dynamic(in_h1)
            
            s_h1 = new_spk_h1
            
        # Rate-based prediction
        _, predicted = torch.max(sum_spk_y, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return 100 * correct / total

# ---------------------------------------------------------
# 7. メイン実行部
# ---------------------------------------------------------
def main():
    print(f"=== SpNCN Strict Full (Joint Architecture) on {CONFIG['device']} ===")
    device = torch.device(CONFIG['device'])
    model = SpNCN_Net().to(device)
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG['batch_size'])
    
    total_synops = 0.0
    start_time = time.time()
    
    print("-" * 90)
    print(f"{'Epoch':<6} | {'Batch':<6} | {'Val Acc':<10} | {'G-SynOps':<15} | {'Time (s)':<10}")
    print("-" * 90)

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device) # Device fix
            
            # Labels to One-Hot Spike Probability
            labels_onehot = torch.zeros(images.size(0), 10, device=device)
            labels_onehot.scatter_(1, labels.view(-1, 1), 1.0)
            
            # Run & Learn
            synops = run_simulation_and_learn(model, images, labels_onehot, CONFIG)
            total_synops += synops
            
            # Validation
            if batch_idx % CONFIG['val_interval'] == 0:
                val_acc = run_validation(model, val_loader, device, CONFIG)
                elapsed = time.time() - start_time
                print(f"{epoch:<6} | {batch_idx:<6} | {val_acc:<10.2f} | {total_synops/1e9:<15.2f} | {elapsed:<10.2f}")
                
                if val_acc >= CONFIG['target_accuracy']:
                    print("-" * 90)
                    print("Target Accuracy Reached!")
                    break
        else:
            continue
        break

    # Final Test
    print("\nRunning Final Test...")
    test_acc = run_validation(model, test_loader, device, CONFIG)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Total Training SynOps: {total_synops/1e9:.2f} G-SynOps")

if __name__ == "__main__":
    main()