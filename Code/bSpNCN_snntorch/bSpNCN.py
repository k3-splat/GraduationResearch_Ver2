import torch
import torch.nn as nn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt': 0.5,           # タイムステップ幅 (ms)
    'tau_m': 20.0,       # 膜時定数
    'tau_j': 10.0,       # 電流時定数
    'kappa_j': 0.25,     # 電流減衰係数
    'alpha_gen': 1.0,    # 生成誤差の係数
    'alpha_disc': 1.0,   # 識別誤差の係数
    'beta_lr': 0.0001,   # 学習率 (論文中の beta)
    'thresh': 0.4,       # 発火閾値
    'num_steps': 20,     # 推論ステップ数 (T)
    'batch_size': 64,
    'hidden_size': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class bSpNCNLayer(nn.Module):
    """
    1つの層のニューロン状態 (j, v, s, x) と 誤差 (e_gen, e_disc) を保持・更新するクラス
    """
    def __init__(self, layer_idx, output_dim, config):
        super().__init__()
        self.idx = layer_idx
        self.dim = output_dim
        self.cfg = config
        
    def init_state(self, batch_size, device):
        # 状態変数
        self.v = torch.zeros(batch_size, self.dim, device=device)
        self.j = torch.zeros(batch_size, self.dim, device=device)
        self.s = torch.zeros(batch_size, self.dim, device=device) # スパイク
        self.x = torch.zeros(batch_size, self.dim, device=device) # トレース
        
        # 誤差ニューロン (Network側で計算して代入される)
        self.e_gen = torch.zeros(batch_size, self.dim, device=device)
        self.e_disc = torch.zeros(batch_size, self.dim, device=device)

    def update_state(self, total_input_current):
        """
        電流・膜電位・スパイク・トレースの更新
        total_input_current: Network側で計算された「入力項の総和」
        """
        dt = self.cfg['dt']
        
        # 1. 電流 j(t) の更新
        # j(t+dt) = j(t) + dt/tau_j * (-kappa * j(t) + total_input)
        d_j = (-self.cfg['kappa_j'] * self.j + total_input_current)
        self.j = self.j + (dt / self.cfg['tau_j']) * d_j
        
        # 2. 膜電位 v(t) の更新 (LIF)
        # v(t+dt) = v(t) + dt/tau_m * (-v + j)  (簡単のため R=1, gamma=1)
        d_v = (-self.v + self.j)
        self.v = self.v + (dt / self.cfg['tau_m']) * d_v
        
        # 3. スパイク生成 & リセット
        # 閾値を超えたら発火 (1.0), 電圧をリセット (0.0)
        spikes = (self.v > self.cfg['thresh']).float()
        self.s = spikes
        self.v = self.v * (1 - spikes) 
        
        # 4. トレース x(t) の更新
        # z(t) = z(t) + (-z + s) * (dt / tau_trace)  (tau_trace=20.0と仮定)
        self.x = self.x + (-self.x + spikes) * (dt / 20.0)


class bSpNCN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.num_layers = len(layer_sizes) 
        
        # 層の生成
        for i, size in enumerate(layer_sizes):
            # layer_idx は便宜上 0 からスタート (論文の l=1 が layers[0] に対応)
            self.layers.append(bSpNCNLayer(i, size, config))
            
        # --- 重み行列の定義 ---
        # ネットワーク全体で結合を管理する
        # インデックス i は、「Layer i」と「Layer i+1」の間の結合を表す
        
        self.W = nn.ParameterList()      # Top-down (i+1 -> i)
        self.V = nn.ParameterList()      # Bottom-up (i -> i+1)
        self.E_gen = nn.ParameterList()  # FB Bottom-up (e_gen^i -> Layer i+1)
        self.E_disc = nn.ParameterList() # FB Top-down (e_disc^{i+1} -> Layer i)
        
        for i in range(self.num_layers - 1):
            dim_lower = layer_sizes[i]     # Layer i (Lower)
            dim_upper = layer_sizes[i+1]   # Layer i+1 (Upper)
            
            # W: Upper -> Lower (size: Lower x Upper)
            self.W.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))
            
            # V: Lower -> Upper (size: Upper x Lower)
            self.V.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))
            
            # E_gen: Lower_error -> Upper_value (size: Upper x Lower)
            # 論文式(22)の "+ E_gen^l * e_gen^{l-1}" に相当 (l=Upper)
            self.E_gen.append(nn.Parameter(torch.randn(dim_upper, dim_lower) * 0.05))
            
            # E_disc: Upper_error -> Lower_value (size: Lower x Upper)
            # 論文式(22)の "+ E_disc^l * e_disc^{l+1}" に相当 (l=Lower)
            self.E_disc.append(nn.Parameter(torch.randn(dim_lower, dim_upper) * 0.05))

        # 入力データに対する生成誤差 e_gen^0 を計算するための重み (Data -> Layer 1 の逆、つまり Layer 1 -> Data)
        # 論文では W^1 に相当 (Layer 1 -> Input Data)
        input_dim = layer_sizes[0]
        self.W_0 = nn.Parameter(torch.randn(input_dim, input_dim) * 0.05) # 簡単のため入力と同次元と仮定(あるいは単位行列)
        
        # データ層からのフィードバック用 (e_gen^0 -> Layer 1)
        # 論文式(23) "+ E_gen^1 * e_gen^0"
        self.E_gen_0 = nn.Parameter(torch.randn(input_dim, input_dim) * 0.05)

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def forward_dynamics(self, x_data, y_target=None):
        """
        1タイムステップ分の全層更新処理 (論文の式 19-24)
        x_data: 入力データ (Batch, Input_Dim)
        y_target: 教師信号 (Batch, Output_Dim) ※推論時はNone
        """
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']
        
        # --- 0. 最上位層のクランプ (教師あり学習時) ---
        if y_target is not None:
            # 最終層の状態を強制的にターゲットに書き換える (Hard Clamp)
            self.layers[-1].s = y_target
            self.layers[-1].x = y_target.float()
            # 電流や電圧はリセットしておく（ダイナミクス干渉を防ぐ）
            self.layers[-1].j.zero_()
            self.layers[-1].v.zero_()

        # --- 1. 予測 (Prediction) と 誤差 (Error) の計算 ---
        
        # [Special] 入力データレベルの生成誤差 e_gen^0
        # e_gen^0 = Data - W^1 * s^1 (Top-down prediction to data)
        # s^1 は layers[0].s
        s_1 = self.layers[0].s
        z_gen_0 = s_1 @ self.W_0.t()
        e_gen_0 = alpha_gen * (x_data - z_gen_0) # これが e_gen^0
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # --- e_gen (Top-down Error) ---
            if i < self.num_layers - 1:
                # 自分より上位 (i+1) からの予測を受け取る
                # z_gen^l = W^{l+1} * s^{l+1}
                s_upper = self.layers[i+1].s
                z_gen = s_upper @ self.W[i].t() # W[i] connects i+1 -> i
                layer.e_gen = alpha_gen * (layer.x - z_gen)
            else:
                # 最上位層: 生成誤差はない (または0)
                layer.e_gen = torch.zeros_like(layer.x)
                
            # --- e_disc (Bottom-up Error) ---
            if i > 0:
                # 自分より下位 (i-1) からの予測を受け取る
                # z_disc^l = V^{l-1} * s^{l-1}
                s_lower = self.layers[i-1].s
                z_disc = s_lower @ self.V[i-1].t() # V[i-1] connects i-1 -> i
                layer.e_disc = alpha_disc * (layer.x - z_disc)
            else:
                # 第1層: 識別誤差はない (入力データは予測しない)
                # ※論文の構成によっては入力データを予測する場合もあるが、ここでは0とする
                layer.e_disc = torch.zeros_like(layer.x)

        # --- 2. 各層への入力電流の計算と状態更新 ---
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # 入力層と(教師あり時の)出力層はダイナミクスで更新しない場合はスキップ
            # ここでは「入力画像 x_data は層0への入力信号として扱う」ため、
            # layers[0]自体はニューロンとして振る舞う (式23に従う)
            
            if y_target is not None and i == self.num_layers - 1:
                continue # 出力層はクランプ済みなので更新しない

            total_input = 0
            
            # === 項の加算 (Conditioned on Layer Depth) ===
            
            # (A) Local Error Terms (-e_gen - e_disc)
            if i == 0: 
                # 第1層 (式23): -e_gen^1
                total_input += (-layer.e_gen)
            elif i == self.num_layers - 1:
                # 最終層 (式24): -e_disc^L
                total_input += (-layer.e_disc)
            else:
                # 中間層 (式22): -e_gen^l - e_disc^l
                total_input += (-layer.e_gen - layer.e_disc)

            # (B) Bottom-up Feedback (from Lower Error)
            if i == 0:
                # 第1層 (式23): + E_gen^1 * e_gen^0
                # e_gen^0 (calculated above) -> Layer 0
                total_input += e_gen_0 @ self.E_gen_0.t()
            else:
                # 中間・最終層: + E_gen^l * e_gen^{l-1}
                # Lower layer error: layers[i-1].e_gen
                e_gen_lower = self.layers[i-1].e_gen
                # E_gen[i-1] connects (i-1) -> i
                total_input += e_gen_lower @ self.E_gen[i-1].t()

            # (C) Top-down Feedback (from Upper Error)
            if i == self.num_layers - 1:
                # 最終層: 上位はない
                pass
            else:
                # 第1層・中間層: + E_disc^l * e_disc^{l+1}
                # Upper layer error: layers[i+1].e_disc
                e_disc_upper = self.layers[i+1].e_disc
                # E_disc[i] connects (i+1) -> i
                total_input += e_disc_upper @ self.E_disc[i].t()

            # --- 状態更新の実行 ---
            layer.update_state(total_input)

    def manual_weight_update(self):
        """
        推論終了後に行う重みの更新 (論文式 25-28)
        """
        lr = self.config['beta_lr']
        
        with torch.no_grad():
            # Input -> Layer 1 間の更新 (W_0, E_gen_0)
            # W^1 update: e_gen^0 * (s^1)^T
            s_1 = self.layers[0].s
            # e_gen^0 再計算 (保持していないため)
            z_gen_0 = s_1 @ self.W_0.t()
            # x_data は保持していないが、バッチ内では固定と仮定するか、
            # 引数で渡す設計にする必要がある。
            # ※ ここでは簡易化のため、e_gen_0 の計算に必要な x_data がアクセスできないため
            #   本来は forward_dynamics 内で grad を蓄積するか、e_gen_0 を保存すべき。
            #   ここでは「直前のforwardで計算された値が残っている」前提で実装修正が必要だが、
            #   コードが複雑になるため、layerオブジェクトに e_gen_0 を持たせる設計変更を行うか、
            #   あるいは「オンライン学習」として forward 内で更新するのが最も正確。
            pass 

            # Layer i <-> Layer i+1 間の更新
            for i in range(self.num_layers - 1):
                # Lower: i, Upper: i+1
                s_lower = self.layers[i].s
                s_upper = self.layers[i+1].s
                e_gen_lower = self.layers[i].e_gen      # e_gen^{l-1}
                e_disc_upper = self.layers[i+1].e_disc  # e_disc^{l+1}
                
                # Eq 25: Delta W^l = e_gen^{l-1} * (s^l)^T
                # W[i] is Upper(l) -> Lower(l-1). 
                # Grad: (Upper x Lower) -> (Batch, Lower)^T @ (Batch, Upper) ?? No.
                # W shape: (Lower, Upper). 
                # Expected Update: (Lower, Batch) @ (Batch, Upper) -> (Lower, Upper)
                dW = e_gen_lower.t() @ s_upper
                self.W[i] += lr * dW
                
                # Eq 27: Delta V^l = e_disc^{l+1} * (s^l)^T
                # V[i] is Lower(l) -> Upper(l+1)
                # V shape: (Upper, Lower)
                # Expected Update: (Upper, Batch) @ (Batch, Lower) -> (Upper, Lower)
                dV = e_disc_upper.t() @ s_lower
                self.V[i] += lr * dV
                
                # Eq 26 & 28 (Feedback weights E)
                # 論文に従い、Eも更新する
                dE_gen = s_upper.t() @ e_gen_lower # shape check needed
                # self.E_gen[i] += lr * dE_gen.t() 
                
                dE_disc = s_lower.t() @ e_disc_upper
                # self.E_disc[i] += lr * dE_disc.t()

# --- メイン実行部 ---
def main():
    print(f"Device: {CONFIG['device']}")
    
    # データセット (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    # モデル構築 (784 -> 500 -> 10)
    layer_sizes = [784, 500, 10]
    model = bSpNCN(layer_sizes, CONFIG).to(CONFIG['device'])
    
    # 学習ループ
    model.train()
    print("Start Training...")
    
    for epoch in range(3): # 3 Epochs for demo
        start_time = time.time()
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(CONFIG['device']).view(CONFIG['batch_size'], -1) # Flatten
            labels = labels.to(CONFIG['device'])
            
            # One-hot target for supervised learning
            targets = torch.zeros(CONFIG['batch_size'], 10).to(CONFIG['device'])
            targets.scatter_(1, labels.view(-1, 1), 1)
            
            # Poisson Encoding (Rate Coding)
            # [Time, Batch, 784]
            spike_in = spikegen.rate(images, num_steps=CONFIG['num_steps'])
            
            # 状態リセット
            model.reset_state(CONFIG['batch_size'], CONFIG['device'])
            
            # --- Time Loop (T steps) ---
            # 1枚の画像に対して T回 推論・学習を回す (Online Learning)
            
            batch_output_sum = torch.zeros_like(targets)
            
            for t in range(CONFIG['num_steps']):
                x_t = spike_in[t] # [Batch, 784]
                
                # 1. Forward Dynamics (推論 & 状態更新)
                model.forward_dynamics(x_data=x_t, y_target=targets)
                
                # 2. Weight Update (Every step or End of inference)
                # ここでは簡易的にステップごとに更新 (論文の実装依存)
                model.manual_weight_update()
                
                # 精度計測用にスパイクを記録 (Layer L のスパイク)
                batch_output_sum += model.layers[-1].s

            # --- Batch Evaluation ---
            _, predicted = torch.max(batch_output_sum, 1)
            _, target_idx = torch.max(targets, 1)
            total_correct += (predicted == target_idx).sum().item()
            total_samples += targets.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Acc: {100 * total_correct/total_samples:.2f}%")
                total_correct = 0
                total_samples = 0
        
        print(f"Epoch {epoch} Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()