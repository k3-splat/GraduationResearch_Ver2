import torch
import torch.nn as nn
import snntorch.spikegen as spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- ハイパーパラメータ設定 ---
CONFIG = {
    'dt': 0.5, # タイムステップ幅 (ms)
    'tau_m': 20.0, # 膜時定数
    'tau_j': 10.0, # 電流時定数
    'kappa_j': 0.25, # 電流減衰係数
    'alpha_gen': 1.0, 
    'alpha_disc': 1.0,
    'beta_lr': 0.001, # 学習率 (beta in paper)
    'thresh': 0.4,
    'num_steps': 20, # T (推論ステップ数)
    'batch_size': 32,
    'hidden_size': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class bSpNCNLayer(nn.Module):
    """
    bSpNCNの1つの層を定義するクラス
    Value Neurons (j, v, s) と Error Neurons (e_gen, e_disc) を内包する
    """
    def __init__(self, layer_idx, input_dim, output_dim, config):
        super().__init__()
        self.idx = layer_idx
        self.input_dim = input_dim   # 下位層のニューロン数 (前層のoutput_dim)
        self.output_dim = output_dim # この層のニューロン数
        self.cfg = config
        
        # --- パラメータ (重み) ---
        # W: Top-down Prediction (l+1 -> l)
        # V: Bottom-up Prediction (l-1 -> l)
        # E_gen: Error Feedback (gen)
        # E_disc: Error Feedback (disc)
        
        # ※ 実装上の注意: 行列積の形状を合わせるため，重みの形状は (out, in) とする
        # W^{l}: 上位(l+1)から自分(l)への予測. shape: (this_dim, upper_dim)
        # しかし，bPCの論文の定義では W_{l+1} * f(x_{l+1}) となっていたりするため，
        # ここでは「自分(l)への入力」として定義します．
        
        # Bottom-up V: (l-1) -> l. 入力次元は input_dim, 出力次元は output_dim
        self.V = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)
        
        # Top-down W: (l+1) -> l. ※最上位層や次層の定義が必要なため，
        # Networkクラスで層間の結合を管理する方が楽ですが，
        # ここでは「自層から下位層への予測 W」と「自層から上位層への予測 V」を保持する形ではなく
        # 「自層に入ってくる結合」を保持する形にします．
        
        # 簡易化のため，Network全体で重みを管理する設計にします．
        # ここでは状態変数のみ保持します．

    def init_state(self, batch_size, device):
        # Value Neuron States
        self.v = torch.zeros(batch_size, self.output_dim, device=device)
        self.j = torch.zeros(batch_size, self.output_dim, device=device)
        self.s = torch.zeros(batch_size, self.output_dim, device=device) # spike
        self.x = torch.zeros(batch_size, self.output_dim, device=device) # trace z(t)

        # Error Neuron States
        self.e_gen = torch.zeros(batch_size, self.output_dim, device=device)
        self.e_disc = torch.zeros(batch_size, self.output_dim, device=device)
        
        # 膜電位リセット用
        self.refrac_count = torch.zeros(batch_size, self.output_dim, device=device)

class bSpNCN(nn.Module):
    def __init__(self, layer_sizes, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # 層の構築
        for i in range(self.num_layers):
            dim = layer_sizes[i]
            prev_dim = layer_sizes[i-1] if i > 0 else 0 # 入力層は特別扱い
            self.layers.append(bSpNCNLayer(i, prev_dim, dim, config))
            
        # --- 重み行列の定義 (層間) ---
        # W[i]: Layer i+1 から Layer i への Top-down 予測重み (shape: size[i] x size[i+1])
        # V[i]: Layer i から Layer i+1 への Bottom-up 予測重み (shape: size[i+1] x size[i])
        # E_gen[i]: Layer i から Layer i+1 への Error Feedback (shape: size[i+1] x size[i])
        # E_disc[i]: Layer i+1 から Layer i への Error Feedback (shape: size[i] x size[i+1])
        
        self.W = nn.ParameterList()
        self.V = nn.ParameterList()
        self.E_gen = nn.ParameterList()
        self.E_disc = nn.ParameterList()
        
        for i in range(self.num_layers - 1):
            # i と i+1 の間の結合
            dim_lower = layer_sizes[i]
            dim_upper = layer_sizes[i+1]
            
            # W: Upper -> Lower
            self.W.append(nn.Parameter(torch.rand(dim_lower, dim_upper) * 0.01))
            # V: Lower -> Upper
            self.V.append(nn.Parameter(torch.rand(dim_upper, dim_lower) * 0.01))
            # E_gen: Lower_err -> Upper_val (Feedback) - 式(19)参照: E_gen * e_gen^{l-1}
            # 文脈依存ですが，bSpNCN資料の式(19)のE_gen^l * e_gen^{l-1}を見ると
            # E_gen^l は l-1層の誤差を l層に送る重みです．
            self.E_gen.append(nn.Parameter(torch.rand(dim_upper, dim_lower) * 0.01)) # l-1 -> l
            # E_disc: l+1 -> l
            self.E_disc.append(nn.Parameter(torch.rand(dim_lower, dim_upper) * 0.01))

    def reset_state(self, batch_size, device):
        for layer in self.layers:
            layer.init_state(batch_size, device)

    def forward_dynamics(self, x_input, y_target=None):
        """
        1ステップ分のダイナミクス (Eq 19-24)
        x_input: [Batch, InputDim] (スパイク入力)
        y_target: [Batch, OutputDim] (教師信号, 推論時はNone)
        """
        dt = self.config['dt']
        tau_j = self.config['tau_j']
        kappa_j = self.config['kappa_j']
        alpha_gen = self.config['alpha_gen']
        alpha_disc = self.config['alpha_disc']
        
        # 1. 予測 (Prediction) と 誤差 (Error) の計算
        # Layer 0 (Input) は x_input に固定
        self.layers[0].s = x_input
        self.layers[0].x = x_input.float() # Input trace is simple
        
        # 最上位層の固定 (教師あり学習時)
        if y_target is not None:
            self.layers[-1].s = y_target
            self.layers[-1].x = y_target.float()
        
        # 各層の予測誤差計算
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # z_gen (Top-down prediction from i+1)
            if i < self.num_layers - 1:
                # s^{l+1}
                s_upper = self.layers[i+1].s.float()
                # W^{i} * s^{i+1} (i番目のWは i+1 -> i)
                # matmul: (Batch, Upper) @ (Upper, Lower)^T
                z_gen = s_upper @ self.W[i].t() 
                layer.e_gen = alpha_gen * (layer.x - z_gen)
            else:
                layer.e_gen = torch.zeros_like(layer.x) # 最上位層にTop-down予測はない
                
            # z_disc (Bottom-up prediction from i-1)
            if i > 0:
                s_lower = self.layers[i-1].s.float()
                # V^{i-1} * s^{i-1} (i-1番目のVは i-1 -> i)
                z_disc = s_lower @ self.V[i-1].t()
                layer.e_disc = alpha_disc * (layer.x - z_disc)
            else:
                layer.e_disc = torch.zeros_like(layer.x) # 入力層の識別誤差は0 (または入力画像との差)

        # 2. 入力電流 j(t) と 膜電位 v(t) の更新 (Eq 22-24)
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # 入力層と(教師あり時の)出力層はダイナミクスで更新しない（固定）
            if i == 0: continue
            if y_target is not None and i == self.num_layers - 1: continue

            # 項の計算
            # - e_gen^l - e_disc^l
            term_local = - layer.e_gen - layer.e_disc
            
            # + E_gen^l * e_gen^{l-1} (下位層からの生成誤差フィードバック)
            term_fb_gen = 0
            if i > 0:
                # E_gen[i-1] は i-1 -> i
                e_gen_lower = self.layers[i-1].e_gen
                term_fb_gen = e_gen_lower @ self.E_gen[i-1].t()
                
            # + E_disc^l * e_disc^{l+1} (上位層からの識別誤差フィードバック)
            term_fb_disc = 0
            if i < self.num_layers - 1:
                # E_disc[i] は i+1 -> i
                e_disc_upper = self.layers[i+1].e_disc
                term_fb_disc = e_disc_upper @ self.E_disc[i].t()
                
            total_input = term_local + term_fb_gen + term_fb_disc
            
            # j(t) update
            d_j = (-kappa_j * layer.j + total_input)
            layer.j = layer.j + (dt / tau_j) * d_j
            
            # v(t) update (LIF)
            # v(t+dt) = v(t) + dt/tau_m * (-gamma*v + R*j)
            d_v = (-1.0 * layer.v + 1.0 * layer.j) # gamma=1, R=1 simplified
            layer.v = layer.v + (dt / self.config['tau_m']) * d_v
            
            # Spike generation & Reset
            # 不応期などは簡易化して省略していますが，必要なら追加
            spikes = (layer.v > self.config['thresh']).float()
            layer.s = spikes
            layer.v = layer.v * (1 - spikes) # Reset to 0
            
            # Trace update (Eq 6)
            # z(t) = z(t) + (-z/tau + s)
            # 時定数 tau_tr が必要
            layer.x = layer.x + (-layer.x / 20.0 + spikes) # tau_tr assumed 20

    def manual_weight_update(self):
        """
        Eq 25-28 に基づく局所学習則
        """
        lr = self.config['beta_lr']
        
        with torch.no_grad():
            for i in range(self.num_layers - 1):
                # i と i+1 の間の重み更新
                
                # Layer i (Lower) and Layer i+1 (Upper)
                s_upper = self.layers[i+1].s
                s_lower = self.layers[i].s
                e_gen_lower = self.layers[i].e_gen  # e_gen^{l-1} (l=i+1のとき, 下位はi)
                e_disc_upper = self.layers[i+1].e_disc # e_disc^{l+1}
                
                # Eq 25: Delta W^l = e_gen^{l-1} * (s^l)^T
                # W[i] は Upper -> Lower なので l=Upper. 
                # 式: e_gen^{i} * (s^{i+1})^T
                # shapes: (Batch, Lower) * (Batch, Upper) -> (Lower, Upper)
                dW = e_gen_lower.t() @ s_upper
                self.W[i] += lr * dW
                
                # Eq 26: Delta E_gen
                dE_gen = s_upper.t() @ e_gen_lower # 転置関係注意 (実装に合わせて調整要)
                # self.E_gen[i] += lr * dE_gen # 寸法チェック必要
                
                # Eq 27: Delta V^l = e_disc^{l+1} * (s^l)^T
                # V[i] は Lower -> Upper. 
                # 式: e_disc^{i+1} * (s^{i})^T
                dV = e_disc_upper.t() @ s_lower
                self.V[i] += lr * dV
                
                # Eq 28: Delta E_disc
                # ...同様に実装

# --- 学習ループ例 ---
def train_bspncn():
    # データセット読み込み (省略: snntorch codeと同様)
    # ...
    
    layer_sizes = [784, 500, 500, 500, 10]
    model = bSpNCN(layer_sizes, CONFIG).to(CONFIG['device'])
    
    # Optimizerは使わない (手動更新)
    
    for epoch in range(CONFIG['max_epochs']):
        for images, labels in train_loader:
            images = images.to(CONFIG['device'])
            # One-hot encoding for labels
            targets = torch.zeros(images.size(0), 10).to(CONFIG['device'])
            targets.scatter_(1, labels.view(-1, 1), 1)
            
            # Poisson Encoding
            spike_in = spikegen.rate(images.view(images.size(0), -1), num_steps=CONFIG['num_steps'])
            
            # Reset State
            model.reset_state(images.size(0), CONFIG['device'])
            
            # Time Loop (T steps)
            for t in range(CONFIG['num_steps']):
                x_t = spike_in[t]
                
                # 推論 & 状態更新
                model.forward_dynamics(x_t, y_target=targets)
                
                # 重み更新 (Online Learningならここ，Batchならループ後)
                model.manual_weight_update()
                
            print("Batch Done")