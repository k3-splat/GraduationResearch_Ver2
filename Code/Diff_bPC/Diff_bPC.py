"""
=============================================================================
DiffPC (Differentiable/Spiking Predictive Coding) - Dual Pathway & Cost Tracking
=============================================================================

【このプログラムを実行すると何ができるか】
このスクリプトは，提案手法となる双方向予測符号化ニューラルネットワークである
Diff-bPCモデルを用いて，MNIST（手書き数字）の学習・画像生成・コスト評価を行うプログラムである．

プログラムを実行すると，以下のプロセスが順に実行される．

1. 双方向経路を持つモデルの学習（Phase 1 & 2）
   - アクティビティ（s_A），識別誤差（s_e_disc），生成誤差（s_e_gen）のそれぞれに対して
     独立したスパイク発火閾値（lt_min）を設定し，学習を行う．
   - 順伝播用の重み（W）と逆伝播用の重み（V）を別々に更新する．

2. テストセット平均画像に基づくRMSE評価と画像生成
   - ネットワークの出力層に「0〜9」の正解ラベルを与え，トップダウンで画像を生成する．
   - 単に画像を保存するだけでなく，実際のテストデータから計算した「各数字の平均画像」と
     生成画像を比較し，RMSE（平均平方二乗誤差）を算出することで，生成品質の定量的な評価を実施．

3. 層別誤差推移のトラッキングと可視化
   - 学習中のバッチごとに，各層の識別誤差（Discriminative Error）と生成誤差（Generative Error）を
     記録し，学習完了時に推移グラフとして出力する．

4. ハードウェア計算コストのトータルレポート出力
   - 学習全体を通して消費された密行列（Dense）および疎行列（Sparse）の計算コスト
     （FLOPs，データ転送量，通信量）を積算し，学習完了時のコンソールに出力する．

【出力されるファイル】
- runs/[run_name].json : 設定値，各エポックの精度（Train/Test），RMSE，および最終的な総計算コスト
- gen_results/epoch_XXX_[日時]_rmse_[値].png : トップダウン推論で生成された0〜9の画像
- plots/batch_error_curves_[日時].png : バッチ進行に伴う各層の識別誤差・生成誤差の推移グラフ
=============================================================================
"""

import math
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt

# =========================
#  Config & Stats
# =========================

@dataclass
class DiffPCConfig:
    """Central configuration for the DiffPC model and training."""
    layer_dims: list[int] = (784, 200, 10)
    lt_m: int = 3; lt_n: int = 4; lt_a: float = 0.5
    
    # --- MODIFICATION: Separate lt_min for different spike types ---
    lt_min_a: float = 0.0       # Minimum threshold for activity spikes (s_A)
    lt_min_e_disc: float = 0.0  # Minimum threshold for discriminative error spikes (s_e_disc)
    lt_min_e_gen: float = 0.0   # Minimum threshold for generative error spikes (s_e_gen)
    # -------------------------------------------------------------

    lt_scheduler_type: str = "cyclic_phase"   # "cyclic_phase" or "constant"
    gamma_value: float = 0.2; gamma_every_n: Optional[int] = None
    t_init_cycles: int = 15; phase2_cycles: int = 15
    alpha_disc: float = 1.0; alpha_gen: float = 0.01
    pc_lr: float = 5e-4
    batch_size: int = 256; epochs: int = 125
    use_adamw: bool = True
    adamw_weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    clip_grad_norm: float = 1.0 # set <=0 to disable
    seed: int = 42
    run_name: Optional[str] = None # If None, will be timestamped
    use_fashion_mnist: bool = False
    dropout_rate: float = 0.0               # unified dropout rate
    v1_dropout: bool = False                # choose v1 fixed-mask or v2 nn.Dropout
    random_crop_padding: int = 0
    normalize: bool = True                  # toggle Normalize(mean,std)
    fmnist_hflip_p: float = 0.0             # NEW: optional horizontal flip prob for FMNIST train
    device: str = "cuda"

@dataclass
class SpikeStats:
    """Container for spike counts returned by runners."""
    sa_total: float = 0.0
    se_disc_total: float = 0.0
    se_gen_total: float = 0.0

@dataclass
class CostStats:
    """Container for computational cost metrics (Total sums)."""
    dense_flops: float = 0.0            # Total FLOPs (assuming dense computation)
    sparse_flops: float = 0.0           # Total FLOPs (skipping zero spikes)
    dense_data_transfer: float = 0.0    # Total Data Transfer in Bytes (Dense)
    sparse_data_transfer: float = 0.0   # Total Data Transfer in Bytes (Sparse)
    dense_comm: float = 0.0             # Inter-layer Communication in Bytes (Dense)
    sparse_comm: float = 0.0            # Inter-layer Communication in Bytes (Sparse)

    def __add__(self, other):
        return CostStats(
            self.dense_flops + other.dense_flops,
            self.sparse_flops + other.sparse_flops,
            self.dense_data_transfer + other.dense_data_transfer,
            self.sparse_data_transfer + other.sparse_data_transfer,
            self.dense_comm + other.dense_comm,
            self.sparse_comm + other.sparse_comm
        )

# =========================
#  Schedulers + factories
# =========================

def _mod_idx(k: int, period: int) -> int:
    if period <= 0: return 0
    return (k % period + period) % period

class LTBaseScheduler:
    def get_l_t(self, sample_step: int, train: bool) -> float: raise NotImplementedError

@dataclass
class LTCyclicPhase(LTBaseScheduler):
    m: int; n: int; a: float = 1.0
    _phase_start_cycle: int = 0; _phase_cycles: int = 1; _phase_a: float = 1.0

    def begin_phase(self, phase_start_step: int, phase_len: int, a: Optional[float] = None):
        assert self.n > 0, "lt_n must be positive for cycle-based decay."
        assert phase_len % self.n == 0, "phase_len must be a multiple of lt_n (full cycles)"
        self._phase_start_cycle = phase_start_step // self.n
        self._phase_cycles = max(1, phase_len // self.n)
        self._phase_a = float(self.a if a is None else a)

    def _decay_mult(self, sample_step: int) -> float:
        current_cycle_in_phase = (sample_step // self.n) - self._phase_start_cycle
        cycle_idx = min(max(current_cycle_in_phase, 0), self._phase_cycles - 1)
        r = 0.0 if self._phase_cycles <= 1 else cycle_idx / (self._phase_cycles - 1)
        return 1.0 - (1.0 - self._phase_a) * r

    def get_l_t(self, sample_step: int, train: bool) -> float:
        k = _mod_idx(sample_step, self.n)
        base = (2.0 ** self.m) / (2.0 ** k)
        return base * self._decay_mult(sample_step)

@dataclass
class LTConstant(LTBaseScheduler):
    m: int; n: int; a: float = 1.0
    _phase_start_cycle: int = 0; _phase_cycles: int = 1; _phase_a: float = 1.0

    def begin_phase(self, phase_start_step: int, phase_len: int, a: Optional[float] = None):
        assert self.n > 0, "lt_n must be positive for cycle-based scheduling."
        assert phase_len % self.n == 0, "phase_len must be a multiple of lt_n (full cycles)"
        self._phase_start_cycle = phase_start_step // self.n
        self._phase_cycles = max(1, phase_len // self.n)
        self._phase_a = float(self.a if a is None else a)

    def _decay_mult(self, sample_step: int) -> float:
        current_cycle_in_phase = (sample_step // self.n) - self._phase_start_cycle
        cycle_idx = min(max(current_cycle_in_phase, 0), self._phase_cycles - 1)
        r = 0.0 if self._phase_cycles <= 1 else cycle_idx / (self._phase_cycles - 1)
        return 1.0 - (1.0 - self._phase_a) * r

    def get_l_t(self, sample_step: int, train: bool) -> float:
        base = 2.0 ** self.m
        return base * self._decay_mult(sample_step)

def get_l_t_scheduler(scheduler_type: str, scheduler_args: dict) -> LTBaseScheduler:
    if scheduler_type == "cyclic_phase": return LTCyclicPhase(**scheduler_args)
    if scheduler_type == "constant":     return LTConstant(**scheduler_args)
    raise ValueError(f"Unknown l_t scheduler type: {scheduler_type}")

class YBaseScheduler:
    def __init__(self, l_t_scheduler: LTBaseScheduler): self.l_t_scheduler = l_t_scheduler
    def get_y(self, sample_step: int, train: bool) -> float: raise NotImplementedError

class YOnCycleStart(YBaseScheduler):
    def __init__(self, l_t_scheduler: LTBaseScheduler, gamma: float, n: Optional[int] = None):
        super().__init__(l_t_scheduler)
        self.gamma = float(gamma)
        self.n = n if n is not None else getattr(l_t_scheduler, "n", None)
        if self.n is None: raise ValueError("YOnCycleStart requires 'n' or l_t_scheduler with 'n'.")

    def get_y(self, sample_step: int, train: bool) -> float:
        return self.gamma if (_mod_idx(sample_step, self.n) == 0) else 0.0

def get_y_scheduler(scheduler_type: str, scheduler_args: dict) -> YBaseScheduler:
    l_t_sched = scheduler_args["l_t_scheduler"]
    args = {k: v for k, v in scheduler_args.items() if k != "l_t_scheduler"}
    if scheduler_type == "on_cycle_start": return YOnCycleStart(l_t_sched, **args)
    raise ValueError(f"Unknown y scheduler type: {scheduler_type}")


# =========================
#  Single layer state machine (batched)
# =========================

class DiffPCLayerTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.device = torch.device(self.device)
        self.x_F_disc, self.x_F_gen, self.x_T, self.x_A, self.e_T_disc, self.e_T_gen, \
        self.e_A_disc, self.e_A_gen, self.s_A, self.s_e_disc, self.s_e_gen, \
        self.s_in_disc, self.s_in_gen, self.e_in_disc, self.e_in_gen, self.data = [None] * 16
        self.l_t, self.y, self.time_step = 0.0, 0.0, 1
        self.reset_state, self.reset_state_type = True, "zero"
        # prev-tick buffers
        self.s_A_prev, self.s_e_disc_prev, self.s_e_gen_prev = None, None, None

        if not hasattr(self, 'lt_min_a'): self.lt_min_a = 0.0
        if not hasattr(self, 'lt_min_e_disc'): self.lt_min_e_disc = 0.0
        if not hasattr(self, 'lt_min_e_gen'): self.lt_min_e_gen = 0.0

    def _alloc(self, B: int):
        z = torch.zeros(B, self.dim, device=self.device)
        self.x_F_disc, self.x_F_gen, self.x_T, self.x_A, self.e_T_disc, self.e_T_gen, \
        self.e_A_disc, self.e_A_gen, self.s_A, self.s_e_disc, self.s_e_gen, \
        self.s_in_disc, self.s_in_gen, self.e_in_disc, self.e_in_gen, self.data = [z.clone() for _ in range(16)]
        self.s_A_prev = torch.zeros_like(z)
        self.s_e_disc_prev = torch.zeros_like(z)
        self.s_e_gen_prev = torch.zeros_like(z)

    def _sample_step(self) -> int: return (self.time_step - 1) % self.sampling_duration

    def reset_states_if_needed(self, clamp_status: bool, sample_step: int):
        if (sample_step == 0) and self.reset_state:
            self.x_F_disc.zero_(); self.x_F_gen.zero_(); self.x_A.zero_()
            self.e_T_disc.zero_(); self.e_T_gen.zero_()
            self.e_A_disc.zero_(); self.e_A_gen.zero_(); self.e_in_disc.zero_(), self.e_in_gen.zero_()
            if not clamp_status: self.x_T.zero_()
            if self.s_A_prev is not None: self.s_A_prev.zero_()
            if self.s_e_disc_prev is not None: self.s_e_disc_prev.zero_()
            if self.s_e_gen_prev is not None: self.s_e_gen_prev.zero_()

    @torch.no_grad()
    def step(self, clamp_status: bool, data_in: Optional[torch.Tensor],
             s_in_disc: torch.Tensor, s_in_gen: torch.Tensor,
             e_in_disc: torch.Tensor, e_in_gen: torch.Tensor,
             bottomup_mask: bool = False, topdown_mask: bool = False, 
             sample_step_override: Optional[int] = None):
        B = s_in_disc.size(0)
        if self.x_F_disc is None or self.x_F_gen is None or \
            self.x_F_disc.size(0) != B or self.x_F_gen.size(0) != B: self._alloc(B)
        
        sample_step = self._sample_step() if sample_step_override is None else sample_step_override
        
        if clamp_status: self.x_T.copy_(data_in.to(self.device))
        self.s_in_disc = s_in_disc.to(self.device)
        self.s_in_gen = s_in_gen.to(self.device)
        self.reset_states_if_needed(clamp_status=clamp_status, sample_step=sample_step)
        
        l_t_cur_base  = self.l_t_scheduler.get_l_t(sample_step,     self.learning_weights)
        l_t_prev_base = self.l_t_scheduler.get_l_t(sample_step - 1, self.learning_weights)
        
        self.l_t = l_t_cur_base
        self.y   = self.y_scheduler.get_y(sample_step, self.learning_weights)

        l_t_prev = max(l_t_prev_base, self.lt_min_a) 
        
        self.x_F_disc.add_(self.s_in_disc * l_t_prev)
        self.x_F_gen.add_(self.s_in_gen * l_t_prev)
        self.e_T_disc = (self.x_T - self.x_F_disc)
        self.e_T_gen = (self.x_T - self.x_F_gen)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration and bottomup_mask:
            self.x_F_disc.zero_()
            self.e_T_disc.zero_()
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration and topdown_mask:
            self.x_F_gen.zero_()
            self.e_T_gen.zero_()

        self.e_in_disc.add_(e_in_disc.to(self.device) * l_t_prev)
        self.e_in_gen.add_(e_in_gen.to(self.device) * l_t_prev)

        if not clamp_status:
            self.x_T.add_(self.y * (-self.alpha_disc * self.e_T_disc - self.alpha_gen * self.e_T_gen + (self.x_T > 0).float() * (self.alpha_disc * self.e_in_disc + self.alpha_gen * self.e_in_gen)))
        
        # --- Variable Thresholds ---
        eff_lt_a = max(l_t_cur_base, self.lt_min_a)
        diff_act = self.x_T - self.x_A
        s_A_new = torch.sign(diff_act) * (diff_act.abs() > eff_lt_a)
        s_A_new = s_A_new * ((self.x_A + s_A_new * eff_lt_a) > 0.0)
        self.x_A.add_(s_A_new * eff_lt_a); self.s_A = s_A_new
        
        eff_lt_disc = max(l_t_cur_base, self.lt_min_e_disc)
        eff_lt_gen  = max(l_t_cur_base, self.lt_min_e_gen)

        diff_disc_err = self.e_T_disc - self.e_A_disc
        diff_gen_err = self.e_T_gen - self.e_A_gen
        
        s_e_disc_new = torch.sign(diff_disc_err) * (diff_disc_err.abs() > eff_lt_disc)
        s_e_gen_new = torch.sign(diff_gen_err) * (diff_gen_err.abs() > eff_lt_gen)
        
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration:
            s_e_disc_new.zero_()
            s_e_gen_new.zero_()

        self.e_A_disc.add_(s_e_disc_new * eff_lt_disc)
        self.e_A_gen.add_(s_e_gen_new * eff_lt_gen)
        self.s_e_disc = s_e_disc_new
        self.s_e_gen = s_e_gen_new

        if self.s_A_prev is None or self.s_A_prev.shape != self.s_A.shape:
            self.s_A_prev = torch.zeros_like(self.s_A)
        if self.s_e_disc_prev is None or self.s_e_disc_prev.shape != self.s_e_disc.shape:
            self.s_e_disc_prev = torch.zeros_like(self.s_e_disc)
        if self.s_e_gen_prev is None or self.s_e_gen_prev.shape != self.s_e_gen.shape:
            self.s_e_gen_prev = torch.zeros_like(self.s_e_gen)
        self.s_A_prev.copy_(self.s_A)
        self.s_e_disc_prev.copy_(self.s_e_disc)
        self.s_e_gen_prev.copy_(self.s_e_gen)

        if sample_step_override is not None:
            self.time_step = int(sample_step_override) + 1
        else:
            self.time_step += 1


# =========================
#  Multi-layer network orchestrator
# =========================

class DiffPCNetworkTorch(nn.Module):
    def __init__(self, cfg: DiffPCConfig, l_t_scheduler_spec: Dict, y_scheduler_spec: Dict, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.layer_dims[i+1], cfg.layer_dims[i], device=self.device))
            for i in range(len(cfg.layer_dims)-1)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.layer_dims[i], cfg.layer_dims[i+1], device=self.device))
            for i in range(len(cfg.layer_dims)-1)
        ])
        self.W_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.cfg.layer_dims[i+1], 1, device=self.device))
            for i in range(len(self.cfg.layer_dims) - 1)
        ])
        self.V_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.cfg.layer_dims[i], 1, device=self.device))
            for i in range(len(self.cfg.layer_dims) - 1)
        ])
        
        for w_list in [self.W, self.V]:
            for w in w_list:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if cfg.use_adamw:
            self.optimizer = torch.optim.AdamW(
                list(self.W) + list(self.W_bias) + list(self.V) + list(self.V_bias),
                lr=cfg.pc_lr, betas=cfg.adamw_betas,
                eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                list(self.W) + list(self.W_bias) + list(self.V) + list(self.V_bias),
                lr=cfg.pc_lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps
            )

        l_t_sched = get_l_t_scheduler(l_t_scheduler_spec["type"], l_t_scheduler_spec["args"])
        y_sched = get_y_scheduler(y_scheduler_spec["type"], {**y_scheduler_spec["args"], "l_t_scheduler": l_t_sched})
        
        layer_args = {
            "sampling_duration": 1, "learning_weights": True, "training": True,
            "l_t_scheduler": l_t_sched, "y_scheduler": y_sched, 
            "alpha_disc": cfg.alpha_disc, "alpha_gen": cfg.alpha_gen,
            "device": self.device,
            "lt_min_a": cfg.lt_min_a,
            "lt_min_e_disc": cfg.lt_min_e_disc,
            "lt_min_e_gen": cfg.lt_min_e_gen
        }

        self.layers = nn.ModuleList([DiffPCLayerTorch(dim=d, **layer_args) for d in cfg.layer_dims[1:]])
        self.input_driver = DiffPCLayerTorch(dim=cfg.layer_dims[0], **layer_args)

        self.input_driver_clamp, self.input_driver_data = False, None
        self._clamp_switch = [False] * len(self.layers)
        self._data_bucket = [None] * len(self.layers)
        
        self._global_step = 0
        self._bias_fired_once = False
        self._layer_dims: List[int] = list(cfg.layer_dims[1:])
        
        # --- NEW: Cost Stats ---
        self.costs = CostStats()
        self.measure_costs = False  # To be enabled during training

    def reset_all_states(self, batch_size: int):
        for lyr in [self.input_driver, *self.layers]:
            lyr._alloc(batch_size); lyr.time_step = 1
        self._global_step = 0
    
    def set_training(self, flag: bool):
        self.train(flag)
        self.measure_costs = flag # Only measure when training
        for lyr in [self.input_driver, *self.layers]:
            lyr.training = bool(flag)

    def swap_schedulers(self, l_t_sched: LTBaseScheduler, y_sched: YBaseScheduler):
        for lyr in [self.input_driver, *self.layers]:
            lyr.l_t_scheduler = l_t_sched; lyr.y_scheduler = y_sched

    def set_sampling_duration(self, new_len: int):
        for lyr in [self.input_driver, *self.layers]: lyr.sampling_duration = new_len
        self._global_step = 0

    def set_clamp(self, layer_index: int, clamp: bool, data: Optional[torch.Tensor] = None):
        if layer_index == 0: self.input_driver_clamp, self.input_driver_data = clamp, data
        else: self._clamp_switch[layer_index-1], self._data_bucket[layer_index-1] = clamp, data

    @torch.no_grad()
    def _accumulate_mm_costs(self, x: torch.Tensor, weight_matrix: torch.Tensor):
        """
        Calculates and accumulates costs for a Matrix Multiplication: Y = X @ W.T
        """
        if not self.measure_costs: return

        # Dimensions
        B, N_in = x.shape
        w_shape = weight_matrix.shape
        if w_shape[1] == N_in:
             N_out = w_shape[0] # Standard Linear layout
        elif w_shape[0] == N_in:
             N_out = w_shape[1] # Transposed usage
        else:
             N_out = w_shape[0] 

        # 1. FLOPs (MACs * 2)
        d_flops = B * N_in * N_out * 2
        
        nnz = torch.count_nonzero(x).item()
        s_flops = nnz * N_out * 2
        
        # 2. Data Transfer (Bytes, float32=4B)
        w_size_bytes = weight_matrix.numel() * 4
        x_size_bytes = x.numel() * 4
        y_size_bytes = B * N_out * 4
        d_mem = w_size_bytes + x_size_bytes + y_size_bytes
        
        # [修正] tanomu.py (Original): s_mem = (nnz * N_out * 4) + (nnz * 8) + y_size_bytes
        # [修正] bottomup一致版: Input読み込みをValueのみ(4B)とする
        s_mem = (nnz * N_out * 4) + (nnz * 4) + y_size_bytes

        # 3. Inter-layer Communication (Bytes)
        d_comm = x_size_bytes

        # [修正] tanomu.py (Original): s_comm = nnz * 8
        # [修正] bottomup一致版: 理論ビット数 (32 + log2(N_in)) をバイトに換算
        import math
        address_bits = math.ceil(math.log2(N_in)) if N_in > 1 else 1
        # ビット数をバイトに変換 ( / 8.0 )
        s_comm = nnz * (4.0 + address_bits / 8.0)

        self.costs.dense_flops += d_flops
        self.costs.sparse_flops += s_flops
        self.costs.dense_data_transfer += d_mem
        self.costs.sparse_data_transfer += s_mem
        self.costs.dense_comm += d_comm
        self.costs.sparse_comm += s_comm


    @torch.no_grad()
    def _build_s_in_and_e_in(self):
        s_in_disc, s_in_gen, e_in_disc, e_in_gen = [], [], [], []

        sd = self.layers[0].sampling_duration if len(self.layers) > 0 else 1
        t  = self._global_step % sd

        emit_bias = (not self._bias_fired_once) and (t == 1)
        if emit_bias:
            bias_spike = 1.0
            self._bias_fired_once = True
        else:
            bias_spike = 0.0

        prev_s_disc = self.input_driver.s_A_prev
        prev_s_gen = self.layers[0].s_A_prev
        
        for l in range(len(self.W)):
            # Forward Disc Spikes: prev_s_disc @ W[l].T
            self._accumulate_mm_costs(prev_s_disc, self.W[l])
            s_in_disc_l = prev_s_disc @ self.W[l].T
            
            # Forward Gen Spikes: prev_s_gen @ V[l].T
            self._accumulate_mm_costs(prev_s_gen, self.V[l])
            s_in_gen_l = prev_s_gen @ self.V[l].T

            if bias_spike != 0.0:
                s_in_disc_l.add_(self.W_bias[l].T)
                s_in_gen_l.add_(self.V_bias[l].T)

            s_in_disc.append(s_in_disc_l)
            prev_s_disc = self.layers[l].s_A_prev
            s_in_gen.append(s_in_gen_l)
            if l < len(self.W) - 1:
                prev_s_gen = self.layers[l+1].s_A_prev

        for l in range(len(self.cfg.layer_dims)):
            if l < len(self.cfg.layer_dims) - 1:
                # Backward Disc Error: s_e_disc_prev @ W[l]
                # self.W[l] shape is (Next, Curr). s_e is (B, Next).
                self._accumulate_mm_costs(self.layers[l].s_e_disc_prev, self.W[l])
                e_in_disc_l = self.layers[l].s_e_disc_prev @ self.W[l]
                
                if l == 0:
                    # Backward Gen Error: s_e_gen_prev @ V[l]
                    self._accumulate_mm_costs(self.input_driver.s_e_gen_prev, self.V[l])
                    e_in_gen_l = self.input_driver.s_e_gen_prev @ self.V[l]
                else:
                    self._accumulate_mm_costs(self.layers[l-1].s_e_gen_prev, self.V[l])
                    e_in_gen_l = self.layers[l-1].s_e_gen_prev @ self.V[l]

            else:
                e_in_disc_l = torch.zeros_like(self.layers[l-1].s_e_disc_prev)

            e_in_disc.append(e_in_disc_l)
            e_in_gen.append(e_in_gen_l)

        return s_in_disc, s_in_gen, e_in_disc, e_in_gen

    @torch.no_grad()
    def one_time_step(self, bottomup_mask: bool = False, topdown_mask : bool = False):
        B  = self.input_driver.x_F_disc.size(0)
        z  = torch.zeros(B, self.input_driver.dim, device=self.device)
        z_top = torch.zeros(B, self.layers[-1].dim, device=self.device)
        sd = self.input_driver.sampling_duration
        t  = self._global_step % sd

        s_in_disc, s_in_gen, e_in_disc, e_in_gen = self._build_s_in_and_e_in()
        s_in_gen.append(z_top)

        self.input_driver.step(self.input_driver_clamp, self.input_driver_data, z, s_in_gen[0],
                               e_in_disc[0], z, False, topdown_mask, sample_step_override=t)
        for i, lyr in enumerate(self.layers):
            lyr.step(self._clamp_switch[i], self._data_bucket[i],
                      s_in_disc[i], s_in_gen[i+1], e_in_disc[i+1], e_in_gen[i],
                      bottomup_mask, topdown_mask, sample_step_override=t)
        
        self._global_step += 1

    @torch.no_grad()
    def apply_phase2_update(self):
        if not self.layers[0].training: return
        Bf = float(self.input_driver.x_T.size(0))
        self.optimizer.zero_grad(set_to_none=True)

        for l, lyr in enumerate(self.layers):
            pre_xT_disc  = self.input_driver.x_T if l == 0 else self.layers[l-1].x_T
            pre_xT_gen  = lyr.x_T
            post_eT_disc = lyr.e_T_disc
            post_eT_gen = self.input_driver.e_T_gen if l == 0 else self.layers[l-1].e_T_gen

            self.W[l].grad = - cfg.alpha_disc * (post_eT_disc.T @ torch.relu(pre_xT_disc)) / Bf
            self.W_bias[l].grad = - cfg.alpha_disc * post_eT_disc.sum(dim=0, keepdim=True).T / Bf
            self.V[l].grad = - (post_eT_gen.T @ torch.relu(pre_xT_gen)) / Bf
            self.V_bias[l].grad = - cfg.alpha_gen * post_eT_gen.sum(dim=0, keepdim=True).T / Bf

        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.W) + list(self.W_bias) + list(self.V) + list(self.V_bias), self.cfg.clip_grad_norm)
        self.optimizer.step()


# =========================
#  Runners
# =========================

@torch.no_grad()
def run_batch_two_phase(net: DiffPCNetworkTorch, x: torch.Tensor, y_onehot: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats, Dict[str, List[float]]]:
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(x.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])

    # Phase-1
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    net.set_training(False)

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    for _ in range(steps_phase1): net.one_time_step(bottomup_mask=False, topdown_mask=True)

    # Phase-2 (continue state)
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)

    net.set_training(True) # This enables cost measurement

    net.set_clamp(len(net.layers), True, y_onehot)
    
    spike_stats = SpikeStats()
    batch_size = x.size(0)

    for _ in range(steps_phase2):
        net.one_time_step(bottomup_mask=False, topdown_mask=False)
        
        # Count Spikes
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_disc_total += (lyr.s_e_disc != 0).sum().item()
            spike_stats.se_gen_total += (lyr.s_e_gen != 0).sum().item()

    batch_error_stats = {
        "input_gen": net.input_driver.e_T_gen.abs().sum().item() / batch_size,
        "layers_disc": [lyr.e_T_disc.abs().sum().item() / batch_size for lyr in net.layers],
        "layers_gen": [lyr.e_T_gen.abs().sum().item() / batch_size for lyr in net.layers]
    }

    net.apply_phase2_update()
    return net.layers[-1].x_T.clone(), spike_stats, batch_error_stats

@torch.no_grad()
def infer_batch_forward_only(net: DiffPCNetworkTorch, x: torch.Tensor, cfg: DiffPCConfig, l_t_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
    lt_n = l_t_spec["args"]["n"]
    steps = cfg.t_init_cycles * lt_n
    net.set_sampling_duration(steps)
    net.reset_all_states(x.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])
    y_forward = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_forward)

    net.set_training(False)

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    
    spike_stats = SpikeStats()
    for _ in range(steps):
        net.one_time_step(bottomup_mask=False, topdown_mask=True)
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_disc_total += (lyr.s_e_disc != 0).sum().item()
            spike_stats.se_gen_total += (lyr.s_e_gen != 0).sum().item()
            
    return net.layers[-1].x_T.clone(), spike_stats


@torch.no_grad()
def infer_batch_backward_phase2(net: DiffPCNetworkTorch, y_onehot: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(y_onehot.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])

    # Phase-1
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    net.set_training(False)

    original_alphas = []
    original_alphas.append(net.input_driver.alpha_gen)
    net.input_driver.alpha_gen = 1.0
    for lyr in net.layers:
        original_alphas.append(lyr.alpha_gen)
        lyr.alpha_gen = 1.0

    net.set_clamp(len(net.layers), True, y_onehot)
    for li in range(len(net.layers)): net.set_clamp(li, False)
    for _ in range(steps_phase1): net.one_time_step(bottomup_mask=True, topdown_mask=False)

    # Phase-2
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)
    
    spike_stats = SpikeStats()
    for _ in range(steps_phase2):
        net.one_time_step(bottomup_mask=False, topdown_mask=False)
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_disc_total += (lyr.s_e_disc != 0).sum().item()
            spike_stats.se_gen_total += (lyr.s_e_gen != 0).sum().item()

    net.input_driver.alpha_gen = original_alphas[0]
    for i, lyr in enumerate(net.layers):
        lyr.alpha_gen = original_alphas[i+1]
    
    return net.input_driver.x_T.clone(), spike_stats


def calculate_test_mean_digits(test_loader: DataLoader, num_classes: int, device: str) -> torch.Tensor:
    """Calculates the average image for each digit class in the test set."""
    print("Pre-calculating mean images for test set...")
    sums = torch.zeros(num_classes, 784, device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        for i in range(num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                sums[i] += images[mask].sum(dim=0)
                counts[i] += mask.sum()
    
    means = sums / counts.view(-1, 1)
    return means


def visualize_generated_digits(net: DiffPCNetworkTorch, cfg: DiffPCConfig, 
                               l_t_spec: dict, y_phase2_spec: dict,
                               epoch: int, mean_test_digits: torch.Tensor, 
                               output_dir: str = "gen_results") -> float:
    """
    Generates digits, saves the image, and calculates RMSE against the mean test digits.
    Returns: Average RMSE.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = net.device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = torch.arange(10, device=device)
    y_onehot = F.one_hot(labels, num_classes=cfg.layer_dims[-1]).float()
    
    generated_data, _ = infer_batch_backward_phase2(net, y_onehot, cfg, l_t_spec, y_phase2_spec)
    
    # Calculate RMSE
    # generated_data: (10, 784), mean_test_digits: (10, 784)
    mse = ((generated_data - mean_test_digits) ** 2).mean(dim=1) # MSE per digit
    rmse_per_digit = torch.sqrt(mse)
    avg_rmse = rmse_per_digit.mean().item()

    gen_imgs = generated_data.view(10, 1, 28, 28)
    if cfg.normalize:
        if cfg.use_fashion_mnist:
            mean, std = 0.2860, 0.3530
        else:
            mean, std = 0.1307, 0.3081
        gen_imgs = gen_imgs * std + mean
    
    gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0)
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_{timestamp}_rmse_{avg_rmse:.4f}.png")
    save_image(gen_imgs, save_path, nrow=10)
    
    return avg_rmse


def plot_batch_error_history(error_history: List[Dict], output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    steps = range(1, len(error_history) + 1)
    input_gen_hist = [h['input_gen'] for h in error_history]
    num_layers = len(error_history[0]['layers_disc'])
    layers_disc_hist = [[h['layers_disc'][i] for h in error_history] for i in range(num_layers)]
    layers_gen_hist = [[h['layers_gen'][i] for h in error_history] for i in range(num_layers)]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for i in range(num_layers):
        plt.plot(steps, layers_disc_hist[i], linewidth=0.5, label=f'Layer {i+1} Disc Err')
    plt.title('Discriminative Error Snapshot')
    plt.xlabel('Global Batch Step'); plt.ylabel('Avg |e_T_disc|')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(steps, input_gen_hist, linewidth=0.5, linestyle='--', label='Input (L0) Gen Err')
    for i in range(num_layers):
        plt.plot(steps, layers_gen_hist[i], linewidth=0.5, label=f'Layer {i+1} Gen Err')
    plt.title('Generative Error Snapshot')
    plt.xlabel('Global Batch Step'); plt.ylabel('Avg |e_T_gen|')
    plt.legend(); plt.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"batch_error_curves_{timestamp}.png")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Full batch error plot saved to: {save_path}")


# =========================
#  Main Loop
# =========================

def main(cfg: DiffPCConfig):
    torch.manual_seed(cfg.seed)
    requested = str(cfg.device).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested)
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")

    l_t_spec = { "type": cfg.lt_scheduler_type, "args": {"m": cfg.lt_m, "n": cfg.lt_n, "a": cfg.lt_a} }
    y_phase2_spec = { "type": "on_cycle_start", "args": {"gamma": cfg.gamma_value, "n": (cfg.gamma_every_n or cfg.lt_n)} }
    
    net = DiffPCNetworkTorch(cfg, l_t_scheduler_spec=l_t_spec, y_scheduler_spec=y_phase2_spec, device=str(device))
    
    if cfg.use_fashion_mnist: mean, std = 0.2860, 0.3530
    else: mean, std = 0.1307, 0.3081

    train_transforms_list = []
    if cfg.random_crop_padding > 0:
        train_transforms_list.append(transforms.RandomCrop(28, padding=cfg.random_crop_padding))
    if cfg.use_fashion_mnist and cfg.fmnist_hflip_p > 0.0:
        train_transforms_list.append(transforms.RandomHorizontalFlip(p=cfg.fmnist_hflip_p))
    train_transforms_list.append(transforms.ToTensor())
    if cfg.normalize: train_transforms_list.append(transforms.Normalize((mean,), (std,)))
    train_transform = transforms.Compose(train_transforms_list)

    test_transforms_list = [transforms.ToTensor()]
    if cfg.normalize: test_transforms_list.append(transforms.Normalize((mean,), (std,)))
    test_transform = transforms.Compose(test_transforms_list)

    if cfg.use_fashion_mnist:
        train_ds = datasets.FashionMNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=test_transform)
    else:
        train_ds = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Pre-calculate Mean Digits for RMSE
    mean_test_digits = calculate_test_mean_digits(test_loader, 10, device)

    def to_vec(x): return x.view(x.size(0), -1).to(device)
    def to_onehot(y): return F.one_hot(y, num_classes=cfg.layer_dims[-1]).float().to(device)

    run_results = []
    global_batch_error_history = [] 

    print(f"Starting run: {run_name}")

    for epoch in range(1, cfg.epochs + 1):
        total_sa, total_se_disc, total_se_gen = 0.0, 0.0, 0.0
        epoch_err = {"input_gen": 0.0, "layers_disc": [0.0]*len(net.layers), "layers_gen": [0.0]*len(net.layers)}
        total_samples = 0

        for images, labels in train_loader:
            _, stats, batch_err_stats = run_batch_two_phase(net, to_vec(images), to_onehot(labels), cfg, l_t_spec, y_phase2_spec)
            
            global_batch_error_history.append(batch_err_stats)
            total_sa += stats.sa_total
            total_se_disc += stats.se_disc_total
            total_se_gen += stats.se_gen_total
            
            bs = images.size(0)
            total_samples += bs
            epoch_err["input_gen"] += batch_err_stats["input_gen"] * bs
            for i in range(len(net.layers)):
                epoch_err["layers_disc"][i] += batch_err_stats["layers_disc"][i] * bs
                epoch_err["layers_gen"][i] += batch_err_stats["layers_gen"][i] * bs

        # Validation
        train_correct, train_total = 0, 0
        test_correct, test_total = 0, 0
        
        with torch.no_grad():
            for images, labels in train_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                train_correct += (logits.argmax(1).cpu() == labels).sum().item()
                train_total += labels.size(0)
            for images, labels in test_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                test_correct += (logits.argmax(1).cpu() == labels).sum().item()
                test_total += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc  = 100.0 * test_correct / test_total
        
        # Generation RMSE
        gen_rmse = visualize_generated_digits(net, cfg, l_t_spec, y_phase2_spec, epoch, mean_test_digits)

        print(f"Epoch {epoch:02d}: Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}% | Gen RMSE {gen_rmse:.4f}")

        run_results.append({
            "epoch": epoch, "train_acc": train_acc, "test_acc": test_acc, "gen_rmse": gen_rmse
        })

    print("Training finished. Saving logs and plots...")
    plot_batch_error_history(global_batch_error_history)

    # Compile Final Costs
    final_costs = asdict(net.costs)
    # Convert to Giga for display
    print("\n=== Computational Cost Report (Total Training) ===")
    print(f"Dense FLOPs:  {final_costs['dense_flops']/1e9:.4f} GFLOPs")
    print(f"Sparse FLOPs: {final_costs['sparse_flops']/1e9:.4f} GFLOPs")
    print(f"Dense Mem:    {final_costs['dense_data_transfer']/1e9:.4f} GB")
    print(f"Sparse Mem:   {final_costs['sparse_data_transfer']/1e9:.4f} GB")
    print(f"Dense Comm:   {final_costs['dense_comm']/1e9:.4f} GB")
    print(f"Sparse Comm:  {final_costs['sparse_comm']/1e9:.4f} GB")
    print("==================================================")

    with open(log_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": run_results, "total_costs": final_costs}, f, indent=4)

if __name__ == "__main__":
    cfg = DiffPCConfig(
        layer_dims=[784, 400, 10],
        lt_m=-4,
        lt_n=6,
        lt_a=1.0,
        lt_min_a=0.0625,
        lt_min_e_disc=0.0625,
        lt_min_e_gen=0.0,
        lt_scheduler_type="cyclic_phase",
        gamma_value=0.05,
        t_init_cycles=20,
        phase2_cycles=30,
        alpha_disc = 1,
        alpha_gen = 0.001,
        pc_lr=0.0001,
        batch_size=256,
        epochs=20,
        use_adamw=True,
        adamw_weight_decay=0.01,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-08,
        clip_grad_norm=1.0,
        seed=2,
        run_name="mnist_2026_02_23_part2",
        use_fashion_mnist=False,
        dropout_rate=0.5,
        v1_dropout=False,
        random_crop_padding=0,
        normalize=True,
        fmnist_hflip_p=0.0,
        device="cuda:0"
    )
    main(cfg)