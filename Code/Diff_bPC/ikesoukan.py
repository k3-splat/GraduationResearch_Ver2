import math
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

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
    
    lt_min_a: float = 0.0       
    lt_min_e_disc: float = 0.0  
    lt_min_e_gen: float = 0.0   

    lt_scheduler_type: str = "cyclic_phase"
    gamma_value: float = 0.2; gamma_every_n: Optional[int] = None
    t_init_cycles: int = 15; phase2_cycles: int = 15
    alpha_disc: float = 1.0; alpha_gen: float = 0.01
    pc_lr: float = 5e-4
    batch_size: int = 256; epochs: int = 125
    use_adamw: bool = True
    adamw_weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    clip_grad_norm: float = 1.0 
    seed: int = 42
    run_name: Optional[str] = None 
    use_fashion_mnist: bool = False
    dropout_rate: float = 0.0               
    v1_dropout: bool = False                
    random_crop_padding: int = 0
    normalize: bool = True                  
    fmnist_hflip_p: float = 0.0             
    device: str = "cuda"


@dataclass
class SpikeStats:
    """Container for spike counts returned by runners."""
    sa_total: float = 0.0
    se_disc_total: float = 0.0
    se_gen_total: float = 0.0

@dataclass
class CostStats:
    """Container for computational cost metrics (Dense vs Sparse)."""
    # FLOPs
    dense_flops: float = 0.0
    sparse_flops: float = 0.0
    
    # Memory (Bytes)
    dense_mem_bytes: float = 0.0
    sparse_mem_bytes: float = 0.0
    
    # Communication (Bits) - CHANGED from words/spikes to bits
    dense_comm_bits: float = 0.0
    sparse_comm_bits: float = 0.0

    def __add__(self, other):
        if not isinstance(other, CostStats):
            return NotImplemented
        return CostStats(
            dense_flops = self.dense_flops + other.dense_flops,
            sparse_flops = self.sparse_flops + other.sparse_flops,
            dense_mem_bytes = self.dense_mem_bytes + other.dense_mem_bytes,
            sparse_mem_bytes = self.sparse_mem_bytes + other.sparse_mem_bytes,
            # Accumulate bits
            dense_comm_bits = self.dense_comm_bits + other.dense_comm_bits,
            sparse_comm_bits = self.sparse_comm_bits + other.sparse_comm_bits
        )

# =========================
#  Schedulers + factories
# =========================
# (省略: 前回のコードと同じ)
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
        
        # --- Thresholding & Spiking ---
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

        # buffer update
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

    def reset_all_states(self, batch_size: int):
        for lyr in [self.input_driver, *self.layers]:
            lyr._alloc(batch_size); lyr.time_step = 1
        self._global_step = 0
    
    def set_training(self, flag: bool):
        self.train(flag)
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
            s_in_disc_l = prev_s_disc @ self.W[l].T
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
                e_in_disc_l = self.layers[l].s_e_disc_prev @ self.W[l]
                if l == 0:
                    e_in_gen_l = self.input_driver.s_e_gen_prev @ self.V[l]
                else:
                    e_in_gen_l = self.layers[l-1].s_e_gen_prev @ self.V[l]

            else:
                e_in_disc_l = torch.zeros_like(self.layers[l-1].s_e_disc_prev)

            e_in_disc.append(e_in_disc_l)
            e_in_gen.append(e_in_gen_l)

        return s_in_disc, s_in_gen, e_in_disc, e_in_gen

    @torch.no_grad()
    def one_time_step(self, bottomup_mask: bool = False, topdown_mask : bool = False) -> CostStats:
        """
        Executes one time step and returns estimated computational costs for this step.
        """
        B  = self.input_driver.x_F_disc.size(0)
        z  = torch.zeros(B, self.input_driver.dim, device=self.device)
        z_top = torch.zeros(B, self.layers[-1].dim, device=self.device)
        sd = self.input_driver.sampling_duration
        t  = self._global_step % sd

        # --- COST ACCUMULATORS ---
        dense_flops = 0.0
        sparse_flops = 0.0
        dense_mem = 0.0
        sparse_mem = 0.0
        dense_comm_bits = 0.0
        sparse_comm_bits = 0.0

        # Helper for cost calc
        def calc_conn_cost_and_bits(vec_in, mat_weight):
            # vec_in: (B, Dim_In), mat_weight: (Dim_Out, Dim_In)
            dim_in = mat_weight.size(1)
            dim_out = mat_weight.size(0)
            
            # DENSE
            d_flops = 2.0 * B * dim_in * dim_out # MVM
            d_mem = (dim_in * dim_out * 4) + (B * (dim_in + dim_out) * 4) # Weights + In/Out vecs
            
            # Dense Comm: Words * 32 bits
            d_comm_count = float(B * dim_in)
            d_bits = d_comm_count * 32.0 

            # SPARSE
            nnz_in = float(vec_in.nonzero().size(0))
            s_flops = 2.0 * nnz_in * dim_out
            s_mem = (nnz_in * dim_out * 4) + (nnz_in * 4 * 2) 
            
            # Sparse Comm: Spikes * Address Width (log2(Dim_In))
            # If dim_in=1, log2(1)=0, but effectively 1 bit or 0 if implicit. Let's say 1 minimum.
            addr_bits = math.ceil(math.log2(dim_in)) if dim_in > 1 else 1.0
            s_bits = nnz_in * addr_bits

            return d_flops, s_flops, d_mem, s_mem, d_bits, s_bits

        # Forward Pass Spikes (s_A)
        # Input Driver -> Layer 0
        prev_s_disc = self.input_driver.s_A_prev
        df, sf, dm, sm, db, sb = calc_conn_cost_and_bits(prev_s_disc, self.W[0])
        dense_flops += df; sparse_flops += sf
        dense_mem += dm; sparse_mem += sm
        dense_comm_bits += db; sparse_comm_bits += sb
        
        # Layer i -> Layer i+1 (Disc) & Layer i+1 -> Layer i (Gen)
        for l in range(len(self.W)):
            # Discriminative Path (W): s_A[l] -> x_F[l+1]
            if l > 0: # 0 handled above (driver)
                prev_s = self.layers[l-1].s_A_prev
                df, sf, dm, sm, db, sb = calc_conn_cost_and_bits(prev_s, self.W[l])
                dense_flops += df; sparse_flops += sf
                dense_mem += dm; sparse_mem += sm
                dense_comm_bits += db; sparse_comm_bits += sb

            # Generative Path (V): s_A[l+1] -> x_F[l]
            if l < len(self.W):
                prev_s_gen_src = self.layers[l].s_A_prev 
                df, sf, dm, sm, db, sb = calc_conn_cost_and_bits(prev_s_gen_src, self.V[l])
                dense_flops += df; sparse_flops += sf
                dense_mem += dm; sparse_mem += sm
                dense_comm_bits += db; sparse_comm_bits += sb

        # Error Propagation
        for l in range(len(self.cfg.layer_dims) - 1):
            # Error Disc: s_e_disc[l+1] @ W[l] -> e_in_disc[l]
            s_e = self.layers[l].s_e_disc_prev
            # Source is Post-synaptic error, so "dim_in" for this calculation is W[l].size(0) (Post)
            # The matrix op effectively uses W[l].T
            
            # DENSE
            d_flops = 2.0 * B * self.W[l].numel()
            dense_flops += d_flops
            dense_mem += (self.W[l].numel() * 4) + (B * sum(self.W[l].shape) * 4)
            # Comm: sending Post-error
            dim_post = self.W[l].size(0)
            dense_comm_bits += float(B * dim_post * 32.0)

            # SPARSE
            nnz_se = float(s_e.nonzero().size(0))
            s_flops = 2.0 * nnz_se * self.W[l].size(1) 
            sparse_flops += s_flops
            sparse_mem += (nnz_se * self.W[l].size(1) * 4)
            # Comm: spikes * log2(dim_post)
            addr_bits = math.ceil(math.log2(dim_post)) if dim_post > 1 else 1.0
            sparse_comm_bits += nnz_se * addr_bits


            # Error Gen: s_e_gen[l] @ V[l] -> e_in_gen[l+1]
            if l == 0: s_e_g = self.input_driver.s_e_gen_prev
            else: s_e_g = self.layers[l-1].s_e_gen_prev
            
            # DENSE
            d_flops = 2.0 * B * self.V[l].numel()
            dense_flops += d_flops
            dense_mem += (self.V[l].numel() * 4) + (B * sum(self.V[l].shape) * 4)
            # Comm: sending Pre-error (V connects Pre->Post, but we are backpropagating? No, e_gen is forward diff)
            # V connects (Dim_Pre) -> (Dim_Post). s_e_gen is at (Dim_Pre).
            # So "Source" is Dim_Pre.
            dim_pre = self.V[l].size(1)
            dense_comm_bits += float(B * dim_pre * 32.0)

            # SPARSE
            nnz_seg = float(s_e_g.nonzero().size(0))
            s_flops = 2.0 * nnz_seg * self.V[l].size(1) # actually broadcasting to dim_post? No V is (Pre, Post).
            # Wait, V is (Pre, Post) in constructor: V[l]: (layer_dims[i], layer_dims[i+1]).
            # s_e_gen_prev is at layer i (Pre). 
            # Logic: s_e_gen @ V[l] -> (B, Pre) @ (Pre, Post) -> (B, Post).
            
            sparse_flops += s_flops
            sparse_mem += (nnz_seg * self.V[l].size(0) * 4) # Accessing columns of V corresponding to input spikes
            
            addr_bits = math.ceil(math.log2(dim_pre)) if dim_pre > 1 else 1.0
            sparse_comm_bits += nnz_seg * addr_bits


        # Actually perform the ops
        s_in_disc, s_in_gen, e_in_disc, e_in_gen = self._build_s_in_and_e_in()
        s_in_gen.append(z_top)

        # --- 2. Neuron Updates (Element-wise) ---
        total_neurons = sum(self.cfg.layer_dims)
        
        # Dense
        dense_flops += total_neurons * B * 20.0
        dense_mem += total_neurons * B * 10 * 4 
        
        # Sparse
        total_spikes = 0.0
        total_spikes += float(self.input_driver.s_A_prev.nonzero().size(0))
        for lyr in self.layers:
            total_spikes += float(lyr.s_A_prev.nonzero().size(0))
            total_spikes += float(lyr.s_e_disc_prev.nonzero().size(0))
            total_spikes += float(lyr.s_e_gen_prev.nonzero().size(0))
        
        sparsity_ratio = min(1.0, total_spikes / (total_neurons * B + 1e-6))
        sparse_flops += (total_neurons * B * 20.0) * sparsity_ratio
        sparse_mem += (total_neurons * B * 10 * 4) * sparsity_ratio

        self.input_driver.step(self.input_driver_clamp, self.input_driver_data, z, s_in_gen[0],
                               e_in_disc[0], z, False, topdown_mask, sample_step_override=t)
        for i, lyr in enumerate(self.layers):
            lyr.step(self._clamp_switch[i], self._data_bucket[i],
                      s_in_disc[i], s_in_gen[i+1], e_in_disc[i+1], e_in_gen[i],
                      bottomup_mask, topdown_mask, sample_step_override=t)
        
        self._global_step += 1
        
        return CostStats(
            dense_flops=dense_flops, sparse_flops=sparse_flops,
            dense_mem_bytes=dense_mem, sparse_mem_bytes=sparse_mem,
            dense_comm_bits=dense_comm_bits, sparse_comm_bits=sparse_comm_bits
        )

    @torch.no_grad()
    def apply_phase2_update(self) -> CostStats:
        """Returns CostStats estimate for the update."""
        if not self.layers[0].training: return CostStats()
        Bf = float(self.input_driver.x_T.size(0))
        self.optimizer.zero_grad(set_to_none=True)
        
        dense_flops = 0.0
        sparse_flops = 0.0
        dense_mem = 0.0
        sparse_mem = 0.0

        for l, lyr in enumerate(self.layers):
            pre_xT_disc  = self.input_driver.x_T if l == 0 else self.layers[l-1].x_T
            pre_xT_gen  = lyr.x_T
            post_eT_disc = lyr.e_T_disc
            post_eT_gen = self.input_driver.e_T_gen if l == 0 else self.layers[l-1].e_T_gen

            dim_post, dim_pre = self.W[l].shape
            
            # Dense Cost
            d_flops_layer = 2 * Bf * dim_post * dim_pre * 2 
            dense_flops += d_flops_layer
            dense_mem += (Bf * (dim_post + dim_pre) * 4 * 2) + ((dim_post * dim_pre) * 4 * 2)

            # Sparse Cost
            nnz_pre = float(torch.count_nonzero(pre_xT_disc))
            nnz_post = float(torch.count_nonzero(post_eT_disc))
            sparsity_pre = nnz_pre / (Bf * dim_pre)
            sparsity_post = nnz_post / (Bf * dim_post)
            
            sparse_flops += d_flops_layer * sparsity_pre * sparsity_post
            sparse_mem += dense_mem * (sparsity_pre + sparsity_post) / 2.0 

            self.W[l].grad = - self.cfg.alpha_disc * (post_eT_disc.T @ torch.relu(pre_xT_disc)) / Bf
            self.W_bias[l].grad = - self.cfg.alpha_disc * post_eT_disc.sum(dim=0, keepdim=True).T / Bf
            self.V[l].grad = - self.cfg.alpha_gen * (post_eT_gen.T @ torch.relu(pre_xT_gen)) / Bf
            self.V_bias[l].grad = - self.cfg.alpha_gen * post_eT_gen.sum(dim=0, keepdim=True).T / Bf

        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.W) + list(self.W_bias) + list(self.V) + list(self.V_bias), self.cfg.clip_grad_norm)
        self.optimizer.step()
        
        return CostStats(
            dense_flops=dense_flops, sparse_flops=sparse_flops,
            dense_mem_bytes=dense_mem, sparse_mem_bytes=sparse_mem,
            dense_comm_bits=0.0, sparse_comm_bits=0.0 # No inter-layer comm during weight update typically
        )


# =========================
#  Runners
# =========================

@torch.no_grad()
def run_batch_two_phase(net: DiffPCNetworkTorch, x: torch.Tensor, y_onehot: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats, Dict[str, List[float]], CostStats]:
    
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(x.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])

    total_costs = CostStats()

    # Phase-1
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)
    net.set_training(False)
    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    
    for _ in range(steps_phase1): 
        step_cost = net.one_time_step(bottomup_mask=False, topdown_mask=True)
        total_costs = total_costs + step_cost

    # Phase-2
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)
    net.set_training(True)
    net.set_clamp(len(net.layers), True, y_onehot)
    
    spike_stats = SpikeStats()
    batch_size = x.size(0)

    for _ in range(steps_phase2):
        step_cost = net.one_time_step(bottomup_mask=False, topdown_mask=False)
        total_costs = total_costs + step_cost
        
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_disc_total += (lyr.s_e_disc != 0).sum().item()
            spike_stats.se_gen_total += (lyr.s_e_gen != 0).sum().item()

    batch_error_stats = {
        "input_gen": net.input_driver.e_T_gen.abs().sum().item() / batch_size,
        "layers_disc": [lyr.e_T_disc.abs().sum().item() / batch_size for lyr in net.layers],
        "layers_gen": [lyr.e_T_gen.abs().sum().item() / batch_size for lyr in net.layers]
    }

    update_costs = net.apply_phase2_update()
    total_costs = total_costs + update_costs

    return net.layers[-1].x_T.clone(), spike_stats, batch_error_stats, total_costs

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

# =========================
#  Analysis Tools
# =========================
# (省略: 前回のコードと同じ)
def compute_average_digit_images(loader: DataLoader, device: torch.device) -> torch.Tensor:
    print("Pre-computing average digit images for RMSE calculation...")
    sums = torch.zeros(10, 784, device=device)
    counts = torch.zeros(10, device=device)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            for i in range(10):
                mask = (labels == i)
                if mask.sum() > 0:
                    sums[i] += imgs[mask].sum(dim=0)
                    counts[i] += mask.sum()
    counts = torch.clamp(counts, min=1.0)
    avg_images = sums / counts.unsqueeze(1)
    return avg_images

def visualize_generated_digits(net: DiffPCNetworkTorch, cfg: DiffPCConfig, 
                               l_t_spec: dict, y_phase2_spec: dict,
                               epoch: int, avg_digit_images: Optional[torch.Tensor] = None, 
                               output_dir: str = "gen_results") -> float:
    os.makedirs(output_dir, exist_ok=True)
    device = net.device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = torch.arange(10, device=device)
    y_onehot = F.one_hot(labels, num_classes=cfg.layer_dims[-1]).float()
    
    generated_data, _ = infer_batch_backward_phase2(net, y_onehot, cfg, l_t_spec, y_phase2_spec)
    
    rmse_val = 0.0
    if avg_digit_images is not None:
        mse = F.mse_loss(generated_data, avg_digit_images)
        rmse_val = torch.sqrt(mse).item()
    
    gen_imgs = generated_data.view(10, 1, 28, 28)
    if cfg.normalize:
        if cfg.use_fashion_mnist:
            mean, std = 0.2860, 0.3530
        else:
            mean, std = 0.1307, 0.3081
        gen_imgs = gen_imgs * std + mean
    gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0)
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_{timestamp}_RMSE_{rmse_val:.4f}.png")
    save_image(gen_imgs, save_path, nrow=10)
    return rmse_val

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
    plt.xlabel('Global Batch Step')
    plt.ylabel('Avg |e_T_disc|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(steps, input_gen_hist, linewidth=0.5, linestyle='--', label='Input (L0) Gen Err')
    for i in range(num_layers):
        plt.plot(steps, layers_gen_hist[i], linewidth=0.5, label=f'Layer {i+1} Gen Err')
    plt.title('Generative Error Snapshot')
    plt.xlabel('Global Batch Step')
    plt.ylabel('Avg |e_T_gen|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"batch_error_curves_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Full batch error plot saved to: {save_path}")


# =========================
#  MNIST training wrapper
# =========================

def main(cfg: DiffPCConfig):
    torch.manual_seed(cfg.seed)
    requested = str(cfg.device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                device = torch.device(requested)
                if device.index is not None and device.index >= torch.cuda.device_count():
                    device = torch.device("cuda:0")
                torch.cuda.set_device(device)
            except Exception as e:
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")

    l_t_spec = {
        "type": cfg.lt_scheduler_type,
        "args": {"m": cfg.lt_m, "n": cfg.lt_n, "a": cfg.lt_a}
    }
    y_phase2_spec = {
        "type": "on_cycle_start",
        "args": {"gamma": cfg.gamma_value, "n": (cfg.gamma_every_n or cfg.lt_n)}
    }
    net = DiffPCNetworkTorch(
        cfg,
        l_t_scheduler_spec=l_t_spec,
        y_scheduler_spec=y_phase2_spec,
        device=str(device)
    )
    
    if cfg.use_fashion_mnist:
        mean, std = 0.2860, 0.3530
    else:
        mean, std = 0.1307, 0.3081

    train_transforms_list = []
    if cfg.random_crop_padding > 0:
        if cfg.use_fashion_mnist:
            train_transforms_list.append(transforms.RandomCrop(28, padding=cfg.random_crop_padding, padding_mode="edge"))
        else:
            train_transforms_list.append(transforms.RandomCrop(28, padding=cfg.random_crop_padding))
    if cfg.use_fashion_mnist and cfg.fmnist_hflip_p > 0.0:
        train_transforms_list.append(transforms.RandomHorizontalFlip(p=cfg.fmnist_hflip_p))
    train_transforms_list.append(transforms.ToTensor())
    if cfg.normalize:
        train_transforms_list.append(transforms.Normalize((mean,), (std,)))
    train_transform = transforms.Compose(train_transforms_list)

    test_transforms_list = [transforms.ToTensor()]
    if cfg.normalize:
        test_transforms_list.append(transforms.Normalize((mean,), (std,)))
    test_transform = transforms.Compose(test_transforms_list)

    if cfg.use_fashion_mnist:
        train_ds = datasets.FashionMNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=test_transform)
    else:
        train_ds = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    avg_digit_images = compute_average_digit_images(test_loader, device)

    def to_vec(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).to(device)
    def to_onehot(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=cfg.layer_dims[-1]).float().to(device)

    run_results = []
    global_batch_error_history = [] 
    cumulative_costs = CostStats()
    total_neurons = sum(cfg.layer_dims[1:])

    print(f"Starting run: {run_name}")
    print(f"Using device: {device}")

    for epoch in range(1, cfg.epochs + 1):
        total_sa_train_p2, total_se_disc_train_p2, total_se_gen_train_p2 = 0.0, 0.0, 0.0
        epoch_err_accum = {
            "input_gen": 0.0,
            "layers_disc": [0.0] * len(net.layers),
            "layers_gen": [0.0] * len(net.layers)
        }
        total_samples = 0

        for images, labels in train_loader:
            _, stats, batch_err_stats, batch_costs = run_batch_two_phase(net, to_vec(images), to_onehot(labels), cfg, l_t_spec, y_phase2_spec)
            
            global_batch_error_history.append(batch_err_stats)
            total_sa_train_p2 += stats.sa_total
            total_se_disc_train_p2 += stats.se_disc_total
            total_se_gen_train_p2 += stats.se_gen_total
            cumulative_costs = cumulative_costs + batch_costs
            bs = images.size(0)
            total_samples += bs
            epoch_err_accum["input_gen"] += batch_err_stats["input_gen"] * bs
            for i in range(len(net.layers)):
                epoch_err_accum["layers_disc"][i] += batch_err_stats["layers_disc"][i] * bs
                epoch_err_accum["layers_gen"][i] += batch_err_stats["layers_gen"][i] * bs

        train_correct, train_total = 0, 0
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                preds = logits.argmax(dim=1).cpu()
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)
            for images, labels in test_loader:
                logits, stats = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                preds = logits.argmax(dim=1).cpu()
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc  = 100.0 * test_correct  / test_total
        rmse = visualize_generated_digits(net, cfg, l_t_spec, y_phase2_spec, epoch, avg_digit_images)

        # Log Cumulative Costs
        d_flops_G = cumulative_costs.dense_flops / 1e9
        s_flops_G = cumulative_costs.sparse_flops / 1e9
        d_mem_GB = cumulative_costs.dense_mem_bytes / 1e9
        s_mem_GB = cumulative_costs.sparse_mem_bytes / 1e9
        # Conversion to Gbits (1e9 bits)
        d_comm_Gb = cumulative_costs.dense_comm_bits / 1e9
        s_comm_Gb = cumulative_costs.sparse_comm_bits / 1e9

        print(
            f"Epoch {epoch:02d}: train acc {train_acc:.2f}% | test acc {test_acc:.2f}% | RMSE {rmse:.4f}"
        )
        print(f"  > Cumulative Costs:")
        print(f"    [Dense]  FLOPs: {d_flops_G:.2f}G, Mem: {d_mem_GB:.2f}GB, Comm: {d_comm_Gb:.2f}G bits")
        print(f"    [Sparse] FLOPs: {s_flops_G:.2f}G, Mem: {s_mem_GB:.2f}GB, Comm: {s_comm_Gb:.2f}G bits")

        run_results.append({
            "epoch": epoch, 
            "train_acc": train_acc, 
            "test_acc": test_acc,
            "rmse": rmse,
            "cumulative_dense_flops": cumulative_costs.dense_flops,
            "cumulative_sparse_flops": cumulative_costs.sparse_flops,
            "cumulative_dense_mem_bytes": cumulative_costs.dense_mem_bytes,
            "cumulative_sparse_mem_bytes": cumulative_costs.sparse_mem_bytes,
            "cumulative_dense_comm_bits": cumulative_costs.dense_comm_bits,
            "cumulative_sparse_comm_bits": cumulative_costs.sparse_comm_bits
        })

    print("Training finished. Plotting full batch error history...")
    plot_batch_error_history(global_batch_error_history)
    with open(log_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": run_results}, f, indent=4)

if __name__ == "__main__":
    cfg = DiffPCConfig(
        layer_dims=[784, 400, 10],
        lt_m=-4,
        lt_n=6,
        lt_a=1.0,
        lt_min_a=0.0,
        lt_min_e_disc=0.0,
        lt_min_e_gen=0.0,
        lt_scheduler_type="cyclic_phase",
        gamma_value=0.05,
        gamma_every_n=None,
        t_init_cycles=20,
        phase2_cycles=20,
        alpha_disc = 1,
        alpha_gen = 0.01,
        pc_lr=0.0001,
        batch_size=256,
        epochs=20,
        use_adamw=True,
        adamw_weight_decay=0.01,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-08,
        clip_grad_norm=1.0,
        seed=2,
        run_name="mnist_cost_analysis",
        use_fashion_mnist=False,
        dropout_rate=0.5,
        v1_dropout=False,
        random_crop_padding=2,
        normalize=True,
        fmnist_hflip_p=0.0,
        device="cuda:0" 
    )
    main(cfg)