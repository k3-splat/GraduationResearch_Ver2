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
import matplotlib.pyplot as plt


# =========================
#  Config & Stats
# =========================

@dataclass
class DiffPCConfig:
    """Central configuration for the DiffPC model and training."""
    layer_dims: list[int] = (784, 200, 10)
    lt_m: int = 3; lt_n: int = 4; lt_a: float = 0.5
    lt_scheduler_type: str = "cyclic_phase"   # "cyclic_phase" or "constant"
    gamma_value: float = 0.2; gamma_every_n: Optional[int] = None
    t_init_cycles: int = 15; phase2_cycles: int = 15
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
    dropout_rate: float = 0.0             # unified dropout rate
    v1_dropout: bool = False              # choose v1 fixed-mask or v2 nn.Dropout
    random_crop_padding: int = 0
    normalize: bool = True                # toggle Normalize(mean,std)
    fmnist_hflip_p: float = 0.0           # optional horizontal flip prob for FMNIST train
    device: str = "cuda"


@dataclass
class SpikeStats:
    """Container for spike counts, error stats, and comprehensive computational costs."""
    sa_total: float = 0.0
    se_total: float = 0.0
    # Error stats (Mean per Data per Neuron)
    et_input: float = 0.0   # Input Layer
    et_hidden: float = 0.0  # Hidden Layers
    et_output: float = 0.0  # Output Layer
    
    # --- Computational Cost Metrics (Accumulated per Batch) ---
    # 1. FLOPs (Giga Operations) - Arithmetic intensity
    flops_dense: float = 0.0
    flops_sparse: float = 0.0
    
    # 2. Data Movement (Bytes) - Memory Access / IO
    dm_dense: float = 0.0
    dm_sparse: float = 0.0
    
    # 3. Transfer Bits (Bits) - Inter-layer Communication Bandwidth
    bits_dense: float = 0.0
    bits_sparse: float = 0.0
    # -----------------------------------------

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

def _sign_ternary(x: torch.Tensor) -> torch.Tensor: return torch.sign(x)

class DiffPCLayerTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.device = torch.device(self.device)
        self.x_F, self.x_T, self.x_A, self.e_T, self.e_A, self.s_A, self.s_e, \
        self.s_in_data, self.e_in_data, self.data = [None] * 10
        self.l_t, self.y, self.time_step = 0.0, 0.0, 1
        self.reset_state, self.reset_state_type = True, "zero"
        # prev-tick buffers
        self.s_A_prev, self.s_e_prev = None, None

    def _alloc(self, B: int):
        z = torch.zeros(B, self.dim, device=self.device)
        self.x_F, self.x_T, self.x_A, self.e_T, self.e_A, self.s_A, self.s_e, \
        self.s_in_data, self.e_in_data, self.data = [z.clone() for _ in range(10)]
        # prev-tick buffers
        self.s_A_prev = torch.zeros_like(z)
        self.s_e_prev = torch.zeros_like(z)

    def _sample_step(self) -> int: return (self.time_step - 1) % self.sampling_duration

    def reset_states_if_needed(self, clamp_status: bool, sample_step: int):
        if (sample_step == 0) and self.reset_state:
            self.x_F.zero_(); self.x_A.zero_(); self.e_T.zero_()
            self.e_A.zero_(); self.e_in_data.zero_()
            if not clamp_status: self.x_T.zero_()
            if self.s_A_prev is not None: self.s_A_prev.zero_()
            if self.s_e_prev is not None: self.s_e_prev.zero_()

    @torch.no_grad()
    def step(self, clamp_status: bool, data_in: Optional[torch.Tensor],
             s_in_recv: torch.Tensor, e_in_recv: torch.Tensor,
             sample_step_override: Optional[int] = None):
        B = s_in_recv.size(0)
        if self.x_F is None or self.x_F.size(0) != B: self._alloc(B)
        
        sample_step = self._sample_step() if sample_step_override is None else sample_step_override
        
        if clamp_status: self.x_T.copy_(data_in.to(self.device))
        self.s_in_data = s_in_recv.to(self.device)
        self.reset_states_if_needed(clamp_status=clamp_status, sample_step=sample_step)
        
        l_t_cur  = self.l_t_scheduler.get_l_t(sample_step,     self.learning_weights)
        l_t_prev = self.l_t_scheduler.get_l_t(sample_step - 1, self.learning_weights)
        self.l_t = l_t_cur
        self.y   = self.y_scheduler.get_y(sample_step, self.learning_weights)

        self.x_F.add_(self.s_in_data * l_t_prev)
        self.e_T = self.x_T - self.x_F
        self.e_in_data.add_(e_in_recv.to(self.device) * l_t_prev)

        if not clamp_status:
            self.x_T.add_(self.y * (-self.e_T + (self.x_T > 0).float() * self.e_in_data))
        
        diff_act = self.x_T - self.x_A
        s_A_new = torch.sign(diff_act) * (diff_act.abs() > self.l_t)
        s_A_new = s_A_new * ((self.x_A + s_A_new * self.l_t) > 0.0)
        self.x_A.add_(s_A_new * self.l_t); self.s_A = s_A_new
        
        diff_err = self.e_T - self.e_A
        s_e_new = torch.sign(diff_err) * (diff_err.abs() > self.l_t)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration: s_e_new.zero_()
        self.e_A.add_(s_e_new * self.l_t); self.s_e = s_e_new

        # publish for next tick
        if self.s_A_prev is None or self.s_A_prev.shape != self.s_A.shape:
            self.s_A_prev = torch.zeros_like(self.s_A)
        if self.s_e_prev is None or self.s_e_prev.shape != self.s_e.shape:
            self.s_e_prev = torch.zeros_like(self.s_e)
        self.s_A_prev.copy_(self.s_A)
        self.s_e_prev.copy_(self.s_e)

        if sample_step_override is not None:
            self.time_step = int(sample_step_override) + 1
        else:
            self.time_step += 1

    def get_elementwise_costs(self, batch_size: int, clamp_status: bool, s_in_active_count: int = None, e_in_active_count: int = None):
        """
        Returns dictionary with Dense and Sparse costs for element-wise operations inside the layer.
        """
        dim = self.dim
        total_neurons = batch_size * dim
        
        # --- FLOPs Estimation ---
        # Dense: Always update all neurons
        # Approx cost per neuron: ~15 ops (add, mul, sub, threshold, update)
        flops_dense = total_neurons * 15

        # Sparse: Update only if inputs (s_in, e_in) are active or internal state changes.
        # Ideally, we track active internal neurons. Here we approximate using input activity ratio.
        # If s_in_active_count is provided, use it. Otherwise assume dense.
        if s_in_active_count is not None:
            # Assume base cost for "waking up" + cost proportional to input activity
            flops_sparse = (s_in_active_count + (e_in_active_count if e_in_active_count else 0)) * 15
            # Clamp to max dense
            flops_sparse = min(flops_sparse, flops_dense)
        else:
            flops_sparse = flops_dense

        # --- Data Movement (Bytes) ---
        # 10 state vars read/written * 4 bytes
        bytes_per_neuron = 10 * 4
        dm_dense = total_neurons * bytes_per_neuron
        
        if s_in_active_count is not None:
            # Sparse access: Only read/write states for active neurons
            dm_sparse = (s_in_active_count + (e_in_active_count if e_in_active_count else 0)) * bytes_per_neuron
            dm_sparse = min(dm_sparse, dm_dense)
        else:
            dm_sparse = dm_dense
            
        return flops_dense, flops_sparse, dm_dense, dm_sparse


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
        self.W_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.cfg.layer_dims[i+1], 1, device=self.device))
            for i in range(len(self.cfg.layer_dims) - 1)
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=cfg.dropout_rate) for _ in range(len(self.W))
        ])

        for w in self.W: nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if cfg.use_adamw:
            self.optimizer = torch.optim.AdamW(
                list(self.W) + list(self.W_bias),
                lr=cfg.pc_lr, betas=cfg.adamw_betas,
                eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                list(self.W) + list(self.W_bias),
                lr=cfg.pc_lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps
            )

        l_t_sched = get_l_t_scheduler(l_t_scheduler_spec["type"], l_t_scheduler_spec["args"])
        y_sched = get_y_scheduler(y_scheduler_spec["type"], {**y_scheduler_spec["args"], "l_t_scheduler": l_t_sched})
        layer_args = {"sampling_duration": 1, "learning_weights": True, "training": True,
                      "l_t_scheduler": l_t_sched, "y_scheduler": y_sched, "device": self.device}
        self.layers = nn.ModuleList([DiffPCLayerTorch(dim=d, **layer_args) for d in cfg.layer_dims[1:]])
        self.input_driver = DiffPCLayerTorch(dim=cfg.layer_dims[0], **layer_args)

        self.input_driver_clamp, self.input_driver_data = False, None
        self._clamp_switch = [False] * len(self.layers)
        self._data_bucket = [None] * len(self.layers)
        
        self._global_step = 0
        self._bias_fired_once = False

        self._v1_dropout_active: bool = False
        self._v1_drop_masks: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._layer_dims: List[int] = list(cfg.layer_dims[1:])

        # Accumulators for cost calculation (Step-level)
        self._metrics = {
            'flops_dense': 0.0, 'flops_sparse': 0.0,
            'dm_dense': 0.0, 'dm_sparse': 0.0,
            'bits_dense': 0.0, 'bits_sparse': 0.0
        }

    def reset_all_states(self, batch_size: int):
        for lyr in [self.input_driver, *self.layers]:
            lyr._alloc(batch_size); lyr.time_step = 1
        self._global_step = 0
    
    def set_training(self, flag: bool):
        self.train(flag)
        for lyr in [self.input_driver, *self.layers]:
            lyr.training = bool(flag)

    @torch.no_grad()
    def set_v1_dropout(self, active: bool, batch_size: int, redraw: bool = True):
        if not active or self.cfg.dropout_rate <= 0.0:
            self._v1_dropout_active = False
            for i in range(len(self._v1_drop_masks)): self._v1_drop_masks[i] = None
            return

        if self._v1_dropout_active and not redraw: return

        self._v1_dropout_active = True
        keep_prob = 1.0 - self.cfg.dropout_rate
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                self._v1_drop_masks[i] = None
            else:
                dim = self._layer_dims[i]
                self._v1_drop_masks[i] = torch.bernoulli(torch.full((batch_size, dim), keep_prob, device=self.device))

    def swap_schedulers(self, l_t_sched: LTBaseScheduler, y_sched: YBaseScheduler):
        for lyr in [self.input_driver, *self.layers]:
            lyr.l_t_scheduler = l_t_sched; lyr.y_scheduler = y_sched

    def set_sampling_duration(self, new_len: int):
        for lyr in [self.input_driver, *self.layers]: lyr.sampling_duration = new_len
        self._global_step = 0

    def set_clamp(self, layer_index: int, clamp: bool, data: Optional[torch.Tensor] = None):
        if layer_index == 0: self.input_driver_clamp, self.input_driver_data = clamp, data
        else: self._clamp_switch[layer_index-1], self._data_bucket[layer_index-1] = clamp, data

    def _calc_matrix_ops_cost(self, input_tensor: torch.Tensor, dim_out: int, is_bias: bool = False):
        """Helper to calculate FLOPs and Memory for Matrix Mult."""
        batch_size, dim_in = input_tensor.shape
        threshold = 1e-6
        
        # Count active inputs (Pseudo-Spikes)
        num_active = (input_tensor.abs() > threshold).sum().item()
        
        # --- FLOPs ---
        # Dense: Batch * In * Out * 2 (MAC)
        f_dense = batch_size * dim_in * dim_out * (1.0 if is_bias else 2.0)
        # Sparse: Active * Out * 2 (MAC)
        f_sparse = num_active * dim_out * (1.0 if is_bias else 2.0)
        
        # --- Data Movement (Bytes) ---
        # Weights (Float32 = 4 bytes)
        # Dense: Read all weights
        dm_dense = (dim_in * dim_out) * 4.0
        # Sparse: Read only rows for active inputs
        dm_sparse = (num_active * dim_out) * 4.0
        
        # Input/Output Data Read/Write (Dense assumption for I/O for simplicity, or sparse if optimized)
        # Here we assume standard read input / write output
        dm_io = (batch_size * dim_in + batch_size * dim_out) * 4.0
        dm_dense += dm_io
        # For sparse, we read only active input values, but write all output values (until thresholded later)
        dm_sparse += (num_active + batch_size * dim_out) * 4.0

        return f_dense, f_sparse, dm_dense, dm_sparse

    @torch.no_grad()
    def _build_s_in_and_e_in(self):
        s_in, e_in = [], []
        
        # Reset step metrics
        self._metrics = {k: 0.0 for k in self._metrics}

        sd = self.layers[0].sampling_duration if len(self.layers) > 0 else 1
        t  = self._global_step % sd

        emit_bias = (not self._bias_fired_once) and (t == 1)
        if emit_bias:
            bias_spike = 1.0; self._bias_fired_once = True
        else:
            bias_spike = 0.0

        prev_s = self.input_driver.s_A_prev
        
        # --- Forward Pass Calculations ---
        for l, w in enumerate(self.W):
            din, dout = w.shape[1], w.shape[0]
            
            # Cost Calculation (Forward Matrix Mult)
            fd, fs, dmd, dms = self._calc_matrix_ops_cost(prev_s, dout)
            self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
            self._metrics['dm_dense'] += dmd;   self._metrics['dm_sparse'] += dms
            
            # Actual Op
            s_in_l = prev_s @ w.T

            # Dropout & Bias logic (simplified cost: bias is 1 op per neuron)
            if self.cfg.v1_dropout:
                pass
            else:
                if self.training and l < len(self.W) - 1:
                    s_in_l = self.dropout_layers[l](s_in_l)

            if bias_spike != 0.0:
                s_in_l.add_(self.W_bias[l].T)
                # Bias add cost
                bs = s_in_l.size(0)
                self._metrics['flops_dense'] += bs * dout
                self._metrics['flops_sparse'] += bs * dout # Bias is dense add usually

            s_in.append(s_in_l)
            prev_s = self.layers[l].s_A_prev

        # --- Backward Pass Calculations ---
        for l in range(len(self.layers)):
            if l < len(self.layers) - 1:
                w_next = self.W[l+1]
                din, dout = w_next.shape[1], w_next.shape[0] # Note: W is (Dout_next, Dout_curr)
                
                # Feedback input comes from s_e_prev of next layer
                feedback_input = self.layers[l+1].s_e_prev
                
                # Cost Calculation (Backward Matrix Mult: e * W)
                # Input to matmul is (Batch, Dout), Output is (Batch, Din)
                # W is (Dout, Din)
                fd, fs, dmd, dms = self._calc_matrix_ops_cost(feedback_input, din)
                self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
                self._metrics['dm_dense'] += dmd;   self._metrics['dm_sparse'] += dms

                e_in_l = feedback_input @ w_next
                
                if self.cfg.v1_dropout and self._v1_dropout_active and self._v1_drop_masks[l+1] is not None:
                     e_in_l = (self.layers[l+1].s_e_prev * self._v1_drop_masks[l+1]) @ w_next
            else:
                e_in_l = torch.zeros_like(self.layers[l].s_e_prev)
            e_in.append(e_in_l)

        return s_in, e_in
    
    def _accumulate_transfer_bits(self):
        """Calculates Transfer Bits (Communication Bandwidth)."""
        # Collect all communicating tensors: s_A_prev (forward) and s_e_prev (backward)
        # Input driver only sends s_A
        communicating_tensors = [self.input_driver.s_A_prev] 
        # Layers send s_A (forward) and s_e (backward)
        for lyr in self.layers:
            communicating_tensors.append(lyr.s_A_prev)
            communicating_tensors.append(lyr.s_e_prev)
            
        for x in communicating_tensors:
            num_elements = x.numel()
            n_neurons = x.shape[1]
            num_active = (x.abs() > 1e-6).sum().item()
            
            # Dense: All 32-bit
            self._metrics['bits_dense'] += num_elements * 32
            
            # Sparse: Active * (32-bit val + Address)
            # Address bits = log2(N_neurons)
            addr_bits = math.ceil(math.log2(n_neurons)) if n_neurons > 1 else 1
            self._metrics['bits_sparse'] += num_active * (32 + addr_bits)

    @torch.no_grad()
    def one_time_step(self):
        B = self.input_driver.x_F.size(0)
        z = torch.zeros(B, self.input_driver.dim, device=self.device)
        sd = self.input_driver.sampling_duration
        t  = self._global_step % sd

        # 1. Matrix Multiplications & Costs
        s_in, e_in = self._build_s_in_and_e_in()
        
        # 2. Transfer Bits Cost
        self._accumulate_transfer_bits()
        
        # 3. Element-wise Steps & Costs
        # Input Driver
        # Input driver receives no s_in/e_in from network, so active counts are 0
        fd, fs, dmd, dms = self.input_driver.get_elementwise_costs(B, self.input_driver_clamp, 0, 0)
        self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
        self._metrics['dm_dense'] += dmd;   self._metrics['dm_sparse'] += dms
        
        self.input_driver.step(self.input_driver_clamp, self.input_driver_data, z, z, sample_step_override=t)
        
        # Layers
        for i, lyr in enumerate(self.layers):
            # Count active inputs for sparse element-wise cost estimation
            s_act = (s_in[i].abs() > 1e-6).sum().item()
            e_act = (e_in[i].abs() > 1e-6).sum().item()
            
            fd, fs, dmd, dms = lyr.get_elementwise_costs(B, self._clamp_switch[i], s_act, e_act)
            self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
            self._metrics['dm_dense'] += dmd;   self._metrics['dm_sparse'] += dms
            
            lyr.step(self._clamp_switch[i], self._data_bucket[i], s_in[i], e_in[i], sample_step_override=t)
        
        self._global_step += 1

    @torch.no_grad()
    def apply_phase2_update(self):
        if not self.layers[0].training: return
        Bf = float(self.input_driver.x_T.size(0))
        self.optimizer.zero_grad(set_to_none=True)

        for l, lyr in enumerate(self.layers):
            pre_xT  = self.input_driver.x_T if l == 0 else self.layers[l-1].x_T
            post_eT = lyr.e_T

            if self.cfg.v1_dropout and self._v1_dropout_active:
                if l > 0 and self._v1_drop_masks[l-1] is not None:
                    pre_xT = pre_xT * self._v1_drop_masks[l-1]
                if self._v1_drop_masks[l] is not None:
                    post_eT = post_eT * self._v1_drop_masks[l]

            self.W[l].grad = - (post_eT.T @ torch.relu(pre_xT)) / Bf
            self.W_bias[l].grad = - post_eT.sum(dim=0, keepdim=True).T / Bf

        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.W) + list(self.W_bias), self.cfg.clip_grad_norm)
        self.optimizer.step()


# =========================
#  Runners
# =========================

@torch.no_grad()
def run_batch_two_phase(net: DiffPCNetworkTorch, x: torch.Tensor, y_onehot: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(x.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])
    spike_stats = SpikeStats()

    # Phase-1
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    if cfg.v1_dropout:
        net.set_training(False)
        net.set_v1_dropout(True, x.size(0), redraw=True)
    else:
        net.set_training(False)

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    
    for _ in range(steps_phase1): 
        net.one_time_step()
        spike_stats.flops_dense += net._metrics['flops_dense']
        spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']
        spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']
        spike_stats.bits_sparse += net._metrics['bits_sparse']

    # Phase-2
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)

    if cfg.v1_dropout:
        net.set_training(True)
        net.set_v1_dropout(True, x.size(0), redraw=False)
    else:
        net.set_training(True)

    net.set_clamp(len(net.layers), True, y_onehot)
    
    for _ in range(steps_phase2):
        net.one_time_step()
        spike_stats.flops_dense += net._metrics['flops_dense']
        spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']
        spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']
        spike_stats.bits_sparse += net._metrics['bits_sparse']

        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()

    # Error Stats
    batch_size = x.size(0)
    if batch_size > 0:
        spike_stats.et_input = net.input_driver.e_T.abs().sum().item() / batch_size / net.input_driver.dim
        spike_stats.et_output = net.layers[-1].e_T.abs().sum().item() / batch_size / net.layers[-1].dim
        hidden_err_sum, hidden_neurons = 0.0, 0
        for i in range(len(net.layers) - 1):
            hidden_err_sum += net.layers[i].e_T.abs().sum().item()
            hidden_neurons += net.layers[i].dim
        if hidden_neurons > 0:
            spike_stats.et_hidden = hidden_err_sum / batch_size / hidden_neurons
    
    net.apply_phase2_update()
    
    return net.layers[-1].x_T.clone(), spike_stats

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

    if cfg.v1_dropout:
        net.set_training(False)
        net.set_v1_dropout(False, x.size(0))
    else:
        net.set_training(False)

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    
    spike_stats = SpikeStats()
    for _ in range(steps):
        net.one_time_step()
        # Accumulate costs during inference too
        spike_stats.flops_dense += net._metrics['flops_dense']
        spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']
        spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']
        spike_stats.bits_sparse += net._metrics['bits_sparse']

        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()
            
    return net.layers[-1].x_T.clone(), spike_stats


# =========================
#  MNIST training wrapper
# =========================

def main(cfg: DiffPCConfig):
    # Setup
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")
    plot_path = os.path.join(log_dir, f"{run_name}_error.png")
    cost_plot_path = os.path.join(log_dir, f"{run_name}_cost.png")

    # Network and Schedulers
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
    
    # Data transforms
    mean, std = (0.2860, 0.3530) if cfg.use_fashion_mnist else (0.1307, 0.3081)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

    if cfg.use_fashion_mnist:
        train_ds = datasets.FashionMNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=test_transform)
    else:
        train_ds = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
        test_ds  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    def to_vec(x: torch.Tensor) -> torch.Tensor: return x.view(x.size(0), -1).to(device)
    def to_onehot(y: torch.Tensor) -> torch.Tensor: return F.one_hot(y, num_classes=cfg.layer_dims[-1]).float().to(device)

    run_results = []
    # Store history for plots
    h_flops_d, h_flops_s = [], []
    h_dm_d, h_dm_s = [], []
    h_bits_d, h_bits_s = [], []
    
    total_neurons = sum(cfg.layer_dims[1:])

    print(f"Starting run: {run_name} on {device}")

    for epoch in range(1, cfg.epochs + 1):
        # Training
        ep_stats = SpikeStats()
        num_batches = 0

        for images, labels in train_loader:
            _, stats = run_batch_two_phase(net, to_vec(images), to_onehot(labels), cfg, l_t_spec, y_phase2_spec)
            
            # Aggregate Costs
            ep_stats.flops_dense += stats.flops_dense
            ep_stats.flops_sparse += stats.flops_sparse
            ep_stats.dm_dense += stats.dm_dense
            ep_stats.dm_sparse += stats.dm_sparse
            ep_stats.bits_dense += stats.bits_dense
            ep_stats.bits_sparse += stats.bits_sparse
            num_batches += 1

            # Append per-batch stats for plots
            h_flops_d.append(stats.flops_dense); h_flops_s.append(stats.flops_sparse)
            h_dm_d.append(stats.dm_dense / (1024**2)); h_dm_s.append(stats.dm_sparse / (1024**2))
            h_bits_d.append(stats.bits_dense / 1e9); h_bits_s.append(stats.bits_sparse / 1e9)

        # Evaluation
        train_correct, train_total = 0, 0
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for images, labels in train_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                train_correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
                train_total   += labels.size(0)

            for images, labels in test_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                test_correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
                test_total   += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc  = 100.0 * test_correct  / test_total

        # Avg costs per batch
        avg_flops_d_g = (ep_stats.flops_dense / num_batches) / 1e9
        avg_flops_s_g = (ep_stats.flops_sparse / num_batches) / 1e9
        avg_dm_d_mb   = (ep_stats.dm_dense / num_batches) / (1024**2)
        avg_dm_s_mb   = (ep_stats.dm_sparse / num_batches) / (1024**2)
        avg_bits_d_gb = (ep_stats.bits_dense / num_batches) / 1e9
        avg_bits_s_gb = (ep_stats.bits_sparse / num_batches) / 1e9

        print(
            f"Epoch {epoch:02d}: Tr {train_acc:.1f}% / Te {test_acc:.1f}% | "
            f"FLOPs(G): D={avg_flops_d_g:.1f}, S={avg_flops_s_g:.1f} | "
            f"DM(MB): D={avg_dm_d_mb:.1f}, S={avg_dm_s_mb:.1f} | "
            f"Bits(Gb): D={avg_bits_d_gb:.1f}, S={avg_bits_s_gb:.1f}"
        )

        run_results.append({
            "epoch": epoch, "train_acc": train_acc, "test_acc": test_acc,
            "avg_flops_dense_G": avg_flops_d_g, "avg_flops_sparse_G": avg_flops_s_g,
            "avg_dm_dense_MB": avg_dm_d_mb, "avg_dm_sparse_MB": avg_dm_s_mb,
            "avg_bits_dense_Gb": avg_bits_d_gb, "avg_bits_sparse_Gb": avg_bits_s_gb
        })

    # Save Results
    with open(log_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": run_results}, f, indent=4)

    # --- Plotting Costs ---
    print("Generating cost plot...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    axes2[0].plot(h_flops_d, label='Dense', color='red', alpha=0.5)
    axes2[0].plot(h_flops_s, label='Sparse', color='green', alpha=0.8)
    axes2[0].set_title('FLOPs'); axes2[0].set_ylabel('Ops/Batch'); axes2[0].legend()

    axes2[1].plot(h_dm_d, label='Dense', color='red', alpha=0.5)
    axes2[1].plot(h_dm_s, label='Sparse', color='green', alpha=0.8)
    axes2[1].set_title('Data Movement'); axes2[1].set_ylabel('MB/Batch'); axes2[1].legend()

    axes2[2].plot(h_bits_d, label='Dense', color='red', alpha=0.5)
    axes2[2].plot(h_bits_s, label='Sparse', color='green', alpha=0.8)
    axes2[2].set_title('Transfer Bits'); axes2[2].set_ylabel('Gb/Batch'); axes2[2].legend()

    for ax in axes2: ax.grid(True, alpha=0.3); ax.set_xlabel('Batch Iterations')
    plt.tight_layout(); plt.savefig(cost_plot_path); plt.close()
    
    print(f"Plots saved to {cost_plot_path}")
    print(f"\nRun complete. Results saved to {log_path}")

if __name__ == "__main__":
    cfg = DiffPCConfig(
        layer_dims=[784, 400, 10],
        lt_m=0,
        lt_n=5,
        lt_a=1.0,
        lt_scheduler_type="cyclic_phase",
        gamma_value=0.05,
        gamma_every_n=None,
        t_init_cycles=15,
        phase2_cycles=15,
        pc_lr=0.0001,
        batch_size=256,
        epochs=20,
        use_adamw=True,
        adamw_weight_decay=0.01,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-08,
        clip_grad_norm=1.0,
        seed=2,
        run_name="mnist_400h",
        use_fashion_mnist=False,
        random_crop_padding=2,
        normalize=True,
        fmnist_hflip_p=0.0,
        device="cuda:0" # Adjust device ID if necessary
    )
    main(cfg)