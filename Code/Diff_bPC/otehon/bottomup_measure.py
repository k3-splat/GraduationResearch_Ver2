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
    
    # Threshold minimums
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
    """Container for stats."""
    sa_total: float = 0.0
    se_total: float = 0.0
    
    # Costs
    flops_dense: float = 0.0
    flops_sparse: float = 0.0
    dm_dense: float = 0.0
    dm_sparse: float = 0.0
    bits_dense: float = 0.0
    bits_sparse: float = 0.0

    # New Metric: Negative Activity Ratio in Hidden Layers
    neg_activity_ratio: float = 0.0 

# =========================
#  Schedulers
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

def get_l_t_scheduler(scheduler_type: str, scheduler_args: dict) -> LTBaseScheduler:
    if scheduler_type == "cyclic_phase": return LTCyclicPhase(**scheduler_args)
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
#  Layer & Network
# =========================

class DiffPCLayerTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.device = torch.device(self.device)
        self.x_F, self.x_T, self.x_A, self.e_T, self.e_A, self.s_A, self.s_e, \
        self.s_in_data, self.e_in_data, self.data = [None] * 10
        self.l_t, self.y, self.time_step = 0.0, 0.0, 1
        self.reset_state = True
        self.s_A_prev, self.s_e_prev = None, None

        if not hasattr(self, 'lt_min_a'): self.lt_min_a = 0.0
        if not hasattr(self, 'lt_min_e_disc'): self.lt_min_e_disc = 0.0
        if not hasattr(self, 'lt_min_e_gen'): self.lt_min_e_gen = 0.0

    def _alloc(self, B: int):
        z = torch.zeros(B, self.dim, device=self.device)
        self.x_F, self.x_T, self.x_A, self.e_T, self.e_A, self.s_A, self.s_e, \
        self.s_in_data, self.e_in_data, self.data = [z.clone() for _ in range(10)]
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

        # Integration
        l_t_integration = max(l_t_prev, self.lt_min_a)
        self.x_F.add_(self.s_in_data * l_t_integration)
        self.e_T = self.x_T - self.x_F
        self.e_in_data.add_(e_in_recv.to(self.device) * l_t_integration)

        if not clamp_status:
            self.x_T.add_(self.y * (-self.alpha_disc * self.e_T + (self.x_T > 0).float() * self.alpha_disc * self.e_in_data))
        
        # Spiking (Forward)
        eff_lt_a = max(l_t_cur, self.lt_min_a)
        diff_act = self.x_T - self.x_A
        s_A_new = torch.sign(diff_act) * (diff_act.abs() > eff_lt_a)
        s_A_new = s_A_new * ((self.x_A + s_A_new * eff_lt_a) > 0.0)
        self.x_A.add_(s_A_new * eff_lt_a); self.s_A = s_A_new
        
        # Spiking (Error)
        eff_lt_err = max(l_t_cur, self.lt_min_e_disc)
        diff_err = self.e_T - self.e_A
        s_e_new = torch.sign(diff_err) * (diff_err.abs() > eff_lt_err)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration: s_e_new.zero_()
        self.e_A.add_(s_e_new * eff_lt_err); self.s_e = s_e_new

        if self.s_A_prev is None: self.s_A_prev = torch.zeros_like(self.s_A)
        if self.s_e_prev is None: self.s_e_prev = torch.zeros_like(self.s_e)
        self.s_A_prev.copy_(self.s_A)
        self.s_e_prev.copy_(self.s_e)

        if sample_step_override is not None: self.time_step = int(sample_step_override) + 1
        else: self.time_step += 1

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
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=cfg.dropout_rate) for _ in range(len(self.W))])

        for w in self.W: nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        if cfg.use_adamw:
            self.optimizer = torch.optim.AdamW(list(self.W)+list(self.W_bias), lr=cfg.pc_lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay)
        else:
            self.optimizer = torch.optim.Adam(list(self.W)+list(self.W_bias), lr=cfg.pc_lr, betas=cfg.adamw_betas, eps=cfg.adamw_eps)

        l_t_sched = get_l_t_scheduler(l_t_scheduler_spec["type"], l_t_scheduler_spec["args"])
        y_sched = get_y_scheduler(y_scheduler_spec["type"], {**y_scheduler_spec["args"], "l_t_scheduler": l_t_sched})
        layer_args = {
            "sampling_duration": 1, "learning_weights": True, "training": True,
            "l_t_scheduler": l_t_sched, "y_scheduler": y_sched, "device": self.device,
            "lt_min_a": cfg.lt_min_a, "lt_min_e_disc": cfg.lt_min_e_disc, "lt_min_e_gen": cfg.lt_min_e_gen,
            "alpha_disc": cfg.alpha_disc # Using alpha_disc for scaling updates if needed
        }
        self.layers = nn.ModuleList([DiffPCLayerTorch(dim=d, **layer_args) for d in cfg.layer_dims[1:]])
        self.input_driver = DiffPCLayerTorch(dim=cfg.layer_dims[0], **layer_args)

        self.input_driver_clamp, self.input_driver_data = False, None
        self._clamp_switch = [False] * len(self.layers)
        self._data_bucket = [None] * len(self.layers)
        self._global_step = 0
        self._bias_fired_once = False
        self._v1_dropout_active = False
        self._v1_drop_masks = [None] * len(self.layers)
        self._layer_dims = list(cfg.layer_dims[1:])
        self._metrics = {'flops_dense': 0.0, 'flops_sparse': 0.0, 'dm_dense': 0.0, 'dm_sparse': 0.0, 'bits_dense': 0.0, 'bits_sparse': 0.0}

    def reset_all_states(self, batch_size: int):
        for lyr in [self.input_driver, *self.layers]: lyr._alloc(batch_size); lyr.time_step = 1
        self._global_step = 0
    
    def set_training(self, flag: bool):
        self.train(flag)
        for lyr in [self.input_driver, *self.layers]: lyr.training = bool(flag)

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
            if i == len(self.layers) - 1: self._v1_drop_masks[i] = None
            else: self._v1_drop_masks[i] = torch.bernoulli(torch.full((batch_size, self._layer_dims[i]), keep_prob, device=self.device))

    def swap_schedulers(self, l_t_sched: LTBaseScheduler, y_sched: YBaseScheduler):
        for lyr in [self.input_driver, *self.layers]: lyr.l_t_scheduler = l_t_sched; lyr.y_scheduler = y_sched

    def set_sampling_duration(self, new_len: int):
        for lyr in [self.input_driver, *self.layers]: lyr.sampling_duration = new_len
        self._global_step = 0

    def set_clamp(self, layer_index: int, clamp: bool, data: Optional[torch.Tensor] = None):
        if layer_index == 0: self.input_driver_clamp, self.input_driver_data = clamp, data
        else: self._clamp_switch[layer_index-1], self._data_bucket[layer_index-1] = clamp, data

    def _calc_matrix_ops_cost(self, input_tensor: torch.Tensor, dim_out: int, is_bias: bool = False):
        batch_size, dim_in = input_tensor.shape
        num_active = (input_tensor.abs() > 1e-6).sum().item()
        f_dense = batch_size * dim_in * dim_out * (1.0 if is_bias else 2.0)
        f_sparse = num_active * dim_out * (1.0 if is_bias else 2.0)
        dm_dense = (dim_in * dim_out * 4.0) + ((batch_size * dim_in + batch_size * dim_out) * 4.0)
        dm_sparse = (num_active * dim_out * 4.0) + ((num_active + batch_size * dim_out) * 4.0)
        return f_dense, f_sparse, dm_dense, dm_sparse

    @torch.no_grad()
    def _accumulate_transfer_bits(self):
        # Collect: s_A_prev (Forward) and s_e_prev (Backward)
        communicating_tensors = [self.input_driver.s_A_prev]
        for lyr in self.layers:
            communicating_tensors.append(lyr.s_A_prev)
            communicating_tensors.append(lyr.s_e_prev)
        for x in communicating_tensors:
            if x is None: continue
            num_elements = x.numel()
            n_neurons = x.shape[1]
            num_active = (x.abs() > 1e-6).sum().item()
            self._metrics['bits_dense'] += num_elements * 32
            addr_bits = math.ceil(math.log2(n_neurons)) if n_neurons > 1 else 1
            self._metrics['bits_sparse'] += num_active * (32 + addr_bits)

    @torch.no_grad()
    def one_time_step(self):
        B = self.input_driver.x_F.size(0)
        z = torch.zeros(B, self.input_driver.dim, device=self.device)
        sd = self.input_driver.sampling_duration
        t  = self._global_step % sd
        self._metrics = {k: 0.0 for k in self._metrics}

        emit_bias = (not self._bias_fired_once) and (t == 1)
        if emit_bias: bias_spike = 1.0; self._bias_fired_once = True
        else: bias_spike = 0.0

        # Forward
        s_in = []
        prev_s = self.input_driver.s_A_prev
        for l, w in enumerate(self.W):
            din, dout = w.shape[1], w.shape[0]
            fd, fs, dmd, dms = self._calc_matrix_ops_cost(prev_s, dout)
            self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
            self._metrics['dm_dense'] += dmd; self._metrics['dm_sparse'] += dms
            s_in_l = prev_s @ w.T
            if not self.cfg.v1_dropout and self.training and l < len(self.W) - 1: s_in_l = self.dropout_layers[l](s_in_l)
            if bias_spike != 0.0:
                s_in_l.add_(self.W_bias[l].T)
                self._metrics['flops_dense'] += B * dout; self._metrics['flops_sparse'] += B * dout
            s_in.append(s_in_l)
            prev_s = self.layers[l].s_A_prev

        # Backward (Error)
        e_in = []
        for l in range(len(self.layers)):
            if l < len(self.layers) - 1:
                w_next = self.W[l+1]
                din, dout = w_next.shape[1], w_next.shape[0] # W is (Dout, Din)
                feedback_input = self.layers[l+1].s_e_prev
                fd, fs, dmd, dms = self._calc_matrix_ops_cost(feedback_input, din)
                self._metrics['flops_dense'] += fd; self._metrics['flops_sparse'] += fs
                self._metrics['dm_dense'] += dmd; self._metrics['dm_sparse'] += dms
                
                e_in_l = feedback_input @ w_next
                if self.cfg.v1_dropout and self._v1_dropout_active and self._v1_drop_masks[l+1] is not None:
                    e_in_l = (self.layers[l+1].s_e_prev * self._v1_drop_masks[l+1]) @ w_next
            else:
                e_in_l = torch.zeros_like(self.layers[l].s_e_prev)
            e_in.append(e_in_l)

        # Comm Cost
        self._accumulate_transfer_bits()

        # Update Driver
        self.input_driver.step(self.input_driver_clamp, self.input_driver_data, z, z, sample_step_override=t)
        
        # Update Layers
        for i, lyr in enumerate(self.layers):
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
                if l > 0 and self._v1_drop_masks[l-1] is not None: pre_xT = pre_xT * self._v1_drop_masks[l-1]
                if self._v1_drop_masks[l] is not None: post_eT = post_eT * self._v1_drop_masks[l]
            self.W[l].grad = - (post_eT.T @ torch.relu(pre_xT)) / Bf
            self.W_bias[l].grad = - post_eT.sum(dim=0, keepdim=True).T / Bf
        if self.cfg.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(list(self.W) + list(self.W_bias), self.cfg.clip_grad_norm)
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
    l_t_sched.begin_phase(0, steps_phase1, cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)
    if cfg.v1_dropout: net.set_training(False); net.set_v1_dropout(True, x.size(0), redraw=True)
    else: net.set_training(False)
    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    for _ in range(steps_phase1): 
        net.one_time_step()
        spike_stats.flops_dense += net._metrics['flops_dense']; spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']; spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']; spike_stats.bits_sparse += net._metrics['bits_sparse']

    # Phase-2
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(steps_phase1, steps_phase2, cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)
    if cfg.v1_dropout: net.set_training(True); net.set_v1_dropout(True, x.size(0), redraw=False)
    else: net.set_training(True)
    net.set_clamp(len(net.layers), True, y_onehot)
    
    for _ in range(steps_phase2):
        net.one_time_step()
        spike_stats.flops_dense += net._metrics['flops_dense']; spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']; spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']; spike_stats.bits_sparse += net._metrics['bits_sparse']
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()

    # --- NEW: Calculate Negative Activity Ratio in Hidden Layers ---
    total_hidden_neurons = 0
    total_neg_hidden = 0
    for lyr in net.layers:
        # Check negative x_T values
        total_neg_hidden += (lyr.x_T < 0).sum().item()
        total_hidden_neurons += lyr.x_T.numel()
    
    if total_hidden_neurons > 0:
        spike_stats.neg_activity_ratio = total_neg_hidden / total_hidden_neurons
    # ----------------------------------------------------------------

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
    l_t_sched.begin_phase(0, steps, cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_forward)
    if cfg.v1_dropout: net.set_training(False); net.set_v1_dropout(False, x.size(0))
    else: net.set_training(False)
    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    
    spike_stats = SpikeStats()
    for _ in range(steps):
        net.one_time_step()
        spike_stats.flops_dense += net._metrics['flops_dense']; spike_stats.flops_sparse += net._metrics['flops_sparse']
        spike_stats.dm_dense += net._metrics['dm_dense']; spike_stats.dm_sparse += net._metrics['dm_sparse']
        spike_stats.bits_dense += net._metrics['bits_dense']; spike_stats.bits_sparse += net._metrics['bits_sparse']
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()
    return net.layers[-1].x_T.clone(), spike_stats

# =========================
#  Visualization Tools
# =========================

def visualize_generated_digits(net: DiffPCNetworkTorch, cfg: DiffPCConfig, l_t_spec: dict, 
                               epoch: int, output_dir: str):
    """
    Generates digits by clamping the label layer and running backward inference.
    (Note: This requires a backward inference runner. We create a simple one here.)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = net.device
    labels = torch.arange(10, device=device)
    y_onehot = F.one_hot(labels, num_classes=cfg.layer_dims[-1]).float()
    
    # Minimal Backward Inference Logic
    lt_n = l_t_spec["args"]["n"]
    steps = cfg.t_init_cycles * lt_n
    net.set_sampling_duration(steps)
    net.reset_all_states(10)
    net._bias_fired_once = False
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])
    y_sched = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(0, steps, cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_sched)
    net.set_training(False)
    
    # Clamp Label (Output Layer), Unclamp Input
    net.set_clamp(len(net.layers), True, y_onehot) 
    for li in range(len(net.layers)): net.set_clamp(li, False)
    
    for _ in range(steps):
        net.one_time_step()
    
    gen_imgs = net.input_driver.x_T.clone().view(10, 1, 28, 28)
    
    if cfg.normalize:
        if cfg.use_fashion_mnist: m, s = 0.2860, 0.3530
        else: m, s = 0.1307, 0.3081
        gen_imgs = gen_imgs * s + m
    
    gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0)
    save_path = os.path.join(output_dir, f"gen_epoch_{epoch:03d}.png")
    save_image(gen_imgs, save_path, nrow=10)

def plot_metrics(h_flops, h_dm, h_bits, h_neg_ratio, output_dir: str):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].plot(h_flops['dense'], label='Dense', color='red', alpha=0.5)
    axes[0].plot(h_flops['sparse'], label='Sparse', color='green', alpha=0.8)
    axes[0].set_title('FLOPs (G)'); axes[0].legend()

    axes[1].plot(h_dm['dense'], label='Dense', color='red', alpha=0.5)
    axes[1].plot(h_dm['sparse'], label='Sparse', color='green', alpha=0.8)
    axes[1].set_title('Data Mvmt (MB)'); axes[1].legend()

    axes[2].plot(h_bits['dense'], label='Dense', color='red', alpha=0.5)
    axes[2].plot(h_bits['sparse'], label='Sparse', color='green', alpha=0.8)
    axes[2].set_title('Comm (Gb)'); axes[2].legend()

    axes[3].plot(h_neg_ratio, label='Neg Ratio', color='blue')
    axes[3].set_title('Hidden x_T < 0 Ratio'); axes[3].legend()

    for ax in axes: ax.grid(True, alpha=0.3); ax.set_xlabel('Batch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_plot.png"))
    plt.close()


# =========================
#  Main
# =========================

def main(cfg: DiffPCConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # --- Unique Directory Setup ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{cfg.run_name or 'run'}_{timestamp}"
    current_run_dir = os.path.join("runs", run_name)
    os.makedirs(current_run_dir, exist_ok=True)
    log_path = os.path.join(current_run_dir, "results.json")
    print(f"Starting run: {run_name}\nSaving results to: {current_run_dir}")

    # Network Setup
    l_t_spec = {"type": cfg.lt_scheduler_type, "args": {"m": cfg.lt_m, "n": cfg.lt_n, "a": cfg.lt_a}}
    y_phase2_spec = {"type": "on_cycle_start", "args": {"gamma": cfg.gamma_value, "n": (cfg.gamma_every_n or cfg.lt_n)}}
    net = DiffPCNetworkTorch(cfg, l_t_spec, y_phase2_spec, str(device))
    
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
    
    # History for plots (batch-wise tracking)
    h_flops = {'dense': [], 'sparse': []}
    h_dm = {'dense': [], 'sparse': []}
    h_bits = {'dense': [], 'sparse': []}
    h_neg_ratio = []

    # --- [変更点1] 累計カウンタの初期化 ---
    # 学習開始からの総コストを保持する変数
    cumulative_stats = SpikeStats() 
    # -----------------------------------

    for epoch in range(1, cfg.epochs + 1):
        ep_stats = SpikeStats() # そのエポック単体のコスト
        num_batches = 0
        total_neg_ratio_accum = 0.0

        # Train Loop
        for images, labels in train_loader:
            _, stats = run_batch_two_phase(net, to_vec(images), to_onehot(labels), cfg, l_t_spec, y_phase2_spec)
            
            # Aggregate Costs for Epoch
            ep_stats.flops_dense += stats.flops_dense
            ep_stats.flops_sparse += stats.flops_sparse
            ep_stats.dm_dense += stats.dm_dense
            ep_stats.dm_sparse += stats.dm_sparse
            ep_stats.bits_dense += stats.bits_dense
            ep_stats.bits_sparse += stats.bits_sparse
            
            # Aggregate Negative Ratio
            total_neg_ratio_accum += stats.neg_activity_ratio
            
            num_batches += 1

            # Append batch stats for plotting (Tracking per batch is kept as average for visualization if needed, or raw)
            h_flops['dense'].append(stats.flops_dense/1e9); h_flops['sparse'].append(stats.flops_sparse/1e9)
            h_dm['dense'].append(stats.dm_dense/(1024**2)); h_dm['sparse'].append(stats.dm_sparse/(1024**2))
            h_bits['dense'].append(stats.bits_dense/1e9); h_bits['sparse'].append(stats.bits_sparse/1e9)
            h_neg_ratio.append(stats.neg_activity_ratio)

        # --- [変更点2] 累計カウンタの更新 ---
        cumulative_stats.flops_dense += ep_stats.flops_dense
        cumulative_stats.flops_sparse += ep_stats.flops_sparse
        cumulative_stats.dm_dense += ep_stats.dm_dense
        cumulative_stats.dm_sparse += ep_stats.dm_sparse
        cumulative_stats.bits_dense += ep_stats.bits_dense
        cumulative_stats.bits_sparse += ep_stats.bits_sparse
        # -----------------------------------

        # Eval Loop
        train_correct, train_total = 0, 0
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                train_correct += (logits.argmax(dim=1).cpu() == labels).sum().item(); train_total += labels.size(0)
            for images, labels in test_loader:
                logits, _ = infer_batch_forward_only(net, to_vec(images), cfg, l_t_spec)
                test_correct += (logits.argmax(dim=1).cpu() == labels).sum().item(); test_total += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc  = 100.0 * test_correct  / test_total
        avg_neg_ratio = total_neg_ratio_accum / num_batches

        # --- [変更点3] 表示用・記録用の変数を「累計値」に変換 ---
        # 単位変換: Giga, MegaByte, GigaBit
        cum_flops_d_g = cumulative_stats.flops_dense / 1e9
        cum_flops_s_g = cumulative_stats.flops_sparse / 1e9
        
        cum_dm_d_mb   = cumulative_stats.dm_dense / (1024**2)
        cum_dm_s_mb   = cumulative_stats.dm_sparse / (1024**2)
        
        cum_bits_d_gb = cumulative_stats.bits_dense / 1e9
        cum_bits_s_gb = cumulative_stats.bits_sparse / 1e9

        print(
            f"Epoch {epoch:02d}: Tr {train_acc:.1f}% / Te {test_acc:.1f}% | "
            f"NegRatio: {avg_neg_ratio:.2%} | "
            f"Cum FLOPs(G): {cum_flops_s_g:.1f} | Cum DM(MB): {cum_dm_s_mb:.1f} | Cum Bits(Gb): {cum_bits_s_gb:.1f}"
        )

        visualize_generated_digits(net, cfg, l_t_spec, epoch, current_run_dir)

        # --- [変更点4] JSON保存データにDenseと累計値を追加 ---
        run_results.append({
            "epoch": epoch, 
            "train_acc": train_acc, 
            "test_acc": test_acc,
            "avg_neg_activity_ratio": avg_neg_ratio,
            
            # 累計コスト (Cumulative Costs)
            "cum_flops_dense_G": cum_flops_d_g,
            "cum_flops_sparse_G": cum_flops_s_g,
            
            "cum_dm_dense_MB": cum_dm_d_mb,
            "cum_dm_sparse_MB": cum_dm_s_mb,
            
            "cum_bits_dense_Gb": cum_bits_d_gb,
            "cum_bits_sparse_Gb": cum_bits_s_gb
        })

    # Save Results
    with open(log_path, "w") as f: json.dump({"config": asdict(cfg), "results": run_results}, f, indent=4)
    plot_metrics(h_flops, h_dm, h_bits, h_neg_ratio, current_run_dir)
    print(f"Run complete. Data saved to {current_run_dir}")

if __name__ == "__main__":
    cfg = DiffPCConfig(
        layer_dims=[784, 400, 10],
        lt_m=0, lt_n=5, lt_a=1.0,
        lt_scheduler_type="cyclic_phase",
        gamma_value=0.05,
        t_init_cycles=15, phase2_cycles=15,
        pc_lr=0.0001, batch_size=256, epochs=20,
        seed=2, run_name="mnist_400h_neg_monitor",
        dropout_rate=0.5, normalize=True,
        device="cuda:0"
    )
    main(cfg)