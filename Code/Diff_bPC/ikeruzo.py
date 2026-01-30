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
from torchvision.utils import save_image  # 追加: 画像保存用


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
    alpha_disc: float = 1.0; alpha_gen: float = 1e-4
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
        self.x_F_disc, self.x_F_gen, self.x_T, self.x_A, self.e_T_disc, self.e_T_gen, \
        self.e_A_disc, self.e_A_gen, self.s_A, self.s_e_disc, self.s_e_gen, \
        self.s_in_disc, self.s_in_gen, self.e_in_disc, self.e_in_gen, self.data = [None] * 16
        self.l_t, self.y, self.time_step = 0.0, 0.0, 1
        self.reset_state, self.reset_state_type = True, "zero"
        # prev-tick buffers
        self.s_A_prev, self.s_e_disc_prev, self.s_e_gen = None, None, None

    def _alloc(self, B: int):
        z = torch.zeros(B, self.dim, device=self.device)
        self.x_F_disc, self.x_F_gen, self.x_T, self.x_A, self.e_T_disc, self.e_T_gen, \
        self.e_A_disc, self.e_A_gen, self.s_A, self.s_e_disc, self.s_e_gen, \
        self.s_in_disc, self.s_in_gen, self.e_in_disc, self.e_in_gen, self.data = [z.clone() for _ in range(16)]
        # prev-tick buffers
        self.s_A_prev = torch.zeros_like(z)
        self.s_e_disc_prev = torch.zeros_like(z)
        self.s_e_gen_prev = torch.zeros_like(z)

    def _sample_step(self) -> int: return (self.time_step - 1) % self.sampling_duration

    def reset_states_if_needed(self, clamp_status: bool, sample_step: int):
        if (sample_step == 0) and self.reset_state:
            self.x_F_disc.zero_(); self.x_F_disc.zero_(); self.x_A.zero_()
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
        
        l_t_cur  = self.l_t_scheduler.get_l_t(sample_step,     self.learning_weights)
        l_t_prev = self.l_t_scheduler.get_l_t(sample_step - 1, self.learning_weights)
        self.l_t = l_t_cur
        self.y   = self.y_scheduler.get_y(sample_step, self.learning_weights)

        self.x_F_disc.add_(self.s_in_disc * l_t_prev)
        self.x_F_gen.add_(self.s_in_gen * l_t_prev)
        self.e_T_disc = self.alpha_disc * (self.x_T - self.x_F_disc)
        self.e_T_gen = self.alpha_gen * (self.x_T - self.x_F_gen)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration and bottomup_mask:
            self.x_F_disc.zero_()
            self.e_T_disc.zero_()
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration and topdown_mask:
            self.x_F_gen.zero_()
            self.e_T_gen.zero_()

        self.e_in_disc.add_(e_in_disc.to(self.device) * l_t_prev)
        self.e_in_gen.add_(e_in_gen.to(self.device) * l_t_prev)

        if not clamp_status:
            self.x_T.add_(self.y * (-self.e_T_disc - self.e_T_gen + (self.x_T > 0).float() * (self.alpha_disc * self.e_in_disc + self.alpha_gen * self.e_in_gen)))
        
        diff_act = self.x_T - self.x_A
        s_A_new = torch.sign(diff_act) * (diff_act.abs() > self.l_t)
        s_A_new = s_A_new * ((self.x_A + s_A_new * self.l_t) > 0.0)
        self.x_A.add_(s_A_new * self.l_t); self.s_A = s_A_new
        
        diff_disc_err = self.e_T_disc - self.e_A_disc
        diff_gen_err = self.e_T_gen - self.e_A_gen
        s_e_disc_new = torch.sign(diff_disc_err) * (diff_disc_err.abs() > self.l_t)
        s_e_gen_new = torch.sign(diff_gen_err) * (diff_gen_err.abs() > self.l_t)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration:
            s_e_disc_new.zero_()
            s_e_gen_new.zero_()

        self.e_A_disc.add_(s_e_disc_new * self.l_t)
        self.e_A_gen.add_(s_e_gen_new * self.l_t)
        self.s_e_disc = s_e_disc_new
        self.s_e_gen = s_e_gen_new

        # publish for next tick
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
        layer_args = {"sampling_duration": 1, "learning_weights": True, "training": True,
                      "l_t_scheduler": l_t_sched, "y_scheduler": y_sched, 
                      "alpha_disc": cfg.alpha_disc, "alpha_gen": cfg.alpha_gen, "device": self.device}
        self.layers = nn.ModuleList([DiffPCLayerTorch(dim=d, **layer_args) for d in cfg.layer_dims[1:]])
        self.input_driver = DiffPCLayerTorch(dim=cfg.layer_dims[0], **layer_args)

        self.input_driver_clamp, self.input_driver_data = False, None
        self._clamp_switch = [False] * len(self.layers)
        self._data_bucket = [None] * len(self.layers)
        
        self._global_step = 0
        # v2 bias: fire once per run
        self._bias_fired_once = False

        self._layer_dims: List[int] = list(cfg.layer_dims[1:])

    def reset_all_states(self, batch_size: int):
        for lyr in [self.input_driver, *self.layers]:
            lyr._alloc(batch_size); lyr.time_step = 1
        self._global_step = 0
        # runner controls _bias_fired_once
    
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

        # v2 bias: one-time at t == 1
        emit_bias = (not self._bias_fired_once) and (t == 1)
        if emit_bias:
            bias_spike = 1.0
            self._bias_fired_once = True
        else:
            bias_spike = 0.0

        # consume previous-tick spikes
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

            self.W[l].grad = - (post_eT_disc.T @ torch.relu(pre_xT_disc)) / Bf
            self.W_bias[l].grad = - post_eT_disc.sum(dim=0, keepdim=True).T / Bf
            self.V[l].grad = - (post_eT_gen.T @ torch.relu(pre_xT_gen)) / Bf
            self.V_bias[l].grad = - post_eT_gen.sum(dim=0, keepdim=True).T / Bf

        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.W) + list(self.W_bias) + list(self.V) + list(self.V_bias), self.cfg.clip_grad_norm)
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

    # Phase-1
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    net.set_training(False)                       # v2: no dropout in P1

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    for _ in range(steps_phase1): net.one_time_step(bottomup_mask=False, topdown_mask=True)

    # Phase-2 (continue state)
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)

    net.set_training(True)                      # v2: nn.Dropout active now

    net.set_clamp(len(net.layers), True, y_onehot)
    
    spike_stats = SpikeStats()
    for _ in range(steps_phase2):
        net.one_time_step(bottomup_mask=False, topdown_mask=False)
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_disc_total += (lyr.s_e_disc != 0).sum().item()
            spike_stats.se_gen_total += (lyr.s_e_gen != 0).sum().item()
    
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

    net.set_training(False)                 # v2: no dropout in inference

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
def run_batch_disc_test_two_phase(net: DiffPCNetworkTorch, x: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
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

    net.set_training(False)                       # v2: no dropout in P1

    net.set_clamp(0, True, x)
    for li in range(1, len(net.layers) + 1): net.set_clamp(li, False)
    for _ in range(steps_phase1): net.one_time_step(bottomup_mask=False, topdown_mask=True)

    # Phase-2 (continue state)
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

    return net.layers[-1].x_T.clone(), spike_stats

@torch.no_grad()
def run_batch_gen_test_two_phase(net: DiffPCNetworkTorch, y_onehot: torch.Tensor,
                        cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
    """
    指定されたラベル(y_onehot)に基づいて画像を生成するランナー。
    Phase 1: ラベルを固定、入力層・中間層を自由にしてトップダウン信号で状態を遷移。
    Phase 2: 安定化のため継続実行 (学習時と同様のダイナミクスを担保)。
    """
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(y_onehot.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])

    # Phase-1: Generation (Top-down settling)
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    net.set_training(False)

    # 変更点: ラベル層(最上位)を固定し、入力層(input_driver)を含む他層をunclampする
    net.set_clamp(len(net.layers), True, y_onehot)
    for li in range(0, len(net.layers)): net.set_clamp(li, False)
    
    # Phase 1: トップダウンで夢を見させる (bottomup_mask=True でボトムアップ信号を切る設定が一般的)
    for _ in range(steps_phase1): net.one_time_step(bottomup_mask=True, topdown_mask=False)

    # Phase-2 (continue state)
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

    # 変更点: 生成された画像データ(input_driverの状態)を返す
    return net.input_driver.x_T.clone(), spike_stats


# =========================
#  Visualization Wrapper
# =========================

def visualize_generated_digits(net: DiffPCNetworkTorch, cfg: DiffPCConfig, 
                               l_t_spec: dict, y_phase2_spec: dict, 
                               epoch: int, output_dir: str = "gen_results"):
    """
    0から9までの数字ラベルをネットワークに与え、生成された画像を保存します。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = net.device
    
    # 0~9のラベルを作成
    labels = torch.arange(10, device=device)
    # One-hotエンコーディング
    y_onehot = F.one_hot(labels, num_classes=cfg.layer_dims[-1]).float()
    
    # 生成を実行 (Batch=10, Dim=784)
    generated_data, _ = run_batch_gen_test_two_phase(net, y_onehot, cfg, l_t_spec, y_phase2_spec)
    
    # 画像形状にリシェイプ (Batch, 1, 28, 28)
    gen_imgs = generated_data.view(10, 1, 28, 28)
    
    # 正規化の解除 (Denormalize)
    if cfg.normalize:
        if cfg.use_fashion_mnist:
            mean, std = 0.2860, 0.3530
        else:
            mean, std = 0.1307, 0.3081
        gen_imgs = gen_imgs * std + mean
    
    # 範囲を[0, 1]にクリップ
    gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0)
    
    # 画像を保存
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_generated.png")
    save_image(gen_imgs, save_path, nrow=10)
    print(f"Generated images saved to: {save_path}")


# =========================
#  MNIST training wrapper
# =========================

def main(cfg: DiffPCConfig):
    # Setup
    torch.manual_seed(cfg.seed)
    
    requested = str(cfg.device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                device = torch.device(requested)
                if device.index is not None and device.index >= torch.cuda.device_count():
                    print(f"Warning: Requested {requested} but only {torch.cuda.device_count()} CUDA device(s) available. Using cuda:0.")
                    device = torch.device("cuda:0")
                torch.cuda.set_device(device)
            except Exception as e:
                print(f"Warning: Could not use '{requested}' ({e}). Falling back to cuda:0.")
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
        else:
            print("Warning: CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    run_name = cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")
    
    # 生成画像の保存先ディレクトリ
    gen_dir = os.path.join(log_dir, "generated_images")

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

    def to_vec(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).to(device)
    def to_onehot(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=cfg.layer_dims[-1]).float().to(device)

    run_results = []
    total_neurons = sum(cfg.layer_dims[1:])

    print(f"Starting run: {run_name}")
    print(f"Using device: {device}")
    print(f"Logging to: {log_path}")

    for epoch in range(1, cfg.epochs + 1):
        # Training
        total_sa_train_p2, total_se_disc_train_p2, total_se_gen_train_p2 = 0.0, 0.0, 0.0

        for images, labels in train_loader:
            _, stats = run_batch_two_phase(net, to_vec(images), to_onehot(labels), cfg, l_t_spec, y_phase2_spec)
            total_sa_train_p2 += stats.sa_total
            total_se_disc_train_p2 += stats.se_disc_total
            total_se_gen_train_p2 += stats.se_gen_total

        # Evaluation
        train_correct, train_total = 0, 0
        test_correct, test_total = 0, 0
        total_sa_test_p1, total_se_disc_test_p1, total_se_gen_test_p1 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, labels in train_loader:
                # 修正: 元のコードで未定義だった関数名を修正 (run_batch_disc_test_two_phase)
                logits, _ = run_batch_disc_test_two_phase(net, to_vec(images), cfg, l_t_spec, y_phase2_spec)
                preds = logits.argmax(dim=1).cpu()
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)

            for images, labels in test_loader:
                # 修正: 元のコードで未定義だった関数名を修正
                logits, stats = run_batch_disc_test_two_phase(net, to_vec(images), cfg, l_t_spec, y_phase2_spec)
                preds = logits.argmax(dim=1).cpu()
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)
                total_sa_test_p1 += stats.sa_total
                total_se_disc_test_p1 += stats.se_disc_total
                total_se_gen_test_p1 += stats.se_gen_total

        train_acc = 100.0 * train_correct / train_total
        test_acc  = 100.0 * test_correct  / test_total

        denom_train = total_neurons * len(train_ds)
        denom_test  = total_neurons * len(test_ds)
        avg_sa_train_p2 = total_sa_train_p2 / denom_train if denom_train > 0 else 0
        avg_se_disc_train_p2 = total_se_disc_train_p2 / denom_train if denom_train > 0 else 0
        avg_se_gen_train_p2 = total_se_gen_train_p2 / denom_train if denom_train > 0 else 0
        avg_sa_test_p1  = total_sa_test_p1  / denom_test  if denom_test  > 0 else 0
        avg_se_disc_test_p1  = total_se_disc_test_p1  / denom_test  if denom_test  > 0 else 0
        avg_se_gen_test_p1  = total_se_gen_test_p1  / denom_test  if denom_test  > 0 else 0

        print(
            f"Epoch {epoch:02d}: train acc {train_acc:.2f}% | test acc {test_acc:.2f}% | "
            f"Avg Spikes/N (Train P2): s_a={avg_sa_train_p2:.2f}, s_e_disc={avg_se_disc_train_p2:.2f} , s_e_gen={avg_se_gen_train_p2:.2f}| "
            f"Avg Spikes/N (Test P1):  s_a={avg_sa_test_p1:.2f}, s_e_disc={avg_se_disc_test_p1:.2f}, s_e_gen={avg_se_gen_test_p1:.2f}"
        )

        run_results.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "avg_spikes_per_neuron_train_p2_sa": avg_sa_train_p2,
            "avg_spikes_per_neuron_train_p2_se_disc": avg_se_disc_train_p2,
            "avg_spikes_per_neuron_train_p2_se_gen": avg_se_gen_train_p2,
            "avg_spikes_per_neuron_test_p1_sa": avg_sa_test_p1,
            "avg_spikes_per_neuron_test_p1_se_disc": avg_se_disc_test_p1,
            "avg_spikes_per_neuron_test_p1_se_gen": avg_se_gen_test_p1,
        })

        if epoch % 5 == 0:
            # 画像生成の可視化
            print(f"Generating digits for epoch {epoch}...")
            visualize_generated_digits(net, cfg, l_t_spec, y_phase2_spec, epoch, output_dir=gen_dir)

    # Save
    with open(log_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": run_results}, f, indent=4)

    print(f"\nRun complete. Results saved to {log_path}")

# =========================
#  Sanity Check Function
# =========================

def debug_lt(cfg: DiffPCConfig, seq_len: Optional[int] = None):
    if seq_len is None:
        seq_len = 2 * cfg.lt_n
    print("\n--- Running l_t Scheduler Sanity Check ---")
    print(f"Using scheduler: {cfg.lt_scheduler_type}")
    print(f"Using config: lt_n={cfg.lt_n}, lt_m={cfg.lt_m}, lt_a={cfg.lt_a}")
    sched = get_l_t_scheduler(cfg.lt_scheduler_type, {"m": cfg.lt_m, "n": cfg.lt_n, "a": cfg.lt_a})
    sched.begin_phase(0, seq_len * 5, a=cfg.lt_a)
    
    print("This shows the l_t value for the *previous* tick (t-1) and the *current* tick (t).")
    print("The one-tick pipeline correctly uses get_l_t(t-1) to scale inputs arriving at t.")
    
    for t in range(seq_len):
        lt_now = sched.get_l_t(t, True)
        lt_prev_correct = sched.get_l_t(t - 1, True) if t > 0 else 0.0
        
        print(f"t={t:02d} | "
              f"l_t(t-1) [used for input]: {lt_prev_correct: >9.6f} | "
              f"l_t(t) [used for update]: {lt_now: >9.6f}")
    print("--- End of Sanity Check ---\n")


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
        phase2_cycles=20,
        alpha_disc = 1,
        alpha_gen = 0.0225,
        pc_lr=0.0001,
        batch_size=256,
        epochs=25,
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