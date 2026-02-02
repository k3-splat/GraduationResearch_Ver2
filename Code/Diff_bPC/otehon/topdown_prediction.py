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


# =========================
#  Config & Stats
# =========================

@dataclass
class DiffPCConfig:
    """Central configuration for the Generative DiffPC model."""
    # CHANGED: Default dims reversed for Generative model (Label -> Hidden -> Image)
    layer_dims: list[int] = (10, 512, 784) 
    lt_m: int = 3; lt_n: int = 5; lt_a: float = 0.5
    lt_scheduler_type: str = "cyclic_phase"
    gamma_value: float = 0.2; gamma_every_n: Optional[int] = None
    t_init_cycles: int = 15; phase2_cycles: int = 15
    pc_lr: float = 5e-4
    batch_size: int = 256; epochs: int = 50
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
    sa_total: float = 0.0
    se_total: float = 0.0

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
        assert self.n > 0, "lt_n must be positive."
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
        if self.n is None: raise ValueError("YOnCycleStart requires 'n'.")

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
        self.x_F, self.x_T, self.x_A, self.e_T, self.e_A, self.s_A, self.s_e, \
        self.s_in_data, self.e_in_data, self.data = [None] * 10
        self.l_t, self.y, self.time_step = 0.0, 0.0, 1
        self.reset_state, self.reset_state_type = True, "zero"
        self.s_A_prev, self.s_e_prev = None, None

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

        # Forward Prediction Update (Top-down in GenPC)
        self.x_F.add_(self.s_in_data * l_t_prev)
        self.e_T = self.x_T - self.x_F
        
        # Error Input Accumulation (Bottom-up in GenPC)
        self.e_in_data.add_(e_in_recv.to(self.device) * l_t_prev)

        if not clamp_status:
            # Update Target Activity based on Prediction Error (Local) and Input Error (Bottom-up)
            self.x_T.add_(self.y * (-self.e_T + (self.x_T > 0).float() * self.e_in_data))
        
        diff_act = self.x_T - self.x_A
        s_A_new = torch.sign(diff_act) * (diff_act.abs() > self.l_t)
        s_A_new = s_A_new * ((self.x_A + s_A_new * self.l_t) > 0.0)
        self.x_A.add_(s_A_new * self.l_t); self.s_A = s_A_new
        
        diff_err = self.e_T - self.e_A
        s_e_new = torch.sign(diff_err) * (diff_err.abs() > self.l_t)
        if hasattr(self, 'ff_init_duration') and sample_step < self.ff_init_duration: s_e_new.zero_()
        self.e_A.add_(s_e_new * self.l_t); self.s_e = s_e_new

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


# =========================
#  Multi-layer network orchestrator
# =========================

class DiffPCNetworkTorch(nn.Module):
    def __init__(self, cfg: DiffPCConfig, l_t_scheduler_spec: Dict, y_scheduler_spec: Dict, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        # In GenPC: W projects Top-down. W[0] connects Layer 0 (Label) to Layer 1 (Hidden).
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
            for i in range(len(self._v1_drop_masks)):
                self._v1_drop_masks[i] = None
            return

        if self._v1_dropout_active and not redraw:
            return

        self._v1_dropout_active = True
        keep_prob = 1.0 - self.cfg.dropout_rate
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                self._v1_drop_masks[i] = None
            else:
                dim = self._layer_dims[i]
                self._v1_drop_masks[i] = torch.bernoulli(
                    torch.full((batch_size, dim), keep_prob, device=self.device)
                )

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
        s_in, e_in = [], []

        sd = self.layers[0].sampling_duration if len(self.layers) > 0 else 1
        t  = self._global_step % sd

        emit_bias = (not self._bias_fired_once) and (t == 1)
        if emit_bias:
            bias_spike = 1.0
            self._bias_fired_once = True
        else:
            bias_spike = 0.0

        prev_s = self.input_driver.s_A_prev
        for l, w in enumerate(self.W):
            s_in_l = prev_s @ w.T

            if self.cfg.v1_dropout:
                if self._v1_dropout_active:
                    pass
            else:
                if self.training and l < len(self.W) - 1:
                    s_in_l = self.dropout_layers[l](s_in_l)

            if bias_spike != 0.0:
                s_in_l.add_(self.W_bias[l].T)
            s_in.append(s_in_l)
            prev_s = self.layers[l].s_A_prev

        # Backward error propagation (Bottom-up in GenPC)
        e_in.append(self.layers[0].s_e_prev @ self.W[0]) # Error from Layer 1 -> Layer 0
        for l in range(len(self.layers)):
            if l < len(self.layers) - 1:
                e_in_l = self.layers[l+1].s_e_prev @ self.W[l+1]
                if self.cfg.v1_dropout:
                    if self._v1_dropout_active and self._v1_drop_masks[l+1] is not None:
                        e_in_l = (self.layers[l+1].s_e_prev * self._v1_drop_masks[l+1]) @ self.W[l+1]
            else:
                e_in_l = torch.zeros_like(self.layers[l].s_e_prev)
            e_in.append(e_in_l)

        return s_in, e_in

    @torch.no_grad()
    def one_time_step(self):
        B  = self.input_driver.x_F.size(0)
        z  = torch.zeros(B, self.input_driver.dim, device=self.device)
        sd = self.input_driver.sampling_duration
        t  = self._global_step % sd

        s_in, e_in = self._build_s_in_and_e_in()

        # Input driver (Layer 0) gets e_in[0] from Layer 1. Forward input (s_in) is assumed 0 (top-level).
        self.input_driver.step(self.input_driver_clamp, self.input_driver_data, z, e_in[0],
                               sample_step_override=t)
        
        for i, lyr in enumerate(self.layers):
            # Layer i+1 receives s_in[i] from Layer i and e_in[i+1] from Layer i+2 (or bottom if last)
            lyr.step(self._clamp_switch[i], self._data_bucket[i], s_in[i], e_in[i+1],
                     sample_step_override=t)
        
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
#  Runners (Generative Logic)
# =========================

@torch.no_grad()
def run_gen_training_batch(net: DiffPCNetworkTorch, label_onehot: torch.Tensor, image_flat: torch.Tensor,
                           cfg: DiffPCConfig, l_t_spec: dict, y_phase2_spec: dict) -> SpikeStats:
    """
    Train GenPC:
    - Input Driver (Layer 0) clamped to Label.
    - Last Layer (Layer N) clamped to Image.
    - Minimize error.
    """
    lt_n = l_t_spec["args"]["n"]
    steps_phase1 = cfg.t_init_cycles * lt_n
    steps_phase2 = cfg.phase2_cycles * lt_n
    
    net.set_sampling_duration(steps_phase1 + steps_phase2)
    net.reset_all_states(label_onehot.size(0))
    net._bias_fired_once = False
    for lyr in [net.input_driver, *net.layers]: lyr.ff_init_duration = steps_phase1
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])

    # Phase 1: Dreaming (Optional, but good for settling hidden states)
    # Clamp Label, Unclamp Image (Generate). 
    # NOTE: Standard DiffPC training often clamps both immediately in Phase 2 logic. 
    # Here we mimic the provided code: Phase 1 (Settle), Phase 2 (Update).
    # For GenPC, settling usually implies clamping the conditioning variable (Label).
    
    y_phase1 = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps_phase1, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase1)

    net.set_training(False)
    if cfg.v1_dropout: net.set_v1_dropout(True, label_onehot.size(0), redraw=True)

    # Clamp Label (Layer 0), Unclamp Image (Layer N) initially to let hidden states form from Top-down?
    # Or Clamp both? Standard PC clamps both to calculate errors.
    # Let's Clamp Label, Unclamp Hidden, Clamp Image for Phase 1 to let errors propagate.
    # Logic in Discriminative was: Clamp Input (Image), Unclamp Label.
    # Logic in Generative Training: Clamp Input (Label), Clamp Output (Image).
    
    net.set_clamp(0, True, label_onehot) # Label
    for li in range(1, len(net.layers)): 
        net.set_clamp(li, False) # Hidden
    net.set_clamp(len(net.layers), True, image_flat) # Image

    for _ in range(steps_phase1): net.one_time_step()

    # Phase 2: Learning
    y_phase2 = get_y_scheduler(y_phase2_spec["type"], {**y_phase2_spec["args"], "l_t_scheduler": l_t_sched})
    l_t_sched.begin_phase(phase_start_step=steps_phase1, phase_len=steps_phase2, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_phase2)

    net.set_training(True)
    if cfg.v1_dropout: net.set_v1_dropout(True, label_onehot.size(0), redraw=False)

    # Keep clamps
    net.set_clamp(0, True, label_onehot)
    net.set_clamp(len(net.layers), True, image_flat)
    
    spike_stats = SpikeStats()
    for _ in range(steps_phase2):
        net.one_time_step()
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()
    
    net.apply_phase2_update()
    return spike_stats

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
        net.one_time_step()
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()
            
    return net.layers[-1].x_T.clone(), spike_stats

@torch.no_grad()
def infer_classification_genpc(net: DiffPCNetworkTorch, image_flat: torch.Tensor, cfg: DiffPCConfig, l_t_spec: dict) -> tuple[torch.Tensor, SpikeStats]:
    """
    Inference in GenPC (Classification):
    - Clamp Image (Bottom).
    - Unclamp Label (Top).
    - Error propagates up, Label changes to minimize error.
    """
    lt_n = l_t_spec["args"]["n"]
    steps = (cfg.t_init_cycles + cfg.phase2_cycles) * lt_n # Give it enough time
    net.set_sampling_duration(steps)
    net.reset_all_states(image_flat.size(0))
    net._bias_fired_once = False
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])
    y_forward = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_forward)

    net.set_training(False)
    if cfg.v1_dropout: net.set_v1_dropout(False, image_flat.size(0))

    # Unclamp Label (Layer 0) - allow it to evolve
    # Note: reset_all_states zeroes it out. This acts as a neutral prior.
    net.set_clamp(0, False, None) 
    
    # Unclamp Hidden
    for li in range(1, len(net.layers)): net.set_clamp(li, False)
    
    # Clamp Image (Layer N)
    net.set_clamp(len(net.layers), True, image_flat)
    
    spike_stats = SpikeStats()
    for _ in range(steps):
        net.one_time_step()
        for lyr in net.layers:
            spike_stats.sa_total += (lyr.s_A != 0).sum().item()
            spike_stats.se_total += (lyr.s_e != 0).sum().item()
            
    # Return Layer 0 (Label) activity
    return net.input_driver.x_T.clone(), spike_stats

@torch.no_grad()
def generate_samples(net: DiffPCNetworkTorch, label_onehot: torch.Tensor, cfg: DiffPCConfig, l_t_spec: dict) -> torch.Tensor:
    """
    Generation Task:
    - Clamp Label (Top).
    - Unclamp Image (Bottom).
    - Network generates image.
    """
    lt_n = l_t_spec["args"]["n"]
    steps = cfg.t_init_cycles * lt_n
    net.set_sampling_duration(steps)
    net.reset_all_states(label_onehot.size(0))
    net._bias_fired_once = False
    
    l_t_sched = get_l_t_scheduler(l_t_spec["type"], l_t_spec["args"])
    y_forward = get_y_scheduler("on_cycle_start", {"l_t_scheduler": l_t_sched, "gamma": 1.0, "n": 1})
    l_t_sched.begin_phase(phase_start_step=0, phase_len=steps, a=cfg.lt_a)
    net.swap_schedulers(l_t_sched, y_forward)

    net.set_training(False)
    if cfg.v1_dropout: net.set_v1_dropout(False, label_onehot.size(0))

    # Clamp Label (Layer 0)
    net.set_clamp(0, True, label_onehot)
    
    # Unclamp rest (Hidden and Image)
    for li in range(1, len(net.layers) + 1): 
        net.set_clamp(li, False)
    
    for _ in range(steps):
        net.one_time_step()
            
    # Return Last Layer (Image) activity
    return net.layers[-1].x_T.clone()


# =========================
#  MNIST training wrapper
# =========================

def generate_mnist_categories(net: DiffPCNetworkTorch, cfg: DiffPCConfig, l_t_spec: dict, device: torch.device, epoch: int, log_dir: str):
    print("Generating category samples...")
    # Create one-hot vectors for 0-9
    labels = torch.arange(10, device=device)
    labels_onehot = F.one_hot(labels, num_classes=10).float()
    
    # Generate
    generated_flat, stats = infer_batch_forward_only(net, labels_onehot, cfg, l_t_spec)
    
    # Reshape and Save
    # MNIST/FMNIST is 28x28
    imgs = generated_flat.view(-1, 1, 28, 28)
    # Denormalize if needed (visual approximation)
    # Since network learns in normalized space, outputs might be outside [0,1].
    # We clip or normalize for visualization.
    imgs = torch.clamp(imgs, -1, 1) # assuming approx range
    imgs = (imgs + 1) / 2.0 # -1..1 -> 0..1
    
    save_path = os.path.join(log_dir, f"ggenerated_epoch_{epoch:03d}.png")
    save_image(imgs, save_path, nrow=5, normalize=False)
    print(f"Saved generated samples to {save_path}")


def main(cfg: DiffPCConfig):
    # Setup
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_name = cfg.run_name or f"genpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = "runs_genpc"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.json")

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
    
    if cfg.use_fashion_mnist:
        mean, std = 0.2860, 0.3530
    else:
        mean, std = 0.1307, 0.3081

    train_transforms_list = [transforms.ToTensor()]
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
        return F.one_hot(y, num_classes=cfg.layer_dims[0]).float().to(device)

    run_results = []
    total_neurons = sum(cfg.layer_dims[1:])

    print(f"Starting Generative PC run: {run_name}")
    print(f"Structure: Label(10) -> Hidden -> Image(784)")
    print(f"Device: {device}")

    for epoch in range(1, cfg.epochs + 1):
        # Training (Generative)
        total_sa, total_se = 0.0, 0.0
        
        for images, labels in train_loader:
            # Swap inputs: Label is "x" (Input), Image is "y" (Output)
            stats = run_gen_training_batch(net, to_onehot(labels), to_vec(images), cfg, l_t_spec, y_phase2_spec)
            total_sa += stats.sa_total
            total_se += stats.se_total

        # Inference (Classification) - Clamp Image, Infer Label
        # Note: GenPC classification is harder/slower than DiscPC.
        test_correct, test_total = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                logits, _ = infer_classification_genpc(net, to_vec(images), cfg, l_t_spec)
                preds = logits.argmax(dim=1).cpu()
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        test_acc  = 100.0 * test_correct  / test_total
        
        denom_train = total_neurons * len(train_ds)
        avg_sa = total_sa / denom_train if denom_train > 0 else 0
        avg_se = total_se / denom_train if denom_train > 0 else 0

        print(f"Epoch {epoch:02d}: Test Classification Acc {test_acc:.2f}% | "
              f"Train Spikes/N: sa={avg_sa:.2f}, se={avg_se:.2f}")

        # Visualize Generation
        if epoch % 1 == 0:
            generate_mnist_categories(net, cfg, l_t_spec, device, epoch, log_dir)

        run_results.append({
            "epoch": epoch,
            "test_acc": test_acc,
            "avg_sa": avg_sa,
            "avg_se": avg_se
        })

    with open(log_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": run_results}, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    cfg = DiffPCConfig(
        # Generative Direction: Label -> Hidden -> Image
        layer_dims=[10, 400, 784], 
        lt_m=0,
        lt_n=5,
        lt_a=1.0,
        lt_scheduler_type="cyclic_phase",
        gamma_value=0.1,
        gamma_every_n=None,
        t_init_cycles=10, # Give more time for errors to propagate up
        phase2_cycles=10,
        pc_lr=0.0005,
        batch_size=256,
        epochs=15,
        use_adamw=True,
        adamw_weight_decay=0.01,
        seed=42,
        run_name="genpc_mnist",
        use_fashion_mnist=False,
        normalize=True,
        device="cuda:0"
    )
    main(cfg)