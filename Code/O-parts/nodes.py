from typing import Iterable, Optional, Union
from bindsnet.network.nodes import LIFNodes

import torch

class bSpNCNNodes(LIFNodes):
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = True,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 30.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        thresh: Union[float, torch.Tensor] = 0.4,
        rest: Union[float, torch.Tensor] = 0,
        reset: Union[float, torch.Tensor] = 0,
        refrac: Union[int, torch.Tensor] = 1,
        tc_decay: Union[float, torch.Tensor] = 20.0,
        lbound: float = 0.0,
        tc_j: Union[float, torch.Tensor] = 10,
        kappa_j: Union[float, torch.Tensor] = 0.25,
        gamma: Union[float, torch.Tensor] = 1.0,
        Rm: Union[float, torch.Tensor] = 1.0,
        alpha_gen: Union[float, torch.Tensor] = 1,
        alpha_disc: Union[float, torch.Tensor] = 1,
        **kwargs,
    ) -> None:
        
        """
        Instantiates a layer of LIF neurons.

        :param n: 層内のニューロン数。
        :param shape: 層の次元数。
        :param traces: スパイクトレースを記録するかどうか。
        :param traces_additive: スパイクトレースを加算的に記録するかどうか。
        :param tc_trace: スパイクトレースの減衰の時定数。
        :param trace_scale: スパイクトレースのスケーリング係数。
        :param sum_input: すべての入力を合計するかどうか。
        :param thresh: スパイク閾値電圧。
        :param rest: 静止膜電圧。
        :param reset: スパイク後リセット電圧。
        :param refrac: ニューロンの不応期（非発火期）。
        :param tc_decay: ニューロン電圧減衰の時定数。
        :param lbound: 電圧の下限。
        """
        
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=False,
            thresh=thresh,
            rest=rest,
            reset=reset,
            refrac=refrac,
            lbound=lbound,
            tc_decay=tc_decay
        )

        self.register_buffer("j", torch.FloatTensor()) # 入力電流j
        self.register_buffer("tc_j", torch.tensor(tc_j, dtype=torch.float))
        self.register_buffer("kappa_j", torch.tensor(kappa_j, dtype=torch.float)) # jのリーク係数
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float)) # 漏れ係数
        self.register_buffer("Rm", torch.tensor(Rm, dtype=torch.float)) # 膜抵抗
        self.register_buffer("alpha_gen", torch.tensor(alpha_gen, dtype=torch.float))
        self.register_buffer("alpha_disc", torch.tensor(alpha_disc, dtype=torch.float))
        self.register_buffer("e_gen", torch.FloatTensor())
        self.register_buffer("e_disc", torch.FloatTensor())

    def forward(
        self, 
        prediction_bottom_up: torch.Tensor,
        prediction_top_down: torch.Tensor,
        feedback_bottom_up: torch.Tensor,
        feedback_top_down: torch.Tensor
    ) -> None:
        
        self.e_gen = self.alpha_gen * prediction_top_down
        self.e_disc = self.alpha_gen * prediction_bottom_up

        self.j = self.j + (self.dt / self.tc_j) * (- self.kappa_j * self.j + (- self.e_gen - self.e_disc + feedback_top_down + feedback_bottom_up))

        self.v = self.v + (self.dt / self.tc_decay) * (- self.gamma * self.v + self.Rm * self.j)
        self.v.masked_fill_(self.refrac_count > 0, 0)

        self.v.clamp_(min=self.lbound, max=1.0)

        self.refrac_count -= self.dt

        self.s = self.v >= self.thresh

        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        self.x = self.x + (- self.x / self.tc_trace + self.s)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.j.zero_()
        self.e_gen.zero_()
        self.e_disc.zero_()

    def compute_decays(self, dt) -> None:
        self.dt = torch.tensor(dt)

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.j = torch.zeros(batch_size, *self.shape, device=self.j.device)
        self.e_gen = torch.zeros(batch_size, *self.shape, device=self.e_gen.device)
        self.e_disc = torch.zeros(batch_size, *self.shape, device=self.e_disc.device)