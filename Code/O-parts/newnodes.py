from typing import Iterable, Optional, Union
from bindsnet.network.nodes import LIFNodes, Nodes

import torch

class bSpNCN_LIFNodes(LIFNodes):
    """
    指定された更新則に従い、電流(j)、膜電位(v)、スパイク(s)、トレース(x)を計算するニューロン層。
    """
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
        **kwargs,
    ) -> None:
        super().__init__(
            n=n, shape=shape, traces=traces, traces_additive=traces_additive,
            tc_trace=tc_trace, trace_scale=trace_scale, sum_input=False,
            thresh=thresh, rest=rest, reset=reset, refrac=refrac,
            lbound=lbound, tc_decay=tc_decay
        )
        self.register_buffer("j", torch.FloatTensor())
        self.register_buffer("tc_j", torch.tensor(tc_j, dtype=torch.float))
        self.register_buffer("kappa_j", torch.tensor(kappa_j, dtype=torch.float))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float))
        self.register_buffer("Rm", torch.tensor(Rm, dtype=torch.float))

    def forward(
        self,
        gen_error_node: "GenerativeErrorNodes",
        disc_error_node: "DiscriminativeErrorNodes",
        feedback_bottom_up: torch.Tensor,
        feedback_top_down: torch.Tensor
    ) -> None:
        # 1. 電流 j の更新 (ご指定の数式)
        self.j = self.j + (self.dt / self.tc_j) * (
            -self.kappa_j * self.j +
            (-gen_error_node.e_gen - disc_error_node.e_disc + feedback_top_down + feedback_bottom_up)
        )

        # 2. 電圧 v の更新 (ご指定の数式)
        self.v = self.v + (self.dt / self.tc_decay) * (-self.gamma * self.v + self.Rm * self.j)
        
        # 3. 不応期にあるニューロンの電圧をリセット (ご指定のロジック)
        # 注意: masked_fill_の第一引数はBoolean Tensorである必要があります
        self.v.masked_fill_(self.refrac_count > 0, self.reset)
        
        # 4. 電圧のクリッピング (ご指定のロジック)
        if self.lbound is not None:
             self.v.clamp_(min=self.lbound, max=1.0)

        # 5. 不応期カウンターのデクリメント (ご指定のロジック)
        self.refrac_count -= self.dt
        self.refrac_count.clamp_(min=0)

        # 6. スパイク s の生成 (ご指定のロジック)
        self.s = (self.v >= self.thresh).byte()

        # 7. スパイクしたニューロンの不応期と電圧のリセット (ご指定のロジック)
        # 注意: masked_fill_の第一引数はBoolean Tensorである必要があります
        self.refrac_count.masked_fill_(self.s.bool(), self.refrac)
        self.v.masked_fill_(self.s.bool(), self.reset)
        
        # 8. トレース x の更新 (ご指定の数式)
        # dtで割るのではなく、dtを掛けるのが一般的ですが、ご指定の式をそのまま実装します。
        # もし `dx/dt = -x/tau + s` を意図している場合は、dtを掛ける必要があります。
        # self.x = self.x + self.dt * (-self.x / self.tc_trace + self.s.float())
        if self.traces:
            self.x = self.x + (-self.x / self.tc_trace + self.s.float())


    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.j.zero_()

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.j = torch.zeros(batch_size, *self.shape, device=self.j.device)


class GenerativeErrorNodes(Nodes):
    """
    トップダウン予測から生成誤差 e_gen を計算するニューロン層。
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        alpha_gen: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(n=n, shape=shape, traces=False, **kwargs)
        self.register_buffer("alpha_gen", torch.tensor(alpha_gen, dtype=torch.float))
        self.register_buffer("e_gen", torch.FloatTensor())

    def forward(self, prediction_top_down: torch.Tensor) -> None:
        self.e_gen = self.alpha_gen * prediction_top_down
        self.s.zero_()

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.e_gen.zero_()

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.e_gen = torch.zeros(batch_size, *self.shape, device=self.e_gen.device)


class DiscriminativeErrorNodes(Nodes):
    """
    ボトムアップ予測から識別誤差 e_disc を計算するニューロン層。
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        alpha_disc: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(n=n, shape=shape, traces=False, **kwargs)
        self.register_buffer("alpha_disc", torch.tensor(alpha_disc, dtype=torch.float))
        self.register_buffer("e_disc", torch.FloatTensor())

    def forward(self, prediction_bottom_up: torch.Tensor) -> None:
        self.e_disc = self.alpha_disc * prediction_bottom_up
        self.s.zero_()

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.e_disc.zero_()

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.e_disc = torch.zeros(batch_size, *self.shape, device=self.e_disc.device)

