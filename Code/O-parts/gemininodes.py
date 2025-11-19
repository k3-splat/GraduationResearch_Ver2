# spncn_nodes.py

import torch
from bindsnet.network.nodes import Nodes
from typing import Optional, Iterable, Union

class SpNCNLIFNodes(Nodes):
    """
    SpNCNで用いられるLeaky Integrate-and-Fire (LIF) ニューロンモデル。
    膜電位の更新(式10)とスパイク・トレースの更新(式1)は論文の定義に従います。
    """
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        dt: float = 1.0,
        v_thresh: float = 0.4,
        refractory_period: int = 1,
        tau_m: float = 20.0,
        gamma_m: float = 1.0,
        R_m: float = 1.0,
        tc_trace: float = 30.0,
        **kwargs,
    ):
        """
        :param n: レイヤーのニューロン数
        :param shape: レイヤーの次元
        :param dt: シミュレーションのタイムステップ
        :param v_thresh: スパイクを発生させる膜電位の閾値
        :param refractory_period: 不応期 (ms)
        :param tau_m: 膜時定数 (τm)
        :param gamma_m: 膜電位の漏れ係数 (γm)
        :param R_m: 膜抵抗 (Rm)
        :param tc_trace: スパイク・トレースの時定数 (τ_tr)
        """
        super().__init__(n=n, shape=shape, traces=True, **kwargs)

        self.dt = dt
        self.register_buffer("v_thresh", torch.tensor(v_thresh, dtype=torch.float))
        self.register_buffer("refractory_period", torch.tensor(refractory_period, dtype=torch.float))
        self.register_buffer("tau_m", torch.tensor(tau_m, dtype=torch.float))
        self.register_buffer("gamma_m", torch.tensor(gamma_m, dtype=torch.float))
        self.register_buffer("R_m", torch.tensor(R_m, dtype=torch.float))
        self.register_buffer("tc_trace", torch.tensor(tc_trace, dtype=torch.float))

        self.register_buffer("v", torch.zeros(self.shape)) # 膜電位
        self.register_buffer("refrac_count", torch.zeros(self.shape)) # 不応期カウンター

    def forward(self, x: torch.Tensor) -> None:
        """
        1タイムステップのシミュレーションを実行します。
        :param x: 入力電流 (j)
        """
        # 不応期にあるニューロンへの入力をマスク
        active_neurons = (self.refrac_count <= 0)

        # 論文 式(10) に基づく膜電位 v の更新 (オイラー法)
        # v(t + Δt) = v(t) + (Δt / τm) * (-γm * v(t) + Rm * j(t))
        update = (self.dt / self.tau_m) * (-self.gamma_m * self.v + self.R_m * x)
        self.v += active_neurons.float() * update

        # 不応期カウンターをデクリメント
        self.refrac_count -= self.dt
        self.refrac_count.clamp_(min=0)

        # スパイク生成
        spikes = self.v >= self.v_thresh
        self.s = spikes.byte()

        # スパイク後のリセットと不応期設定
        self.v.masked_fill_(spikes, 0.0) # 論文に基づき静止電位(0)にリセット
        self.refrac_count.masked_fill_(spikes, self.refractory_period)
        
        # 論文 式(1) に基づくトレース z (self.x) の更新 (オイラー法)
        # dz/dt = -z/τ_tr + s(t)
        self.x += ((-self.x / self.tc_trace) + self.s.float())

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.v.zero_()
        self.refrac_count.zero_()

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros(batch_size, *self.shape, device=self.refrac_count.device)