# 遺物。おそらく使わない。ただパーツや参考としておいておく
import torch
from bindsnet.network.nodes import LIFNodes
from typing import Optional, Iterable, Union

class bPC_LIFNodes(LIFNodes):
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0, 
        rest: Union[float, torch.Tensor] = 0.0, 
        reset: Union[float, torch.Tensor] = 0.0, 
        refrac: Union[int, torch.Tensor] = 1,
        tc_decay: Union[float, torch.Tensor] = 20.0, # 
        lbound: float = None,
        tc_j_decay: Union[float, torch.Tensor] = 10.0, # 膜電位の時定数
        kappa_j: Union[float, torch.Tensor] = 0.25, # 論文のκ_jに相当
        alpha_gen: Union[float, torch.Tensor] = 1.0, # 論文のα_genに相当
        alpha_disc: Union[float, torch.Tensor] = 1.0, # 論文のα_discに相当
        gamma: Union[float, torch.Tensor] = 1.0,
        Rm: Union[float, torch.Tensor] = 1.0,
        dt: float = 0.25, # 論文のΔtに相当, シミュレーション時間ステップ
        **kwargs,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input, # sum_inputをFalseにするか、無視する設計
            thresh=thresh,
            rest=rest,
            reset=reset,
            refrac=refrac,
            tc_decay=tc_decay, # ここは論文のτ_m (membrane time constant) に対応
            lbound=lbound,
            **kwargs,
        )

        # 私たちのモデル固有の状態変数
        self.register_buffer("j", torch.zeros(*self.shape)) # ニューロンへの入力電流j
        
        # 誤差フィードバックを受け取るための変数 (接続から受け取る)
        # これらはforwardメソッドのinputs辞書から取得するようにする
        # self.e_gen_feedback = None 
        # self.e_disc_feedback = None

        # 私たちのモデル固有のパラメータを登録
        self.register_buffer("tc_j_decay", torch.tensor(tc_j_decay, dtype=torch.float)) # jの減衰時定数
        self.register_buffer("kappa_j", torch.tensor(kappa_j, dtype=torch.float)) # jのリーク係数
        self.register_buffer("alpha_gen", torch.tensor(alpha_gen, dtype=torch.float)) # 生成的誤差の重み
        self.register_buffer("alpha_disc", torch.tensor(alpha_disc, dtype=torch.float)) # 識別的誤差の重み
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float)) # 漏れ係数
        self.register_buffer("R", torch.tensor(Rm, dtype=torch.float)) # 膜抵抗
        self.dt = dt # BindsNETのdtが使われることを確認


    def forward(self, inputs: dict) -> None: # x: torch.Tensor ではなく inputs: dict に変更
        # inputs辞書から、接続が計算して送ってきた誤差フィードバック信号を取得
        # Connectionクラスのforwardでtarget.e_gen_feedback = ... のように設定する
        e_gen_feedback = inputs.get('e_gen_feedback', torch.zeros_like(self.j))
        e_disc_feedback = inputs.get('e_disc_feedback', torch.zeros_like(self.j))

        # --- 私たちのモデルの電流jの更新式 ---
        # 誤差項の合計 (phi_eは恒等関数と仮定)
        error_sum = (
            -self.alpha_gen * self.e_gen
            -self.alpha_disc * self.e_disc
            + e_gen_feedback
            + e_disc_feedback
        )
        # 式(5) or (6)
        self.j += (-self.kappa_j * self.j + error_sum) * self.dt / self.tc_j_decay # オイラー法による更新

        # --- ここから、BindsNETのLIFNodesの通常のforwardロジックを模倣または呼び出す ---

        # 電圧vの減衰 (論文のEquation 10の左側に対応)
        # 論文のτ_m (tc_decay) を使う
        self.v = self.decay * (self.v - self.rest) + self.rest 
        
        # 論文のEquation 10の右側: Rm * j^l(t) 項を電圧に加算
        # BindsNETのLIFNodesのx入力がこのRm*jの役割を果たすようにする
        # ここでRmは通常1と仮定するか、パラメータとして追加
        current_input_to_v = self.Rm * self.j * self.dt / self.tc_decay # dt/tc_decay を掛けることで、Rm*jのスケールを合わせる (論文Eq10)

        # 屈折期間中の入力と電圧をリセット
        # これらはBindsNETのLIFNodesのx入力に直接加算される前に処理される
        current_input_to_v.masked_fill_(self.refrac_count > 0, 0.0) # 屈折期間中は電流0

        self.refrac_count -= self.dt # 屈折カウンタを減らす

        self.v += current_input_to_v # 電圧に電流を加算

        # 発火チェック
        self.s = self.v >= self.thresh

        # 屈折期間と電圧リセット
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # 電圧のクリッピング
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # トレース z の更新 (super().forward(x) の代わりにここで明示的に処理)
        # super().forward(x) は、内部で self.s を更新し、self.z を計算する
        # bPC_LIFNodesのinitでtraces=Trueに設定されているはずなので、super().forward()を呼び出すのではなく、
        # ここで自前でzを更新するか、sを更新した後にsuper()を呼び出すか要検討
        # 論文のz更新式: z(t) = z(t) + (-z(t)/τ_tr + s(t))
        if self.traces:
            self.z = self.z - self.z / self.tc_j_decay + self.s


        # BindsNETのNodesクラスのforward (空実装) を呼び出す必要は基本ない
        # super().forward(x) # これはNodesクラスのforwardを呼び出すが、特に何も行わないことが多い

    # set_batch_size, reset_state_variables, compute_decays は親クラスと同じでOK
    # ただし、jとe_gen/e_discも初期化されるように reset_state_variables をオーバーライドすべき
    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        # j, e_gen, e_disc もバッチサイズに合わせて初期化
        self.j = torch.zeros(batch_size, *self.shape, device=self.j.device)
        self.e_gen = torch.zeros(batch_size, *self.shape, device=self.e_gen.device)
        self.e_disc = torch.zeros(batch_size, *self.shape, device=self.e_disc.device)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.j.zero_()
        self.e_gen.zero_()
        self.e_disc.zero_()
        # 必要に応じて、e_gen_feedback, e_disc_feedback もリセット