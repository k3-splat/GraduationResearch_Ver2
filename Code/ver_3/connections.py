# custom_connections.py

import torch
from bindsnet.bindsnet.network.topology import Connection
from nodes import ValueNodes, GenerativeErrorNodes, DiscriminativeErrorNodes
from typing import Optional, Union, Sequence

class NowPredictTraceConnection(Connection):
    """
    ソース層のスパイク(s)の代わりに、トレース(x)を伝播させるためのConnectionクラス。
    
    `run`関数内の `connection.compute(source.s)` という呼び出しに対応するため、
    `compute`メソッドは引数 `s` を受け取りますが、それを無視して
    内部で `self.source.x` を直接参照します。
    """
    def __init__(
        self,
        source: ValueNodes,
        target: Union[GenerativeErrorNodes, DiscriminativeErrorNodes],
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(source, target, nu, **kwargs)

    def compute(self) -> torch.Tensor:
        """
        ソース層のトレース `x` を使って、ターゲット層への入力を計算します。
        
        :param s: Bindsnetのネットワークループから渡されるソース層のスパイク。この引数は無視されます。
        :return: ソース層のトレースとシナプス重みを乗算した結果。
        """
        # この接続の本当の情報源であるソース層のトレースを取得
        trace = self.source.x
        
        # 親クラス `Connection` の compute メソッドのロジックを、
        # `s` の代わりに `trace` を使って再現
        if self.b is None:
            post = trace.view(trace.size(0), -1).float() @ self.w
        else:
            post = trace.view(trace.size(0), -1).float() @ self.w + self.b
            
        return post.view(trace.size(0), *self.target.shape)