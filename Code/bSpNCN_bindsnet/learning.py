from typing import Optional, Sequence, Union

import torch
import network
from bindsnet.bindsnet.network.topology import AbstractConnection
from bindsnet.bindsnet.learning import LearningRule

class update_WorV(LearningRule):
    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super.__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self):
        delta_w = self.connection.target.e @ self.connection.source.s.T
        self.connection.w += self.nu * delta_w

        network.clip_weight_norm_(self.connection.w)


class update_Err_Fd(LearningRule):
    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super.__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self):
        delta_w = self.connection.target.s @ self.connection.source.e.T
        self.connection.w += self.nu * delta_w

        network.clip_weight_norm_(self.connection.w)