# components/connections.py

from bindsnet.network.topology import Connection
# 役割ごとに分離した学習則をインポート
from ..learning_rules import GenerativeSTLRA, DiscriminativeSTLRA

class GenerativeConnection(Connection): # トップダウン予測 (W)
    def __init__(self, source, target, nu, w, **kwargs):
        super().__init__(source=source, target=target, nu=nu, **kwargs)
        self.w = w
        self.add_learning_rule(GenerativeSTLRA(self, nu=nu), name="GenerativeSTLRA")

class DiscriminativeConnection(Connection): # ボトムアップ予測 (V)
    def __init__(self, source, target, nu, w, **kwargs):
        super().__init__(source=source, target=target, nu=nu, **kwargs)
        self.w = w # この接続では、wは重みVを指す
        self.add_learning_rule(DiscriminativeSTLRA(self, nu=nu), name="DiscriminativeSTLRA")

# ... ErrorFeedbackConnectionも同様に、E_gen用とE_disc用に分離するのが望ましい ...
class GenErrorFeedbackConnection(Connection):
    # ... e_gen を取得し、e_gen_feedback を更新するロジック ...

class DiscErrorFeedbackConnection(Connection):
    # ... e_disc を取得し、e_disc_feedback を更新するロジック ...
