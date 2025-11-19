# components/learning_rules.py

import torch
from bindsnet.network.topology import AbstractConnection
from bindsnet.learning import LearningRule

def normalize_columns(weights: torch.Tensor, max_norm: float = 20.0) -> torch.Tensor:
    """
    重み行列の各列ベクトルのL2ノルムがmax_norm以下になるように正規化します。

    :param weights: 正規化対象の重み行列 (torch.Tensor)。
    :param max_norm: 許容されるL2ノルムの最大値。
    :return: 正規化された重み行列 (torch.Tensor)。
    """
    # 1. 各列ベクトルのL2ノルムを計算
    #    dim=0 は列ごとの計算を指定
    #    keepdim=True はブロードキャストのために次元を維持 (例: (4,) -> (1, 4))
    norms = torch.linalg.norm(weights, ord=2, dim=0, keepdim=True)

    # 2. 各列に掛けるべきスケール係数を計算
    #    ノルムがmax_normより大きい列だけ縮小させたい
    #    - ノルム > max_norm の場合: scale_factor < 1.0 となり、列を縮小
    #    - ノルム <= max_norm の場合: scale_factor >= 1.0 となるが、torch.minimumで1.0に制限され、列は変更されない
    clipped_scale_factor = torch.minimum(max_norm / norms, torch.tensor(1.0))

    # 3. 元の重みにスケール係数を掛けて正規化
    #    ブロードキャスト機能により、各列ベクトル全体に正しい係数が適用される
    return weights * clipped_scale_factor

class TopDownPredictionRule(LearningRule):
    """
    bSpNCNモデルのためのカスタム学習則。
    共通の学習率を使用し、予測誤差に基づいて重みを更新する。
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: float = 1e-3,  # 単一の学習率を受け取る
        **kwargs,
    ):
        # 親クラスの初期化
        super().__init__(connection=connection, nu=nu, **kwargs)

    def update(self, **kwargs) -> None:
        # 1. 共通の学習率を取得
        #    - network.run()で 'nu' が指定されればそれを使う
        #    - 指定がなければ、__init__で設定されたデフォルト値を使う
        #    - self.nu[0] を使うのは、親クラスがnuをペアで保持するため
        learning_rate = kwargs.get('nu', self.nu[0])

        # 2. 必要な情報を取得
        source_s = self.source.s.float().view(-1)
        target_e_gen = self.target.e_gen.float().view(-1)
        
        # 3. 重みの変化量を計算
        dw_gen = torch.ger(target_e_gen, source_s)
        
        # 4. 接続の重みを更新
        self.connection.w += learning_rate * dw_gen
        
        # 5. 親クラスの共通処理 (重み減衰など) を呼び出す
        #    ※ ノルム正規化を使う場合は、このsuper().update()は呼び出さない
        # super().update()

        # ※ ノルム正規化を使う場合は、代わりに以下を有効化
        if self.weight_decay < 1.0:
            self.connection.w *= self.weight_decay

        self.connection.w.data = normalize_columns(self.connection.w.data, max_norm=20.0)

class BottomUpPredictionRule(LearningRule):
    """
    bSpNCNモデルのためのカスタム学習則。
    共通の学習率を使用し、予測誤差に基づいて重みを更新する。
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: float = 1e-3,  # 単一の学習率を受け取る
        **kwargs,
    ):
        # 親クラスの初期化
        super().__init__(connection=connection, nu=nu, **kwargs)

    def update(self, **kwargs) -> None:
        # 1. 共通の学習率を取得
        #    - network.run()で 'nu' が指定されればそれを使う
        #    - 指定がなければ、__init__で設定されたデフォルト値を使う
        #    - self.nu[0] を使うのは、親クラスがnuをペアで保持するため
        learning_rate = kwargs.get('nu', self.nu[0])

        # 2. 必要な情報を取得
        source_s = self.source.s.float().view(-1)
        target_e_disc = self.target.e_disc.float().view(-1)
        
        # 3. 重みの変化量を計算
        dw_disc = torch.ger(target_e_disc, source_s)
        
        # 4. 接続の重みを更新
        self.connection.w += learning_rate * dw_disc
        
        # 5. 親クラスの共通処理 (重み減衰など) を呼び出す
        #    ※ ノルム正規化を使う場合は、このsuper().update()は呼び出さない
        # super().update()

        # ※ ノルム正規化を使う場合は、代わりに以下を有効化
        if self.weight_decay < 1.0:
            self.connection.w *= self.weight_decay

        self.connection.w.data = normalize_columns(self.connection.w.data, max_norm=20.0)

class TopDownFeedbackRule(LearningRule):
    """
    bSpNCNモデルのためのカスタム学習則。
    共通の学習率を使用し、予測誤差に基づいて重みを更新する。
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: float = 1e-3,  # 単一の学習率を受け取る
        **kwargs,
    ):
        # 親クラスの初期化
        super().__init__(connection=connection, nu=nu, **kwargs)

    def update(self, **kwargs) -> None:
        # 1. 共通の学習率を取得
        #    - network.run()で 'nu' が指定されればそれを使う
        #    - 指定がなければ、__init__で設定されたデフォルト値を使う
        #    - self.nu[0] を使うのは、親クラスがnuをペアで保持するため
        learning_rate = kwargs.get('nu', self.nu[0])

        # 2. 必要な情報を取得
        source_e_gen = self.source.e_gen.float().view(-1)
        target_s = self.target.s.float().view(-1)
        
        # 3. 重みの変化量を計算
        dE_gen = torch.ger(target_s, source_e_gen)
        
        # 4. 接続の重みを更新
        self.connection.w += learning_rate * dE_gen
        
        # 5. 親クラスの共通処理 (重み減衰など) を呼び出す
        #    ※ ノルム正規化を使う場合は、このsuper().update()は呼び出さない
        # super().update()

        # ※ ノルム正規化を使う場合は、代わりに以下を有効化
        if self.weight_decay < 1.0:
            self.connection.w *= self.weight_decay

        self.connection.w.data = normalize_columns(self.connection.w.data, max_norm=20.0)

class BottomUpFeedbackRule(LearningRule):
    """
    bSpNCNモデルのためのカスタム学習則。
    共通の学習率を使用し、予測誤差に基づいて重みを更新する。
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: float = 1e-3,  # 単一の学習率を受け取る
        **kwargs,
    ):
        # 親クラスの初期化
        super().__init__(connection=connection, nu=nu, **kwargs)

    def update(self, **kwargs) -> None:
        # 1. 共通の学習率を取得
        #    - network.run()で 'nu' が指定されればそれを使う
        #    - 指定がなければ、__init__で設定されたデフォルト値を使う
        #    - self.nu[0] を使うのは、親クラスがnuをペアで保持するため
        learning_rate = kwargs.get('nu', self.nu[0])

        # 2. 必要な情報を取得
        source_e_disc = self.source.e_disc.float().view(-1)
        target_s = self.target.s.float().view(-1)
        
        # 3. 重みの変化量を計算
        dE_disc = torch.ger(target_s, source_e_disc)
        
        # 4. 接続の重みを更新
        self.connection.w += learning_rate * dE_disc
        
        # 5. 親クラスの共通処理 (重み減衰など) を呼び出す
        #    ※ ノルム正規化を使う場合は、このsuper().update()は呼び出さない
        # super().update()

        # ※ ノルム正規化を使う場合は、代わりに以下を有効化
        if self.weight_decay < 1.0:
            self.connection.w *= self.weight_decay

        self.connection.w.data = normalize_columns(self.connection.w.data, max_norm=20.0)
