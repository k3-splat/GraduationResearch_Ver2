import torch
from typing import Dict, Iterable, Optional, Tuple

# ユーザー定義のカスタムノードをインポート
from nodes import bSpNCNNodes

# Bindsnetのコアコンポーネントをインポート
from bindsnet.network import Network
from bindsnet.network.topology import AbstractConnection

class bSpNCNNetwork(Network):
    """
    bSpNCNNodesのように複数の名前付き入力を持つノードをサポートし、
    かつ同じ層間に複数の名前付き接続を許容するように拡張されたNetworkクラス。
    """

    def __init__(self, dt: float = 1.0, **kwargs) -> None:
        """
        ネットワークを初期化します。
        接続はユニークな名前によって管理されます。
        """
        # 親クラスの初期化を呼び出すが、一部の属性は独自に管理する
        super().__init__(dt=dt, **kwargs)
        
        # 接続を管理する辞書を上書き
        # キー: 接続のユニーク名 (str)
        # 値: (Connectionオブジェクト, source名, target名) のタプル
        self.connections: Dict[str, Tuple[AbstractConnection, str, str]] = {}
        
        # 接続ルールを管理する辞書も、ユニーク名をキーとする
        self.connection_types: Dict[str, str] = {}
        self.connection_info_types: Dict[str, str] = {}

    def add_connection(
        self,
        connection: AbstractConnection,
        source: str,
        target: str,
        name: str,  # 接続を一位に特定するための名前
        input_type: Optional[str] = None,
        info_type: str = 'spike',
    ) -> None:
        """
        ネットワークに名前付きの接続を追加します。

        :param connection: `Connection` クラスのインスタンス。
        :param source: 接続元のレイヤー名。
        :param target: 接続先のレイヤー名。
        :param name: この接続を一位に特定するためのユニークな名前。
        :param input_type: `bSpNCNNodes` の入力名（例: 'prediction_bottom_up'）。
        :param info_type: 接続が伝える情報の種類（例: 'spike', 'feedback_top_down'）。
        """
        # PyTorchモジュールとして登録し、後で保存などができるようにする
        self.add_module(name, connection)
        
        # 独自の方法で接続を管理
        self.connections[name] = (connection, source, target)
        
        # 接続ルールを名前付きで保存
        self.connection_info_types[name] = info_type

        if input_type is not None:
            if target in self.layers and isinstance(self.layers[target], bSpNCNNodes):
                self.connection_types[name] = input_type
            else:
                print(f"Warning: 'input_type' ignored for non-bSpNCNNodes target '{target}'.")

    def _get_inputs(self, layers: Optional[Iterable[str]] = None) -> Dict:
        """
        ネットワーク内の接続から各層への入力を計算し、仕分けします。
        """
        inputs = {}
        layers_to_process = self.layers if layers is None else {k for k in layers if k in self.layers}

        for conn_name, (connection, source_name, target_name) in self.connections.items():
            if target_name in layers_to_process:
                source_layer = self.layers[source_name]
                target_layer = self.layers[target_name]
                
                info_type = self.connection_info_types.get(conn_name, 'spike')
                
                if info_type == 'spike':
                    data_to_transmit = source_layer.s
                elif info_type == 'feedback_top_down':
                    data_to_transmit = source_layer.e_gen
                elif info_type == 'feedback_bottom_up':
                    data_to_transmit = source_layer.e_disc
                else:
                    data_to_transmit = source_layer.s

                computed_input = connection.compute(data_to_transmit)

                if isinstance(target_layer, bSpNCNNodes):
                    input_type = self.connection_types.get(conn_name)
                    if input_type is None:
                        raise ValueError(
                            f"Connection '{conn_name}' from '{source_name}' to bSpNCNNodes layer "
                            f"'{target_name}' must have an 'input_type' specified."
                        )
                    
                    if target_name not in inputs:
                        inputs[target_name] = {}
                    
                    if input_type not in inputs[target_name]:
                        inputs[target_name][input_type] = torch.zeros_like(target_layer.s)
                    
                    inputs[target_name][input_type] += computed_input
                
                else:
                    if target_name not in inputs:
                        inputs[target_name] = torch.zeros_like(target_layer.s)
                    
                    inputs[target_name] += computed_input
        
        return inputs

    def run(self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs) -> None:
        """
        与えられた入力と時間でネットワークをシミュレートします。
        オリジナルのBindsnetの機能を全て含んでいます。
        """
        # 入力形式のチェック
        assert type(inputs) == dict, "Inputs must be a dictionary."

        # キーワード引数の解析
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # 報酬の計算
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # バッチサイズの動的設定
        if inputs:
            # 入力テンソルの次元を [time, batch, ...] に整形
            for key in inputs:
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif inputs[key].dim() == 2:
                    inputs[key] = inputs[key].unsqueeze(1)
            
            # 最初の入力からバッチサイズを取得し、ネットワーク全体に適用
            # Get first input key
            first_key = next(iter(inputs))
            if inputs[first_key].size(1) != self.batch_size:
                self.batch_size = inputs[first_key].size(1)
                for l in self.layers:
                    self.layers[l].set_batch_size(self.batch_size)
                for m in self.monitors:
                    self.monitors[m].reset_state_variables()

        timesteps = int(time / self.dt)

        # シミュレーションのメインループ
        for t in range(timesteps):
            # 内部接続からの入力を取得
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            # 各レイヤーの更新
            for l_name, l_layer in self.layers.items():
                # 外部からの入力を追加
                if l_name in inputs:
                    if l_name in current_inputs:
                        if not isinstance(l_layer, bSpNCNNodes):
                             current_inputs[l_name] += inputs[l_name][t]
                    else:
                        current_inputs[l_name] = inputs[l_name][t]
                
                if one_step:
                    current_inputs.update(self._get_inputs(layers=[l_name]))
                
                # 電圧の注入
                inject_v = injects_v.get(l_name, None)
                if inject_v is not None:
                    if inject_v.dim() == 1:
                        l_layer.v += inject_v
                    else:
                        l_layer.v += inject_v[t]
                
                # forwardメソッドの呼び出し
                if l_name in current_inputs:
                    if isinstance(l_layer, bSpNCNNodes):
                        l_layer.forward(**current_inputs[l_name])
                    else:
                        l_layer.forward(x=current_inputs[l_name])
                else:
                    if isinstance(l_layer, bSpNCNNodes):
                        shape = (self.batch_size, *l_layer.shape)
                        device = l_layer.s.device
                        l_layer.forward(
                            prediction_bottom_up=torch.zeros(*shape, device=device),
                            prediction_top_down=torch.zeros(*shape, device=device),
                            feedback_bottom_up=torch.zeros(*shape, device=device),
                            feedback_top_down=torch.zeros(*shape, device=device),
                        )
                    else:
                        l_layer.forward(x=torch.zeros_like(l_layer.s))

                # クランプ処理
                clamp = clamps.get(l_name, None)
                if clamp is not None:
                    if clamp.dim() == 1:
                        l_layer.s[:, clamp] = 1
                    else:
                        l_layer.s[:, clamp[t]] = 1
                
                unclamp = unclamps.get(l_name, None)
                if unclamp is not None:
                    if unclamp.dim() == 1:
                        l_layer.s[:, unclamp] = 0
                    else:
                        l_layer.s[:, unclamp[t]] = 0

            # 各接続の学習則を更新
            for conn_name, (connection, _, _) in self.connections.items():
                mask = masks.get(conn_name, None)
                connection.update(mask=mask, learning=self.learning, **kwargs)

            # モニターの記録
            for m in self.monitors:
                self.monitors[m].record()

        # シミュレーション後の正規化
        for conn_name, (connection, _, _) in self.connections.items():
            connection.normalize()