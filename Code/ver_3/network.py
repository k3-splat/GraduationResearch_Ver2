from typing import Dict, Iterable, Optional, Type

import torch

from bindsnet.bindsnet.network import Network
from bindsnet.bindsnet.learning.reward import AbstractReward
from bindsnet.bindsnet.network.topology import Connection, AbstractConnection
from nodes import ValueNodes, InputOutputNodes, GenerativeErrorNodes, DiscriminativeErrorNodes

def clip_weight_norm_(weights: torch.Tensor, max_norm: float = 20.0, norm_type: float = 2):
    """
    重み行列の列ベクトルのノルムが max_norm を超えている場合のみ、
    max_norm になるようにスケーリングする（In-place操作）。

    Args:
        weights (torch.Tensor): 重み行列 (Pre_neurons, Post_neurons)
        max_norm (float): 許容する最大ノルム
        norm_type (float): ノルムの種類 (2=L2ユークリッド距離, 1=L1絶対値和)
    """
    # 1. 各列のノルムを計算 (dim=0 は列方向：各後シナプスニューロンへの入力重みの総和/距離)
    current_norms = weights.norm(p=norm_type, dim=0)

    # 2. 上限を超えているインデックス（列）を探す
    mask = current_norms > max_norm

    # 3. 超えている列だけスケーリング（In-placeで書き換え）
    if mask.any():
        # scale = target / current
        scale_factor = max_norm / (current_norms[mask] + 1e-8) # 0除算防止
        weights[:, mask] *= scale_factor


class bSpNCNNetwork(Network):
    """
    bSpNCNNodesのように複数の名前付き入力を持つノードをサポートするNetworkクラス。
    """

    def __init__(self,
        dt: float = 0.25,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None
    ) -> None:
        super().__init__(
            dt=dt,
            batch_size=batch_size,
            learning=learning,
            reward_fn=reward_fn
        )
        self.layer_depth = 0

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str, connection_name: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target, connection_name)] = connection
        self.add_module(connection_name, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_middle_bspncn_layer(
        self,
        num_value_neurons: int,
        num_error_neurons: Optional[int] = None, # 誤差ニューロンの数は価値ニューロンと同じが多い
        **kwargs
    ) -> None:
        if self.layer_depth == 0:
            print("Add the input layer first.")
            return

        """
        SpNCNネットワークに論理的な層（ValueNodesと対応するErrorNodes）を追加する。

        :param depth: この層の深度（1から始まる整数）。
        :param num_value_neurons: ValueNodesのニューロン数。
        :param num_error_neurons: ErrorNodesのニューロン数。指定なければnum_value_neuronsと同じ。
        """
        if num_error_neurons is None:
            num_error_neurons = num_value_neurons

        former_layer_depth = self.layer_depth
        self.layer_depth += 1

        # 1. ValueNodes の追加
        value_node_name = f'Value_{self.layer_depth}'
        value_node = ValueNodes(
            n=num_value_neurons, 
            dt=self.dt, # ネットワークのdtを渡す
            **kwargs.get('value_node_params', {}) # ValueNodes固有のパラメータ
        )
        super().add_layer(value_node, name=value_node_name) # 親クラスのadd_layerを呼び出す

        # 2. GenerativeErrorNodes (生成的予測誤差ニューロン) の追加
        gen_error_node_name = f'Error_gen_{self.layer_depth}'
        gen_error_node = GenerativeErrorNodes(
            n=num_error_neurons,
            **kwargs.get('gen_error_node_params', {})
        )
        super().add_layer(gen_error_node, name=gen_error_node_name)

        # 3. DiscriminativeErrorNodes (識別的予測誤差ニューロン) の追加
        disc_error_node_name = f'Error_disc_{self.layer_depth}'
        disc_error_node = DiscriminativeErrorNodes(
            n=num_error_neurons,
            **kwargs.get('disc_error_node_params', {})
        )
        super().add_layer(disc_error_node, name=disc_error_node_name)

        w_identity = torch.eye(num_value_neurons)

        gen_tr_conn=Connection(
            source=value_node,
            target=gen_error_node,
            w=w_identity,
            update_rule=None
        )
        self.add_connection(
            connection=gen_tr_conn,
            source=value_node_name,
            target=gen_error_node_name,
            connection_name="gen_tr_conn"
        )

        disc_tr_conn=Connection(
            source=value_node,
            target=disc_error_node,
            w=w_identity,
            update_rule=None
        )
        self.add_connection(
            connection=disc_tr_conn,
            source=value_node_name,
            target=disc_error_node_name,
            connection_name="disc_tr_conn"
        )

        gen_pd_err_conn=Connection(
            source=gen_error_node,
            target=value_node,
            w=w_identity,
            update_rule=None
        )
        self.add_connection(
            connection=gen_pd_err_conn,
            source=gen_error_node_name,
            target=value_node_name,
            connection_name="gen_pd_err_conn"
        )

        disc_pd_err_conn=Connection(
            source=disc_error_node,
            target=value_node,
            w=w_identity,
            update_rule=None
        )
        self.add_connection(
            connection=disc_pd_err_conn,
            source=disc_error_node_name,
            target=value_node_name,
            connection_name="disc_pd_err_conn"
        )

        former_value_node_name = f'Value_{former_layer_depth}'
        former_gen_error_node_name = f'Error_gen_{former_layer_depth}'
        former_value_node = self.layers[former_value_node_name]
        former_gen_error_node = self.layers[former_gen_error_node_name]
        former_node_num = former_value_node.n

        w = clip_weight_norm_(torch.rand(num_value_neurons, former_node_num))
        v = clip_weight_norm_(torch.rand(former_node_num, num_value_neurons))
        E_gen = clip_weight_norm_(torch.rand(former_node_num, num_value_neurons))
        E_disc = clip_weight_norm_(torch.rand(num_value_neurons, former_node_num))

        gen_pd_tr_conn=Connection(
            source=value_node,
            target=former_gen_error_node,
            update_rule=None,
            w=w
        )
        self.add_connection(
            connection=gen_pd_tr_conn,
            source=value_node_name,
            target=former_gen_error_node_name,
            connection_name="gen_pd_tr_conn"
        )

        disc_pd_tr_conn=Connection(
            source=former_value_node,
            target=disc_error_node,
            update_rule=None,
            w=v
        )
        self.add_connection(
            connection=disc_pd_tr_conn,
            source=former_value_node_name,
            target=disc_error_node_name,
            connection_name="disc_pd_tr_conn"
        )

        gen_pd_err_fd_conn=Connection(
            source=former_gen_error_node,
            target=value_node,
            update_rule=None,
            w=E_gen
        )
        self.add_connection(
            connection=gen_pd_err_fd_conn,
            source=former_gen_error_node_name,
            target=value_node_name,
            connection_name="gen_pd_err_fd_conn"
        )

        disc_pd_err_fd_conn=Connection(
            source=disc_error_node,
            target=former_value_node,
            update_rule=None,
            w=E_disc
        )
        self.add_connection(
            connection=disc_pd_err_fd_conn,
            source=disc_error_node_name,
            target=former_value_node_name,
            connection_name="disc_pd_err_fd_conn"
        )

        print(f"Added SpNCN layer at depth {self.layer_depth}: {value_node_name}, {gen_error_node_name}, {disc_error_node_name}")

    def add_input_layer(
        self,
        num_value_neurons: int,
        num_error_neurons: Optional[int] = None, # 誤差ニューロンの数は価値ニューロンと同じが多い
        **kwargs
    ) -> None:
        if self.layer_depth != 0:
            print("Some layers are already existed.")
            return
        
        if num_error_neurons is None:
            num_error_neurons = num_value_neurons

        # 1. ValueNodes の追加
        input_node_name = 'Input'
        input_node = InputOutputNodes(
            n=num_value_neurons, 
            dt=self.dt, # ネットワークのdtを渡す
            **kwargs.get('value_node_params', {}) 
        )
        super().add_layer(input_node, name=input_node_name) # 親クラスのadd_layerを呼び出す

        # 1. ValueNodes の追加
        value_node_name = f'Value_{self.layer_depth}'
        value_node = ValueNodes(
            n=num_value_neurons, 
            dt=self.dt, # ネットワークのdtを渡す
            **kwargs.get('value_node_params', {}) # ValueNodes固有のパラメータ
        )
        super().add_layer(value_node, name=value_node_name) # 親クラスのadd_layerを呼び出す

        # 2. GenerativeErrorNodes (生成的予測誤差ニューロン) の追加
        gen_error_node_name = f'Error_gen_{self.layer_depth}'
        gen_error_node = GenerativeErrorNodes(
            n=num_error_neurons,
            **kwargs.get('gen_error_node_params', {})
        )
        super().add_layer(gen_error_node, name=gen_error_node_name)

        # 3. DiscriminativeErrorNodes (識別的予測誤差ニューロン) の追加
        disc_error_node_name = f'Error_disc_{self.layer_depth}'
        disc_error_node = DiscriminativeErrorNodes(
            n=num_error_neurons,
            **kwargs.get('disc_error_node_params', {})
        )
        super().add_layer(disc_error_node, name=disc_error_node_name)

        self.layer_depth += 1

        gen_tr_conn=Connection(
            source=value_node,
            target=gen_error_node
        )
        self.add_connection(
            connection=gen_tr_conn,
            source=value_node_name,
            target=gen_error_node_name,
            connection_name="gen_tr_conn"
        )

        disc_tr_conn=Connection(
            source=value_node,
            target=disc_error_node
        )
        self.add_connection(
            connection=disc_tr_conn,
            source=value_node_name,
            target=disc_error_node_name,
            connection_name="disc_tr_conn"
        )

        gen_pd_err_conn=Connection(
            source=gen_error_node,
            target=value_node
        )
        self.add_connection(
            connection=gen_pd_err_conn,
            source=gen_error_node_name,
            target=value_node_name,
            connection_name="gen_pd_err_conn"
        )

        disc_pd_err_conn=Connection(
            source=disc_error_node,
            target=value_node
        )
        self.add_connection(
            connection=disc_pd_err_conn,
            source=disc_error_node_name,
            target=value_node_name,
            connection_name="disc_pd_err_conn"
        )

        gen_tr_conn=Connection(
            source=value_node,
            target=input_node
        )
        self.add_connection(
            connection=gen_tr_conn,
            source=value_node_name,
            target=input_node_name,
            connection_name="gen_tr_conn"
        )

        gen_pd_err_fd_conn=Connection(
            source=input_node,
            target=value_node
        )
        self.add_connection(
            connection=gen_pd_err_fd_conn,
            source=input_node_name,
            target=value_node_name,
            connection_name="gen_pd_err_fd_conn"
        )

        disc_pd_tr_conn=Connection(
            source=input_node,
            target=disc_error_node
        )
        self.add_connection(
            connection=disc_pd_tr_conn,
            source=input_node_name,
            target=disc_error_node_name,
            connection_name="disc_pd_tr_conn"
        )
    
    def add_output_layer(
        self,
        num_value_neurons: int,
        num_error_neurons: Optional[int] = None, # 誤差ニューロンの数は価値ニューロンと同じが多い
        **kwargs
    ) -> None:
        if self.layer_depth == 0:
            print("No layers are existed.")
            return
        
        if num_error_neurons is None:
            num_error_neurons = num_value_neurons

        # 1. ValueNodes の追加
        output_node_name = 'Output'
        output_node = InputOutputNodes(
            n=num_value_neurons, 
            dt=self.dt, # ネットワークのdtを渡す
            **kwargs.get('value_node_params', {}) 
        )
        super().add_layer(output_node, name=output_node_name) # 親クラスのadd_layerを呼び出す

        # 1. ValueNodes の追加
        value_node_name = f'Value_{self.layer_depth}'
        value_node = self.layers[value_node_name]

        # 2. GenerativeErrorNodes (生成的予測誤差ニューロン) の追加
        gen_error_node_name = f'Error_gen_{self.layer_depth}'
        gen_error_node = self.layers[gen_error_node_name]

        disc_pd_conn=Connection(
            source=value_node,
            target=output_node
        )
        self.add_connection(
            connection=disc_pd_conn,
            source=value_node_name,
            target=output_node_name,
            connection_name="disc_pd_conn"
        )

        disc_tr_fd_conn=Connection(
            source=output_node,
            target=value_node
        )
        self.add_connection(
            connection=disc_tr_fd_conn,
            source=output_node_name,
            target=value_node_name,
            connection_name="disc_tr_fd_conn"
        )

        gen_pd_tr_conn=Connection(
            source=output_node,
            target=gen_error_node
        )
        self.add_connection(
            connection=gen_pd_tr_conn,
            source=output_node_name,
            target=gen_error_node_name,
            connection_name="gen_pd_tr_conn"
        )

    def run(
        self, 
        inputs: Dict[str, torch.Tensor], 
        time: int, 
        outputs: Optional[Dict[str, torch.Tensor]] = None, # <--- 変更点: outputs引数を追加
        **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        :param Bool progress_bar: Show a progress bar while running the network.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # outputsがNoneの場合のデフォルト処理
        if outputs is None:
            outputs = {}

        # --- メインシミュレーションループ (アルゴリズム1) ---
        for t in range(timesteps):
            input_t = inputs.get('Input', {})[t]
            
            # <--- 変更点: 現在のタイムステップの教師データを取得 ---
            output_t = outputs.get(f'Value_{self.layer_depth}', {}) # 最上位層の名前をキーとする
            if output_t is not None and len(output_t.shape) > 0:
                output_t = output_t[t]
            
            # ==========================================================
            # 1. 予測の計算 (z_μ の計算)
            # ==========================================================
            # (... このセクションは前回と同じ ...)
            predictions_gen = {} 
            predictions_disc = {} 

            # --- トップダウン予測 (Generative Prediction) ---
            # Value_l -> Error_gen_{l-1} への接続 (gen_pd_tr_conn)
            for l in range(self.layer_depth, 0, -1):
                source_name = f'Value_{l}'
                target_name = f'Error_gen_{l-1}' if l > 1 else 'Error_gen_0' # 仮のターゲット名
                # 論文では l-1 層の状態を予測する
                conn_name = 'gen_pd_tr' # (仮のコネクション名)
                if (source_name, target_name, conn_name) in self.connections:
                    conn = self.connections[(source_name, target_name, conn_name)]
                    predictions_gen[l-1] = conn.compute(self.layers[source_name].s)

            # --- ボトムアップ予測 (Discriminative Prediction) ---
            # Value_{l-1} -> Error_disc_l への接続 (disc_pd_tr_conn)
            for l in range(1, self.layer_depth + 1):
                source_name = f'Value_{l-1}' if l > 1 else 'Input'
                target_name = f'Error_disc_{l}'
                conn_name = f'disc_pd_tr_{l-1}_{l}' # (仮のコネクション名)
                if (source_name, target_name, conn_name) in self.connections:
                    conn = self.connections[(source_name, target_name, conn_name)]
                    predictions_disc[l] = conn.compute(self.layers[source_name].s)

            # ==========================================================
            # 2. 誤差の計算 (e_gen, e_disc の計算)
            # ==========================================================
            # (... このセクションは前回と同じ ...)
            
            # --- 入力層の誤差 e_gen^0 ---
            # (... 中略 ...)
            
            # --- 中間層・最終層の誤差 ---
            for l in range(1, self.layer_depth + 1):None
                # (... 中略 ...)

            # ==========================================================
            # 3. ニューロン状態の更新 (j, v, s, x の更新)
            # ==========================================================
            
            # --- 入力層 (InputOutputNodes) の更新 ---
            # (... このセクションは前回と同じ ...)
            
            # --- 中間層 (ValueNodes) の更新 ---
            # <--- 変更点: ループの範囲を最上位層の一つ手前までにする ---
            for l in range(1, self.layer_depth): 
                value_layer = self.layers[f'Value_{l}']
                # (... 計算ロジックは前回と同じ ...)
                total_error_input = (...)
                value_layer.forward(total_error_input)

            # --- 最上位層 (ValueNodes) の特別処理 (教師あり) ---
            top_layer_name = f'Value_{self.layer_depth}'
            top_layer = self.layers[top_layer_name]
            
            if output_t is not None and len(output_t.shape) > 0:
                # --- 教師ありモード: 状態を教師データでクランプ ---
                # one-hotベクトルのような教師データをスパイクとトレースに直接設定
                top_layer.s = output_t.byte() # スパイクを教師データに固定
                top_layer.x = output_t.float() # トレースも教師データに固定
                # 電圧や電流は更新しない、あるいはリセットする（設計による）
                top_layer.v.fill_(top_layer.reset) 
                top_layer.j.zero_()
            else:
                # --- 教師なし/推論モード: 通常通り更新 ---
                l = self.layer_depth
                local_gen_err_term = -self.layers[f'Error_gen_{l}'].s
                local_disc_err_term = -self.layers[f'Error_disc_{l}'].s
                source_err_gen = self.layers[f'Error_gen_{l-1}'].s
                conn_gen_fb = self.connections[(f'Error_gen_{l-1}', f'Value_{l}', 'gen_pd_err_fd_conn')]
                feedback_gen_err_term = conn_gen_fb.compute(source_err_gen)
                
                # 最上位層なので、上位からの識別誤差フィードバックはない
                feedback_disc_err_term = torch.zeros_like(top_layer.j)
                
                total_error_input = (local_gen_err_term + local_disc_err_term + 
                                    feedback_gen_err_term + feedback_disc_err_term)
                
                top_layer.forward(total_error_input)

            # ==========================================================
            # 4. シナプス重みの更新 (学習)
            # ==========================================================
            if self.learning:
                # (... このセクションのロジックは前回と同じ ...)
                # 最上位層の状態が教師データで固定されたことにより、
                # そこから計算されるトップダウンの予測 (z_gen^{L-1}) が教師信号となり、
                # 誤差 e_gen^{L-1} を通じて下位層の重みが正しく更新される。
                for (source, target, name), conn in self.connections.items():None
                    # (... 中略 ...)

            # モニターの記録
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections
        for c in self.connections:
            self.connections[c].normalize()