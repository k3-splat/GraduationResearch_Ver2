import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from bindsnet.bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
# DatasetをCIFAR10に変更
from bindsnet.bindsnet.datasets import CIFAR10 
from bindsnet.bindsnet.encoding import PoissonEncoder
from bindsnet.bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.bindsnet.models import DiehlAndCook2015
from bindsnet.bindsnet.network.monitors import Monitor
from bindsnet.bindsnet.utils import get_square_assignments, get_square_weights

# --- 引数設定 ---
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training") # バッチサイズ引数追加
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=32)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)

# ベンチマーク用拡張引数
parser.add_argument("--target_accuracy", type=float, default=0.85, help="Target accuracy to stop training (0.0-1.0)")
parser.add_argument("--validation_interval", type=int, default=1000, help="Interval to check validation accuracy")
parser.add_argument("--n_val_samples", type=int, default=1000, help="Number of samples to use for fast validation")
parser.add_argument("--max_samples", type=int, default=300000, help="Maximum number of training samples")

parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size # バッチサイズ取得
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
sim_time = args.time # 変数名変更 (time -> sim_time)
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
device_id = args.device_id

target_accuracy = args.target_accuracy
validation_interval = args.validation_interval
n_val_samples = args.n_val_samples
max_samples = args.max_samples

# --- GPU設定 ---
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{device_id}")
else:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

n_classes = 10
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
per_class = int(n_neurons / n_classes)

# --- ネットワーク構築 (Diehl & Cook 2015) ---
# CIFAR-10用にパラメータを変更 (32x32x3 = 3072)
n_inpt = 3072 
inpt_shape = (3, 32, 32)

network = DiehlAndCook2015(
    n_inpt=n_inpt,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=inpt_shape,
    device=device # 【重要】GPUエラー対策: ここでdeviceを渡す
)

# 念のため to(device) も呼ぶ（device引数で解決済みだが安全策）
network.to(device)

# モニター設定
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=sim_time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=sim_time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=sim_time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# --- データセットの準備 (CIFAR-10) ---
# 保存先パスの設定
data_path = os.path.join("..", "..", "data", "CIFAR10")

full_train_dataset = CIFAR10(
    PoissonEncoder(time=sim_time, dt=dt),
    None,
    root=data_path,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

train_size = 45000 
val_size = 5000
if len(full_train_dataset) != 50000:
    val_size = int(len(full_train_dataset) * 0.1)
    train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset = CIFAR10(
    PoissonEncoder(time=sim_time, dt=dt),
    None,
    root=data_path,
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# 【重要】バッチサイズを指定してDataLoaderを作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=gpu)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=gpu)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=gpu)

# --- 変数初期化 ---
# spike_record はラベル割り当て(assign_labels)のために「直近のデータ」の発火を記録する
# バッチ処理に合わせて、割り当て計算時は集計用のバッファを使う形に変更も考えられるが、
# ここでは簡易的に「update_interval」分のデータを保持する形を維持する。
# ただし、バッチサイズが大きいと update_interval を超える可能性があるため、記録ロジックに注意が必要。
spike_record = torch.zeros(update_interval, sim_time, n_neurons, device=device)
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
accuracy = {"all": [], "proportion": []}
labels = torch.empty(update_interval, device=device)

# コスト計測用変数
total_spikes = 0 

# プロット用
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

# --- ヘルパー関数 ---
def run_evaluation(network, loader, n_samples_to_test, description="Evaluation"):
    network.train(mode=False)
    total_correct = 0
    total_tested = 0
    pbar_eval = tqdm(total=n_samples_to_test, desc=description, leave=False)
    
    for step, batch in enumerate(loader):
        if total_tested >= n_samples_to_test:
            break
        
        # データ取得と整形
        # batch["encoded_image"] は (Batch, Time, 3, 32, 32)
        # ネットワークは (Time, Batch, ...) を期待するので permute する
        image = batch["encoded_image"]
        label = batch["label"]
        current_batch_size = image.size(0)
        
        # 時間軸を先頭に入れ替え: (Time, Batch, 3, 32, 32)
        inputs = {"X": image.permute(1, 0, 2, 3, 4).to(device)}
            
        # 実行
        network.run(inputs=inputs, time=sim_time)
        
        # スパイク取得: Ae層のスパイク (Time, Batch, Neurons) -> (Batch, Neurons) に集約
        # 時間方向の合計を取るかどうかは all_activity の仕様によるが、
        # ここでは全時刻のスパイク列 (Time, Batch, Neurons) を取得
        s = spikes["Ae"].get("s") # (Time, Batch, Neurons)
        
        # all_activity は (n_samples, n_time_steps, n_neurons) を期待している場合が多いが、
        # BindsNETの実装によっては (Time, Neurons) (バッチなし) の場合もある。
        # evaluation.py の all_activity を確認すると、dim=1 (時間) でsumしている。
        # バッチ対応させるために、一旦 (Batch, Time, Neurons) に戻して渡すのが安全
        s_batch = s.permute(1, 0, 2) # (Batch, Time, Neurons)
        
        label_tensor = label.to(device)
        
        # 予測
        # all_activity はバッチ対応している (dim=1でsumするので、(Batch, Time, Neurons)ならOK)
        all_activity_pred = all_activity(
            spikes=s_batch, assignments=assignments, n_labels=n_classes
        )
        
        # 正解数カウント
        total_correct += (label_tensor.long() == all_activity_pred).sum().item()
        total_tested += current_batch_size
        
        network.reset_state_variables()
        pbar_eval.update(current_batch_size)
    
    pbar_eval.close()
    network.train(mode=True)
    
    if total_tested == 0: return 0
    return total_correct / total_tested

# --- 学習ループ ---
print(f"Begin training. Target Accuracy: {target_accuracy*100:.1f}%")
print(f"Checking validation every {validation_interval} samples.")
print(f"Batch size: {batch_size}")

start_time = time.time()
total_samples_processed = 0
goal_reached = False

pbar = tqdm(total=max_samples, desc="Training")

while total_samples_processed < max_samples and not goal_reached:
    for i, datum in enumerate(train_loader):
        if total_samples_processed >= max_samples:
            break

        # データ取得
        image = datum["encoded_image"] # (Batch, Time, 3, 32, 32)
        label = datum["label"]         # (Batch)
        current_batch_size = image.size(0)

        # --- バリデーションチェック ---
        # 前回チェック時から validation_interval 以上経過していれば実行
        # (バッチ単位で進むため、厳密な割り切れではなく範囲で判定)
        if total_samples_processed > 0 and \
           (total_samples_processed // validation_interval) > \
           ((total_samples_processed - current_batch_size) // validation_interval):
            
            # ラベル割り当ての更新 (学習データの統計を使う)
            # spike_record は (update_interval, Time, Neurons) 
            # assign_labels は (n_samples, n_time_steps, n_neurons) を期待
            # ここでは直近の update_interval 個のサンプルを使う
            # バッチ学習だと spike_record への格納が少し複雑になるが、
            # 簡易的に「現在バッファにある分だけ」で更新する
            valid_samples = min(total_samples_processed, update_interval)
            
            # spike_record から有効な部分だけ取り出す
            # assignments の更新
            assignments, proportions, rates = assign_labels(
                spike_record[:valid_samples], 
                labels[:valid_samples], 
                n_classes, 
                rates
            )
            
            print(f"\nExecuting Validation (Sample {total_samples_processed})...")
            val_acc = run_evaluation(network, val_loader, n_val_samples, "Validation")
            print(f"Validation Acc: {val_acc*100:.2f}% (Total Spikes: {int(total_spikes)})")
            
            if val_acc >= target_accuracy:
                print("\nTarget reached! Checking Test Data...")
                test_acc = run_evaluation(network, test_loader, 10000, "Final Testing")
                print(f"Final Test Acc: {test_acc*100:.2f}%")
                
                if test_acc >= target_accuracy:
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    print("\n" + "="*30)
                    print(" GOAL REACHED ")
                    print("="*30)
                    print(f"Total Time     : {total_time:.4f} s")
                    print(f"Total Samples  : {total_samples_processed}")
                    print(f"Total Spikes   : {int(total_spikes)}")
                    print(f"Final Accuracy : {test_acc*100:.2f}%")
                    print("="*30 + "\n")
                    
                    goal_reached = True
                    break
                else:
                    print("Test accuracy insufficient. Resuming...")

        if goal_reached: break

        # --- 1. 入力データの準備 (バッチ対応) ---
        # (Batch, Time, C, H, W) -> (Time, Batch, C, H, W)
        inputs = {"X": image.permute(1, 0, 2, 3, 4).to(device)}

        # --- 2. クランプ（教師信号）の準備 (バッチ対応) ---
        # (Batch, Neurons) のマスクを作成
        clamp_mask = torch.zeros(current_batch_size, n_neurons, dtype=torch.bool, device=device)
        
        for b in range(current_batch_size):
            target_label = label[b].item()
            # クラスごとの担当ニューロン範囲
            start_idx = int(per_class * target_label)
            # ランダムに選択
            choice = np.random.choice(per_class, size=n_clamp, replace=False)
            clamp_idx = start_idx + choice
            # マスクをオン
            clamp_mask[b, clamp_idx] = True
            
        clamp = {"Ae": clamp_mask}

        # --- 3. ネットワーク実行 ---
        network.run(inputs=inputs, time=sim_time, clamp=clamp)

        # --- 4. スパイク数のカウント ---
        batch_spike_count = 0
        for layer in spikes:
            s = spikes[layer].get("s") # (Time, Batch, Neurons)
            batch_spike_count += s.sum().item()
        total_spikes += batch_spike_count

        # --- 5. ラベル割り当て用データの記録 ---
        # 現在のバッチのスパイクデータを spike_record バッファに書き込む
        # リングバッファのように扱う
        s_ae = spikes["Ae"].get("s").permute(1, 0, 2) # (Batch, Time, Neurons)
        
        # バッチ内の各サンプルについて
        for b in range(current_batch_size):
            idx = (total_samples_processed + b) % update_interval
            spike_record[idx] = s_ae[b]
            labels[idx] = label[b]

        # --- 6. プロット処理 ---
        if plot:
            # 重み取得の修正 (.w ではなく .params["weight"])
            # ※ MulticompartmentConnection対応
            input_exc_conn = network.connections[("X", "Ae")]
            if hasattr(input_exc_conn, "w"):
                w = input_exc_conn.w
            else:
                w = input_exc_conn.params["weight"]

            # 可視化用にデータを整形 (バッチの先頭だけ表示)
            # CIFAR画像 (Time, Batch, 3, 32, 32) -> (3, 32, 32) -> (32, 32) sum dim0
            inpt_view = inputs["X"][:, 0, :, :, :].sum(0).sum(0) # 簡易的にRGB合計
            
            # モニターデータ取得 (バッチの先頭)
            exc_voltages = exc_voltage_monitor.get("v")[:, 0, :] # (Time, Neurons)
            inh_voltages = inh_voltage_monitor.get("v")[:, 0, :]
            
            # スパイクデータ (Monitorは (Time, Batch, Neurons) を返す)
            s_plot = {layer: spikes[layer].get("s")[:, 0, :].unsqueeze(1) for layer in spikes}

            # 重みの可視化 (CIFAR対応は難しいため、エラー回避しつつ簡易表示)
            # get_square_weights は 2D (784, N) を想定しているため、
            # CIFAR (3072, N) を無理やり渡すと崩れるが、エラー回避のためreshape
            # 実際にはCIFARの重み可視化は専用関数が必要
            w_reshaped = w.view(3072, n_neurons)

            # プロット実行 (try-catchでエラー停止防止)
            try:
                inpt_axes, inpt_ims = plot_input(
                    image[0].sum(0), inpt_view, label=label[0], axes=inpt_axes, ims=inpt_ims
                )
                spike_ims, spike_axes = plot_spikes(
                    s_plot, ims=spike_ims, axes=spike_axes
                )
                voltage_ims, voltage_axes = plot_voltages(
                    {"Ae": exc_voltages, "Ai": inh_voltages}, ims=voltage_ims, axes=voltage_axes
                )
                # 重みプロットはCIFARでは形状が合わないためスキップ推奨だが、動く範囲で
                # weights_im = plot_weights(get_square_weights(w_reshaped, n_sqrt, 32), im=weights_im)
                
                plt.pause(1e-8)
            except Exception as e:
                print(f"Plot error (ignoring): {e}")

        network.reset_state_variables()
        total_samples_processed += current_batch_size
        pbar.set_description(f"Training (Spikes: {int(total_spikes)})")
        pbar.update(current_batch_size)

pbar.close()

if not goal_reached:
    print(f"\nMax samples reached without hitting target accuracy.")