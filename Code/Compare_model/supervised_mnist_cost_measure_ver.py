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
from bindsnet.bindsnet.datasets import MNIST
from bindsnet.bindsnet.encoding import PoissonEncoder
from bindsnet.bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.bindsnet.models import DiehlAndCook2015
from bindsnet.bindsnet.network.monitors import Monitor
from bindsnet.bindsnet.utils import get_square_assignments, get_square_weights

# --- 引数設定 ---
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
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

# 追加した引数: 早期終了とベンチマーク用
parser.add_argument("--target_accuracy", type=float, default=0.85, help="Target accuracy to stop training (0.0-1.0)")
parser.add_argument("--validation_interval", type=int, default=1000, help="Interval to check validation accuracy")
parser.add_argument("--n_val_samples", type=int, default=1000, help="Number of samples to use for fast validation")
parser.add_argument("--max_samples", type=int, default=300000, help="Maximum number of training samples before giving up")

parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
device_id = args.device_id

# 新しい設定値
target_accuracy = args.target_accuracy
validation_interval = args.validation_interval
n_val_samples = args.n_val_samples
max_samples = args.max_samples

# --- GPU設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

n_classes = 10
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
per_class = int(n_neurons / n_classes)

# --- ネットワーク構築 (Diehl & Cook 2015) ---
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

if gpu:
    network.to("cuda")

# モニター設定
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# --- データセットの準備と分割 ---
# 1. 全訓練データの読み込み
full_train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# 2. 訓練データ(50000)と検証データ(10000)に分割
train_size = 50000
val_size = 10000
# MNIST train is 60000. Ensure split matches.
if len(full_train_dataset) != 60000:
    # サブセットなどで数が合わない場合の安全策
    val_size = int(len(full_train_dataset) * 0.1)
    train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 3. テストデータの読み込み（最終確認用）
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# DataLoaderの作成
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# --- 変数初期化 ---
spike_record = torch.zeros(update_interval, time, n_neurons, device=device)
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
accuracy = {"all": [], "proportion": []}
labels = torch.empty(update_interval, device=device)

# プロット用変数
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

# --- ヘルパー関数: 評価実行用 ---
def run_evaluation(network, loader, n_samples_to_test, description="Evaluation"):
    """
    指定されたローダーを使って評価を行う関数
    学習モードを一時的にオフにして実行する
    """
    # ネットワークを評価モードへ（学習停止）
    network.train(mode=False)
    
    total_correct = 0
    pbar_eval = tqdm(total=n_samples_to_test, desc=description, leave=False)
    
    eval_spike_record = torch.zeros(1, int(time / dt), n_neurons, device=device)
    
    count = 0
    for step, batch in enumerate(loader):
        if count >= n_samples_to_test:
            break
            
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # ネットワーク実行
        network.run(inputs=inputs, time=time)
        
        # スパイク記録
        eval_spike_record[0] = spikes["Ae"].get("s").squeeze()
        label_tensor = torch.tensor(batch["label"], device=device)
        
        # 予測
        all_activity_pred = all_activity(
            spikes=eval_spike_record, assignments=assignments, n_labels=n_classes
        )
        
        if label_tensor.long() == all_activity_pred:
            total_correct += 1
            
        network.reset_state_variables()
        count += 1
        pbar_eval.update()
    
    pbar_eval.close()
    
    # ネットワークを学習モードに戻す
    network.train(mode=True)
    
    return total_correct / count

# --- 学習ループ開始 ---
print(f"Begin training with Target Accuracy: {target_accuracy*100:.1f}%")
print(f"Checking validation every {validation_interval} samples.\n")

start_time = time.time()
total_samples_processed = 0
goal_reached = False

# 無限ループ的に回すため、外側に while を配置
pbar = tqdm(total=max_samples, desc="Training Progress")

while total_samples_processed < max_samples and not goal_reached:
    for i, datum in enumerate(train_loader):
        if total_samples_processed >= max_samples:
            break

        # データの取り出し
        image = datum["encoded_image"]
        label = datum["label"]

        # --- ラベルの更新と精度の記録 (Training Accuracy) ---
        # update_interval ごとに、それまでの spike_record を使ってニューロンの役割(assignments)を更新する
        if total_samples_processed % update_interval == 0 and total_samples_processed > 0:
            # 予測精度の計算 (直近 update_interval 分の訓練データに対して)
            all_activity_pred = all_activity(spike_record, assignments, n_classes)
            proportion_pred = proportion_weighting(
                spike_record, assignments, proportions, n_classes
            )

            acc_all = 100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
            acc_prop = 100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
            
            accuracy["all"].append(acc_all)
            accuracy["proportion"].append(acc_prop)

            # ログ出力
            print(
                f"\n[Train Acc] All: {acc_all:.2f} (avg: {np.mean(accuracy['all']):.2f}) | "
                f"Prop: {acc_prop:.2f} (avg: {np.mean(accuracy['proportion']):.2f})"
            )

            # 重要: ニューロンへのラベル割り当てを更新
            assignments, proportions, rates = assign_labels(
                spike_record, labels, n_classes, rates
            )

        # --- バリデーションチェック ---
        # validation_interval ごとに検証を行う
        # ※重要: 少なくとも一度は assign_labels が走った後でないと意味がないため > 0
        if total_samples_processed % validation_interval == 0 and total_samples_processed > 0:
            
            # バリデーション時間の計測は除外したい場合はここで time.time() を操作するが、
            # 実用コストとしては含めるのが一般的。
            print(f"\nExecuting Validation (Sample {total_samples_processed})...")
            
            val_acc = run_evaluation(
                network, 
                val_loader, 
                n_samples_to_test=n_val_samples, 
                description="Validation"
            )
            
            print(f"Validation Accuracy: {val_acc*100:.2f}% (Target: {target_accuracy*100:.1f}%)")
            
            if val_acc >= target_accuracy:
                print("\nValidation target reached! Checking Test Data for confirmation...")
                
                # 最終テスト (全データまたは十分な量で)
                test_acc = run_evaluation(
                    network, 
                    test_loader, 
                    n_samples_to_test=10000, # 全テストデータ
                    description="Final Testing"
                )
                
                print(f"Final Test Accuracy: {test_acc*100:.2f}%")
                
                if test_acc >= target_accuracy:
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    print("\n" + "="*30)
                    print(" GOAL REACHED ")
                    print("="*30)
                    print(f"Total Time Cost  : {total_time:.4f} seconds")
                    print(f"Total Samples    : {total_samples_processed}")
                    print(f"Final Accuracy   : {test_acc*100:.2f}%")
                    print("="*30 + "\n")
                    
                    goal_reached = True
                    break
                else:
                    print("Test accuracy did not meet target. Resuming training...\n")


        # --- ネットワークの実行 (学習ステップ) ---
        # 直近のラベルを記録
        labels[total_samples_processed % update_interval] = label[0]

        # クランプの設定 (教師あり学習)
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        
        if gpu:
            inputs = {"X": image.cuda().view(time, 1, 1, 28, 28)}
        else:
            inputs = {"X": image.view(time, 1, 1, 28, 28)}
            
        # STDP学習実行
        network.run(inputs=inputs, time=time, clamp=clamp)

        # 電圧などの取得（プロット用）
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # スパイク記録（ラベル割り当て用）
        spike_record[total_samples_processed % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

        # --- プロット処理 ---
        if plot:
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

            inpt_axes, inpt_ims = plot_input(
                image.sum(1).view(28, 28), inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(
                {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
                ims=spike_ims,
                axes=spike_axes,
            )
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1e-8)

        # 状態リセットとカウンタ更新
        network.reset_state_variables()
        total_samples_processed += 1
        pbar.update(1)

pbar.close()

if not goal_reached:
    print(f"\nMaximum samples ({max_samples}) reached without achieving target accuracy.")