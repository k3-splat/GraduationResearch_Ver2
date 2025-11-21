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
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
sim_time = args.time
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
# CIFAR-10用にパラメータを変更
# 入力サイズ: 32x32x3 = 3072
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
    device=device # GPUエラー対策
)

if gpu:
    network.to("cuda")

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
full_train_dataset = CIFAR10(
    PoissonEncoder(time=sim_time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "CIFAR10"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

train_size = 45000 # CIFAR-10 train is 50000
val_size = 5000
if len(full_train_dataset) != 50000:
    val_size = int(len(full_train_dataset) * 0.1)
    train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset = CIFAR10(
    PoissonEncoder(time=sim_time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "CIFAR10"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# --- 変数初期化 ---
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
    pbar_eval = tqdm(total=n_samples_to_test, desc=description, leave=False)
    
    eval_spike_record = torch.zeros(1, int(sim_time / dt), n_neurons, device=device)
    
    count = 0
    for step, batch in enumerate(loader):
        if count >= n_samples_to_test:
            break
            
        # CIFAR-10は (3, 32, 32) なので viewのサイズを調整
        inputs = {"X": batch["encoded_image"].view(int(sim_time / dt), 1, 3, 32, 32)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        network.run(inputs=inputs, time=sim_time)
        
        eval_spike_record[0] = spikes["Ae"].get("s").squeeze()
        label_tensor = torch.tensor(batch["label"], device=device)
        
        all_activity_pred = all_activity(
            spikes=eval_spike_record, assignments=assignments, n_labels=n_classes
        )
        
        if label_tensor.long() == all_activity_pred:
            total_correct += 1
            
        network.reset_state_variables()
        count += 1
        pbar_eval.update()
    
    pbar_eval.close()
    network.train(mode=True)
    return total_correct / count

# --- 学習ループ ---
print(f"Begin training. Target Accuracy: {target_accuracy*100:.1f}%")
print(f"Checking validation every {validation_interval} samples.")

start_time = time.time()
total_samples_processed = 0
goal_reached = False

pbar = tqdm(total=max_samples, desc="Training")

while total_samples_processed < max_samples and not goal_reached:
    for i, datum in enumerate(train_loader):
        if total_samples_processed >= max_samples:
            break

        image = datum["encoded_image"]
        label = datum["label"]

        # バリデーションチェック
        if total_samples_processed % validation_interval == 0 and total_samples_processed > 0:
            assignments, proportions, rates = assign_labels(
                spike_record, labels, n_classes, rates
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

        # ネットワーク実行
        labels[total_samples_processed % update_interval] = label[0]
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        
        if gpu:
            # CIFAR-10: (time, batch, channel, height, width) -> (time, 1, 3, 32, 32)
            inputs = {"X": image.cuda().view(sim_time, 1, 3, 32, 32)}
        else:
            inputs = {"X": image.view(sim_time, 1, 3, 32, 32)}
            
        network.run(inputs=inputs, time=sim_time, clamp=clamp)

        # スパイク数のカウント
        batch_spike_count = 0
        for layer in spikes:
            s = spikes[layer].get("s")
            batch_spike_count += s.sum().item()
        
        total_spikes += batch_spike_count

        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        spike_record[total_samples_processed % update_interval] = spikes["Ae"].get("s").view(sim_time, n_neurons)

        if plot:
            # プロット設定もCIFAR用に調整が必要だが、複雑になるため
            # 入力画像の表示のみ修正 (3チャンネルを合計して表示する簡易対応)
            inpt = inputs["X"].view(sim_time, 3072).sum(0).view(3, 32, 32).sum(0) 
            
            # 重みの可視化は単純な変形では崩れるため、注意が必要
            # input_exc_weights = network.connections[("X", "Ae")].params["weight"]
            # square_weights = get_square_weights(...) 
            
            # 簡易プロット（入力のみ）
            inpt_axes, inpt_ims = plot_input(
                image.sum(1).view(3, 32, 32).sum(0), inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            # 他のプロット関数はエラーになる可能性があるため、
            # CIFAR-10の場合は --plot をFalseにすることを推奨します
            plt.pause(1e-8)

        network.reset_state_variables()
        total_samples_processed += 1
        pbar.set_description(f"Training (Spikes: {int(total_spikes)})")
        pbar.update(1)

pbar.close()

if not goal_reached:
    print(f"\nMax samples reached. Final Acc: {accuracy['all'][-1] if accuracy['all'] else 0:.2f}%")