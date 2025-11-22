import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import CIFAR10 
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

if __name__ == "__main__":
    # --- 引数設定 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_neurons", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
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

    # ベンチマーク用
    parser.add_argument("--target_accuracy", type=float, default=0.85)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--n_val_samples", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=300000)

    parser.set_defaults(plot=False, gpu=True)
    args = parser.parse_args()

    # 変数展開
    seed = args.seed
    n_neurons = args.n_neurons
    batch_size = args.batch_size
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

    # --- ネットワーク構築 ---
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
        device=device 
    )
    network.to(device)

    # モニター設定
    exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=sim_time, device=device)
    inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=sim_time, device=device)
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    # 【修正1】Monitorに device=device を追加
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=sim_time, device=device)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    # --- データセット準備 ---
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=gpu)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=gpu)

    # --- 変数初期化 ---
    spike_record = torch.zeros(update_interval, sim_time, n_neurons, device=device)
    assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
    proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
    rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
    accuracy = {"all": [], "proportion": []}
    labels = torch.empty(update_interval, device=device)

    total_spikes = 0 

    # --- 評価関数 ---
    def run_evaluation(network, loader, n_samples_to_test, description="Evaluation"):
        network.train(mode=False)
        total_correct = 0
        total_tested = 0
        pbar_eval = tqdm(total=n_samples_to_test, desc=description, leave=False)
        
        for step, batch in enumerate(loader):
            if total_tested >= n_samples_to_test:
                break
            
            image = batch["encoded_image"]
            label = batch["label"]
            current_batch_size = image.size(0)
            
            # network.run は評価時ならバッチ実行可能（クランプなしならエラー起きないため）
            # ただし入力形状は (Time, Batch, C, H, W) にする
            inputs = {"X": image.permute(1, 0, 2, 3, 4).to(device)}
            network.run(inputs=inputs, time=sim_time)
            
            s = spikes["Ae"].get("s") # (Time, Batch, Neurons)
            
            # 【修正2】時間次元を合計して (Batch, Neurons) にする
            s_batch = s.sum(0) 
            
            label_tensor = label.to(device)
            
            all_activity_pred = all_activity(
                spikes=s_batch, assignments=assignments, n_labels=n_classes
            )
            
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

            image = datum["encoded_image"]
            label = datum["label"]
            current_batch_size = image.size(0)

            # --- バリデーションチェック ---
            if total_samples_processed > 0 and \
            (total_samples_processed // validation_interval) > \
            ((total_samples_processed - current_batch_size) // validation_interval):
                
                valid_samples = min(total_samples_processed, update_interval)
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

            # --- 教師信号（クランプマスク） ---
            clamp_mask = torch.zeros(current_batch_size, n_neurons, dtype=torch.bool, device=device)
            for b in range(current_batch_size):
                target_label = label[b].item()
                start_idx = int(per_class * target_label)
                choice = np.random.choice(per_class, size=n_clamp, replace=False)
                clamp_idx = start_idx + choice
                clamp_mask[b, clamp_idx] = True

            # --- 手動学習ループ ---
            input_spikes = image.permute(1, 0, 2, 3, 4).to(device)

            if network.batch_size != current_batch_size:
                network.batch_size = current_batch_size
                for l in network.layers:
                    network.layers[l].set_batch_size(current_batch_size)
                for m in network.monitors:
                    network.monitors[m].reset_state_variables()

            timesteps = int(sim_time / dt)
            network.train(True)

            for t in range(timesteps):
                current_inputs = network._get_inputs()
                inp_t = input_spikes[t]
                if "X" in current_inputs:
                    current_inputs["X"] += inp_t
                else:
                    current_inputs["X"] = inp_t

                for l in network.layers:
                    if l in current_inputs:
                        network.layers[l].forward(x=current_inputs[l])
                    else:
                        network.layers[l].forward(
                            x=torch.zeros(network.layers[l].s.shape, device=device)
                        )

                    if l == "Ae":
                        network.layers[l].s.masked_fill_(clamp_mask, 1.0)

                for c in network.connections:
                    network.connections[c].update(mask=None, learning=True)

                for m in network.monitors:
                    network.monitors[m].record()

            for c in network.connections:
                network.connections[c].normalize()

            # --- スパイクカウント ---
            batch_spike_count = 0
            for layer in spikes:
                s = spikes[layer].get("s")
                batch_spike_count += s.sum().item()
            total_spikes += batch_spike_count

            # --- 記録更新 ---
            s_ae = spikes["Ae"].get("s").permute(1, 0, 2)
            for b in range(current_batch_size):
                idx = (total_samples_processed + b) % update_interval
                spike_record[idx] = s_ae[b]
                labels[idx] = label[b]

            # プロットは省略

            network.reset_state_variables()
            total_samples_processed += current_batch_size
            pbar.set_description(f"Training (Spikes: {int(total_spikes)})")
            pbar.update(current_batch_size)

    pbar.close()

    if not goal_reached:
        print(f"\nMax samples reached without hitting target accuracy.")