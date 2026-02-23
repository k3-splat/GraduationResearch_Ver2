import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 現在のディレクトリをパスに追加
sys.path.append(os.getcwd())

try:
    import Code.Diff_bPC.otehon.Diff_PC as bu
    import topdown_prediction as td
except ImportError:
    print("エラー: 'bottomup_measure.py' または 'topdown_prediction.py' が見つかりません。")
    sys.exit(1)

def get_hidden_xt(net):
    """隠れ層(layers[0])の状態を取得"""
    return net.layers[0].x_T.detach().cpu()

def run_comparison_final():
    # =========================
    # 1. 設定
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    hidden_dim = 400
    epochs = 5
    seed = 42
    
    # 保存用ディレクトリ作成
    save_dir = "comparison_results_final_1"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {device}, Saving results to: {save_dir}/")

    # =========================
    # 2. データセット (Train & Test)
    # =========================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    torch.manual_seed(seed)
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # =========================
    # 3. モデル初期化
    # =========================
    # --- Bottom-Up (識別) ---
    cfg_bu = bu.DiffPCConfig(
        layer_dims=[784, hidden_dim, 10],
        batch_size=batch_size, seed=seed, device=device,
        pc_lr=0.0001, epochs=epochs
    )
    lt_spec_bu = {"type": "cyclic_phase", "args": {"m": 0, "n": 5, "a": 1.0}}
    y_spec_bu = {"type": "on_cycle_start", "args": {"gamma": 0.05, "n": 5}}
    net_bu = bu.DiffPCNetworkTorch(cfg_bu, lt_spec_bu, y_spec_bu, device)

    # --- Top-Down (生成) ---
    cfg_td = td.DiffPCConfig(
        layer_dims=[10, hidden_dim, 784],
        batch_size=batch_size, seed=seed, device=device,
        pc_lr=0.0005, epochs=epochs,
        alpha_in=1.0, alpha_out=0.01
    )
    lt_spec_td = {"type": "cyclic_phase", "args": {"m": 0, "n": 5, "a": 1.0}}
    y_spec_td = {"type": "on_cycle_start", "args": {"gamma": 0.1, "n": 5}}
    net_td = td.DiffPCNetworkTorch(cfg_td, lt_spec_td, y_spec_td, device)

    def to_vec(x): return x.view(x.size(0), -1).to(device)
    def to_onehot(y, n_cls): return F.one_hot(y, num_classes=n_cls).float().to(device)

    # =========================
    # 4. 学習ループ
    # =========================
    history = {
        "epoch": [], 
        "bu_acc": [], 
        "bu_sparsity": [], "td_sparsity": [],
        "bu_mean": [], "td_mean": []
    }

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # --- A. 学習フェーズ ---
        bu_acts, td_acts = [], []
        
        for i, (images, labels) in enumerate(train_loader):
            # Bottom-Up 学習
            bu.run_batch_two_phase(net_bu, to_vec(images), to_onehot(labels, 10), cfg_bu, lt_spec_bu, y_spec_bu)
            
            # Top-Down 学習
            td.run_gen_training_batch(net_td, to_onehot(labels, 10), to_vec(images), cfg_td, lt_spec_td, y_spec_td)
            
            # 統計用データの収集 (20バッチに1回)
            if i % 20 == 0:
                bu_acts.append(get_hidden_xt(net_bu))
                td_acts.append(get_hidden_xt(net_td))

        # --- B. 内部状態の分析 & ヒストグラム描画 (復活) ---
        all_bu = torch.cat(bu_acts, dim=0)
        all_td = torch.cat(td_acts, dim=0)
        
        # 統計量
        sp_bu = (all_bu.abs() < 0.01).float().mean().item()
        sp_td = (all_td.abs() < 0.01).float().mean().item()
        mu_bu = all_bu.abs().mean().item()
        mu_td = all_td.abs().mean().item()

        history["bu_sparsity"].append(sp_bu)
        history["td_sparsity"].append(sp_td)
        history["bu_mean"].append(mu_bu)
        history["td_mean"].append(mu_td)
        
        # ヒストグラム保存
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(all_bu.flatten().numpy(), bins=100, alpha=0.6, color='blue', label='Bottom-Up', density=True, range=(-2, 2))
        plt.hist(all_td.flatten().numpy(), bins=100, alpha=0.6, color='orange', label='Top-Down', density=True, range=(-2, 2))
        plt.title(f"Distribution of $x_T$ (Epoch {epoch})")
        plt.xlabel("Value"); plt.yscale('log'); plt.legend()

        plt.subplot(1, 2, 2)
        mask_bu = all_bu.abs() > 0.1
        mask_td = all_td.abs() > 0.1
        if mask_bu.sum() > 0:
            plt.hist(all_bu[mask_bu].numpy(), bins=50, alpha=0.6, color='blue', label='BU (Active)', density=True)
        if mask_td.sum() > 0:
            plt.hist(all_td[mask_td].numpy(), bins=50, alpha=0.6, color='orange', label='TD (Active)', density=True)
        plt.title("Active Neurons (>0.1)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dist_epoch_{epoch:02d}.png")
        plt.close()
        print(f"  [Analysis] Histograms saved. Sparsity BU:{sp_bu:.3f}, TD:{sp_td:.3f}")

        # --- C. 評価フェーズ (精度 & 生成画像) ---
        
        # 1. 精度
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits, _ = bu.infer_batch_forward_only(net_bu, to_vec(images), cfg_bu, lt_spec_bu)
                preds = logits.argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        history["bu_acc"].append(acc)
        print(f"  [Bottom-Up] Test Accuracy: {acc:.2f}%")

        # 2. 生成画像
        with torch.no_grad():
            sample_labels = torch.arange(10, device=device)
            sample_onehot = to_onehot(sample_labels, 10)
            generated_flat, _ = td.infer_batch_forward_only(net_td, sample_onehot, cfg_td, lt_spec_td)
            
            imgs = generated_flat.view(-1, 1, 28, 28)
            imgs = imgs * 0.3081 + 0.1307
            imgs = torch.clamp(imgs, 0, 1)
            
            save_image(imgs, f"{save_dir}/gen_epoch_{epoch:02d}.png", nrow=10)
            print(f"  [Top-Down] Generated images saved.")

        # === 修正箇所: epochを記録 ===
        history["epoch"].append(epoch)

    # =========================
    # 5. 最終サマリーグラフ
    # =========================
    plt.figure(figsize=(15, 4))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history["epoch"], history["bu_acc"], 'o-', label='BU Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
    
    # Sparsity
    plt.subplot(1, 3, 2)
    plt.plot(history["epoch"], history["bu_sparsity"], 'o-', label='BU Sparsity')
    plt.plot(history["epoch"], history["td_sparsity"], 'o-', label='TD Sparsity')
    plt.xlabel('Epoch'); plt.ylabel('Sparsity Rate'); plt.legend(); plt.grid(True)

    # Mean Activity
    plt.subplot(1, 3, 3)
    plt.plot(history["epoch"], history["bu_mean"], 'o-', label='BU Mean |x|')
    plt.plot(history["epoch"], history["td_mean"], 'o-', label='TD Mean |x|')
    plt.xlabel('Epoch'); plt.ylabel('Mean Activity'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_metrics_summary.png")
    print(f"\nAll Done. Results are in '{save_dir}'")

if __name__ == "__main__":
    run_comparison_final()