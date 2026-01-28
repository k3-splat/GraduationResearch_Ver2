#!/usr/bin/env python
# coding: utf-8

# # Bidirectional Predictive Coding
# Modified to support epoch-based training

import jpc
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')  # ignore warnings

# ## Hyperparameters

SEED = 0

INPUT_DIM = 10
WIDTH = 400
DEPTH = 2
OUTPUT_DIM = 784
ACT_FN = "leaky_relu"

ACTIVITY_LR = 5e-1
PARAM_LR = 1e-3
BATCH_SIZE = 256
TEST_EVERY = 50       # ログ出力の頻度（ステップ数）
N_EPOCHS = 10         # 学習するエポック数

# ## Dataset

def get_mnist_loaders(batch_size):
    train_data = MNIST(train=True, normalise=True)
    test_data = MNIST(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_loader, test_loader


class MNIST(datasets.MNIST):
    def __init__(self, train, normalise=True, save_dir="data"):
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.1307), std=(0.3081)
                    )
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label)
        return img, label


def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


def plot_mnist_imgs(imgs, labels, n_imgs=10):
    plt.figure(figsize=(20, 2))
    for i in range(n_imgs):
        plt.subplot(1, n_imgs, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i].reshape(28, 28), cmap=plt.cm.binary_r)
        plt.xlabel(jnp.argmax(labels, axis=1)[i])
    plt.show()


# ## Train and test

def evaluate(generator, amortiser, test_loader):
    amort_accs = 0.
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        preds = jpc.init_activities_with_ffwd(
            model=amortiser[::-1],
            input=img_batch
        )[-1]
        amort_accs += jpc.compute_accuracy(label_batch, preds)

    img_preds = jpc.init_activities_with_ffwd(
        model=generator,
        input=label_batch
    )[-1]

    return (
        amort_accs / len(test_loader),
        label_batch,
        img_preds
    )


def train(
      seed,
      input_dim,
      width,
      depth,
      output_dim,
      act_fn,
      batch_size,
      activity_lr,
      param_lr,
      test_every,
      n_epochs,
      forward_energy_weight=1e-4
):
    key = jax.random.PRNGKey(seed)
    gen_key, amort_key = jax.random.split(key, 2)

    # models (NOTE: input and output are inverted for the amortiser)
    generator = jpc.make_mlp(
        gen_key, 
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn
    )
    amortiser = jpc.make_mlp(
        amort_key,
        input_dim=output_dim,
        width=width,
        depth=depth,
        output_dim=input_dim,
        act_fn=act_fn
    )[::-1]
        
    # optimisers
    activity_optim = optax.sgd(activity_lr)
    gen_optim = optax.adamw(param_lr)
    amort_optim = optax.adamw(param_lr)
    
    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))

    # data
    train_loader, test_loader = get_mnist_loaders(batch_size)
    
    global_step = 0
    print(f"Start training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        for batch_idx, (img_batch, label_batch) in enumerate(train_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            
            # discriminative loss & initialisation
            activities = jpc.init_activities_with_ffwd(
                model=amortiser[::-1],
                input=img_batch
            )
            amort_loss = jpc.mse_loss(activities[-1], label_batch)
            activity_opt_state = activity_optim.init(activities)

            # generative loss
            gen_activities = jpc.init_activities_with_ffwd(
                model=generator,
                input=label_batch
            )
            gen_loss = jpc.mse_loss(gen_activities[-1], img_batch)

            # inference
            for _ in range(8):
                activity_update_result = jpc.update_bpc_activities(
                    top_down_model=generator,
                    bottom_up_model=amortiser,
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=img_batch,
                    input=label_batch,
                    forward_energy_weight=forward_energy_weight
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]

            # learning
            param_update_result = jpc.update_bpc_params(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                top_down_optim=gen_optim,
                bottom_up_optim=amort_optim,
                top_down_opt_state=gen_opt_state,
                bottom_up_opt_state=amort_opt_state,
                output=img_batch,
                input=label_batch,
                forward_energy_weight=forward_energy_weight
            )
            generator, amortiser = param_update_result["models"]
            gen_opt_state, amort_opt_state  = param_update_result["opt_states"]

            global_step += 1

            if (global_step % test_every) == 0:
                amort_acc, label_batch, img_preds = evaluate(
                    generator,
                    amortiser,
                    test_loader
                )
                print(
                    f"Epoch {epoch+1}/{n_epochs} (Step {global_step}), "
                    f"gen loss={gen_loss:.4f}, "
                    f"amort loss={amort_loss:.4f}, "
                    f"avg amort test accuracy={amort_acc:.4f}"
                )
    
    print("Training finished.")            
    plot_mnist_imgs(img_preds, label_batch)


# ## Run

if __name__ == "__main__":
    # To achieve SOTA classification performance, [Olivers et al., 2025] showed that one can simply 
    # decrease the relative weight of the forward energies (tied to generation), or equivalently, 
    # upweight the backward energies (tied to classification). 
    # We reproduce this below by downscaling the forward energies by a factor of alpha_f = 1e-3.

    train(
        seed=SEED,
        input_dim=INPUT_DIM,
        width=WIDTH,
        depth=DEPTH,
        output_dim=OUTPUT_DIM,
        act_fn=ACT_FN,
        batch_size=BATCH_SIZE,
        activity_lr=ACTIVITY_LR,
        param_lr=PARAM_LR,
        test_every=TEST_EVERY,
        n_epochs=N_EPOCHS,
        forward_energy_weight=1e-3
    )