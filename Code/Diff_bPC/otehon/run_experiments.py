import torch.multiprocessing as mp
import time
from queue import Empty
from diffpc_flip import DiffPCConfig, main

# ==========================================
# 1. Common Configuration
# ==========================================
COMMON_CONFIG = {
    "lt_m": 0,
    "lt_a": 1.0,
    "lt_scheduler_type": "cyclic_phase",
    "gamma_every_n": None,
    "t_init_cycles": 15,
    "phase2_cycles": 15,
    "pc_lr": 0.0001,
    "batch_size": 256,
    "epochs": 200,
    "use_adamw": True,
    "adamw_weight_decay": 0.01,
    "adamw_betas": (0.9, 0.999),
    "adamw_eps": 1e-08,
    "clip_grad_norm": 1.0,
    "dropout_rate": 0.5,
    "v1_dropout": False,
}

SEEDS = [1, 2, 3]

# ==========================================
# 2. Experiment Definitions
# ==========================================
experiment_setups = [
    # 1. MNIST with a 784-400-10 Network
    {
        "name_base": "mnist_400h",
        "params": {
            "layer_dims": [784, 400, 10],
            "lt_n": 5,
            "gamma_value": 0.05,
            "use_fashion_mnist": False,
            "random_crop_padding": 2,
            "normalize": True,
            "fmnist_hflip_p": 0.0,
        }
    },
    # 2. MNIST with a 784-1024-512-10 Network
    {
        "name_base": "mnist_1024_512h",
        "params": {
            "layer_dims": [784, 1024, 512, 10],
            "lt_n": 8,
            "gamma_value": 0.05,
            "use_fashion_mnist": False,
            "random_crop_padding": 2,
            "normalize": True,
            "fmnist_hflip_p": 0.0,
        }
    },
    # 3. Fashion-MNIST with a 784-400-10 Network
    {
        "name_base": "fmnist_400h",
        "params": {
            "layer_dims": [784, 400, 10],
            "lt_n": 8,
            "gamma_value": 0.2,
            "use_fashion_mnist": True,
            "random_crop_padding": 0,
            "normalize": False,
            "fmnist_hflip_p": 0.0,
        }
    },
    # 4. Fashion-MNIST with a 784-1000-10 Network and Augmentation
    {
        "name_base": "fmnist_1000h_augment",
        "params": {
            "layer_dims": [784, 1000, 10],
            "lt_n": 8,
            "gamma_value": 0.2,
            "use_fashion_mnist": True,
            "random_crop_padding": 0,
            "normalize": False,
            "fmnist_hflip_p": 0.5,
        }
    }
]

# ==========================================
# 3. Worker Function
# ==========================================
def run_worker(gpu_id, task_queue, worker_id):
    """
    Pulls a config from the queue and runs it on the specific GPU.
    """
    print(f"[Worker {worker_id}] Initialized on GPU {gpu_id}")
    
    while True:
        try:
            # Get task with a generic timeout to allow clean exit if empty
            # (Though we check .empty() first, race conditions exist)
            config_dict = task_queue.get(timeout=1)
        except Empty:
            print(f"[Worker {worker_id}] Queue empty. Exiting.")
            break

        run_name = config_dict['run_name']
        print(f"[Worker {worker_id}] STARTING: {run_name} on cuda:{gpu_id}")

        # Update the device for this specific worker
        config_dict['device'] = f"cuda:{gpu_id}"
        
        # Create Config Object
        cfg = DiffPCConfig(**config_dict)

        try:
            # Run the experiment
            main(cfg)
            print(f"[Worker {worker_id}] FINISHED: {run_name}")
        except Exception as e:
            print(f"!!! [Worker {worker_id}] CRASHED on {run_name}: {e}")
            import traceback
            traceback.print_exc()

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # IMPORTANT: 'spawn' is required for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)

    # 1. Build the Task Queue
    task_queue = mp.Queue()
    
    total_tasks = 0
    for exp in experiment_setups:
        for seed in SEEDS:
            # Combine settings
            full_params = COMMON_CONFIG.copy()
            full_params.update(exp['params'])
            full_params['seed'] = seed
            full_params['run_name'] = f"{exp['name_base']}_seed{seed}"
            
            task_queue.put(full_params)
            total_tasks += 1

    print(f"Loaded {total_tasks} experiments into the queue.")

    # 2. Define Hardware Allocation
    # We want 3 workers on GPU 0, and 3 workers on GPU 1
    workers = []
    
    # GPU 0 Workers
    for i in range(3):
        p = mp.Process(target=run_worker, args=(0, task_queue, f"0-{i+1}"))
        workers.append(p)
        
    # GPU 1 Workers
    for i in range(3):
        p = mp.Process(target=run_worker, args=(1, task_queue, f"1-{i+1}"))
        workers.append(p)

    # 3. Start Processes
    print("Starting 6 concurrent workers (3 per GPU)...")
    for p in workers:
        p.start()
        # Slight stagger to prevent all file I/O or startup spikes hitting at exact same ms
        time.sleep(1) 

    # 4. Wait for completion
    for p in workers:
        p.join()

    print("\nAll parallel experiments completed.")