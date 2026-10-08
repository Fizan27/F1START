# check_gpu.py
#
# What it does: proves that PyTorch can really TRAIN on this machine's GPU,
#   not just see it. It trains the same small neural network on the CPU and on
#   the GPU, checks that the loss goes down on both, and compares the speed.
# What it reads: nothing.
# What it produces: a short report printed to the screen.
# Which files use it: none. Run it once after setup, or any time the GPU
#   seems not to be working:  .venv\Scripts\python.exe check_gpu.py

import time

import torch
from torch import nn

# Sized so the CPU run takes well under a minute. Bigger problems favour the
# GPU even more, but the check should be quick to run.
ROWS = 50_000  # fake training examples
FEATURES = 64  # numbers describing each example
STEPS = 50  # training steps to time


def print_gpu_details():
    """Print what PyTorch knows about the GPU, so setup problems are obvious."""
    print(f"PyTorch version : {torch.__version__}")
    print(f"Built for CUDA  : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU architecture: sm_{major}{minor}")
    # RTX 50 series is sm_120. If that is missing from this list, the PyTorch
    # build is too old for the card and training would crash.
    print(f"Build supports  : {', '.join(torch.cuda.get_arch_list())}")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory      : {total_gb:.1f} GB")


def make_fake_data(device):
    """Make a regression problem with a known answer, directly on the device."""
    generator = torch.Generator(device=device).manual_seed(0)
    inputs = torch.randn(ROWS, FEATURES, device=device, generator=generator)
    true_weights = torch.randn(FEATURES, 1, device=device, generator=generator)
    targets = inputs @ true_weights
    return inputs, targets


def train(device):
    """Train a small network. Returns (first loss, last loss, seconds taken)."""
    torch.manual_seed(0)  # same starting weights on CPU and GPU, for a fair race
    inputs, targets = make_fake_data(device)
    model = nn.Sequential(
        nn.Linear(FEATURES, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    start = time.perf_counter()
    for _ in range(STEPS):
        loss = nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if device == "cuda":
        # GPU work runs in the background; wait for it so the timing is honest.
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return losses[0], losses[-1], seconds


def main():
    if not torch.cuda.is_available():
        print("FAIL: PyTorch cannot see a CUDA GPU. Check the driver and the")
        print("PyTorch build (RTX 50 series needs a CUDA 12.8 or newer build).")
        raise SystemExit(1)

    print_gpu_details()
    print(f"\nTraining a small network for {STEPS} steps on {ROWS:,} rows...")

    train("cuda")  # warm up: the first GPU run includes one-off startup cost
    gpu_first, gpu_last, gpu_seconds = train("cuda")
    gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    cpu_first, cpu_last, cpu_seconds = train("cpu")

    print(f"\n{'device':<8}{'first loss':>12}{'last loss':>12}{'seconds':>10}")
    print(f"{'GPU':<8}{gpu_first:>12.3f}{gpu_last:>12.3f}{gpu_seconds:>10.2f}")
    print(f"{'CPU':<8}{cpu_first:>12.3f}{cpu_last:>12.3f}{cpu_seconds:>10.2f}")
    print(f"\nGPU memory used at peak: {gpu_peak_mb:.0f} MB")
    print(f"GPU speed-up over CPU  : {cpu_seconds / gpu_seconds:.1f}x")

    learned = gpu_last < gpu_first * 0.5
    used_gpu_memory = gpu_peak_mb > 0
    if learned and used_gpu_memory:
        print("\nPASS: training ran on the GPU and the loss went down.")
    else:
        print("\nFAIL: the GPU run did not learn or did not use GPU memory.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
