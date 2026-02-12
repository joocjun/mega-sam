# MegaSaM: Slurm Cluster Setup Guide

This guide walks through installing MegaSaM on a Slurm cluster with NVIDIA GPUs and CUDA 12.x drivers. It was tested on a cluster with A100 80GB GPUs, CUDA 12.4 driver, and Miniconda.

## Prerequisites

- Slurm cluster with NVIDIA GPUs (A100 or similar)
- CUDA 12.x driver on compute nodes (verified with `nvidia-smi`)
- Miniconda or Anaconda installed
- Git with SSH access to GitHub

## 1. Clone the Repository

```bash
git clone --recursive git@github.com:mega-sam/mega-sam.git
cd mega-sam
```

> **Important:** The `--recursive` flag is required to pull the `base` submodule (contains lietorch and DROID-SLAM backends).

If you already cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

## 2. Create the Conda Environment

```bash
conda create -n mega_sam python=3.10 -y
conda activate mega_sam
```

## 3. Install PyTorch with CUDA 12.x

The original `environment.yml` targets CUDA 11.8 + PyTorch 2.0.1, which won't match most modern Slurm clusters. Install PyTorch with CUDA 12.x support instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** PyTorch `cu126` wheels ship their own CUDA runtime libraries, so they work even if the system CUDA driver is 12.4. The driver just needs to be >= 12.0. If you see a warning about CUDA version mismatch during compilation, it is safe to ignore.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 4. Install Python Dependencies

```bash
pip install \
    opencv-python-headless==4.9.0.80 \
    tqdm \
    imageio \
    einops \
    scipy \
    matplotlib \
    wandb \
    timm \
    ninja \
    numpy==1.26.3 \
    huggingface-hub==0.23.4 \
    kornia
```

## 5. Install xformers

On Slurm clusters, install xformers from the PyTorch wheel index:

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
```

## 6. Install torch-scatter

torch-scatter must be compiled from source to match your PyTorch version. The build requires torch to be importable, so use `--no-build-isolation`:

```bash
pip install torch-scatter --no-build-isolation
```

### Troubleshooting torch-scatter

**`ModuleNotFoundError: No module named 'torch'` during build:**

This happens when pip's build isolation creates a clean virtualenv that doesn't have torch. The fix is `--no-build-isolation` as shown above.

**`ImportError: libnvshmem_host.so.3: cannot open shared object file`:**

PyTorch 2.10+ depends on `nvidia-nvshmem-cu12`. Install it:

```bash
pip install nvidia-nvshmem-cu12
```

**`BackendUnavailable: Cannot import 'setuptools.build_meta'`:**

This often happens when pip from `~/.local` (user site-packages) shadows the conda env's pip. Fix by prefixing commands with `PYTHONNOUSERSITE=1`:

```bash
PYTHONNOUSERSITE=1 python -m pip install torch-scatter --no-build-isolation
```

## 7. Fix Source Code for PyTorch 2.x Compatibility

The CUDA C++ extensions in `base/` use a deprecated PyTorch API (`.type()` in dispatch macros) that was removed in PyTorch 2.x. You must patch the source files before compiling.

### 7a. Fix `base/src/correlation_kernels.cu`

Replace `volume.type()` with `volume.scalar_type()` on two lines:

```diff
- AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
+ AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_forward_kernel", ([&] {
```

```diff
- AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {
+ AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_backward_kernel", ([&] {
```

### 7b. Fix `base/src/altcorr_kernel.cu`

```diff
- AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.type(), "altcorr_forward_kernel", ([&] {
+ AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "altcorr_forward_kernel", ([&] {
```

### 7c. Fix `base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu`

Replace all `.type()` calls inside `DISPATCH_GROUP_AND_FLOATING_TYPES` macros with `.scalar_type()`. There are 19 occurrences (2x `a.type()` and 17x `X.type()`):

```bash
cd base/thirdparty/lietorch/lietorch/src
sed -i 's/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a\.type()/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a.scalar_type()/g' lietorch_gpu.cu
sed -i 's/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X\.type()/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.scalar_type()/g' lietorch_gpu.cu
cd -
```

### 7d. Fix `base/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp`

Same as above, 19 occurrences:

```bash
cd base/thirdparty/lietorch/lietorch/src
sed -i 's/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a\.type()/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a.scalar_type()/g' lietorch_cpu.cpp
sed -i 's/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X\.type()/DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.scalar_type()/g' lietorch_cpu.cpp
cd -
```

### 7e. Fix `base/thirdparty/lietorch/lietorch/extras/extras.cpp`

```diff
- #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
+ #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
```

### 7f. Fix `base/thirdparty/lietorch/lietorch/extras/corr_index_kernel.cu`

```diff
- AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
+ AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_forward_kernel", ([&] {
```

```diff
- AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {
+ AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_backward_kernel", ([&] {
```

### 7g. Fix `base/thirdparty/lietorch/lietorch/include/dispatch.h`

The custom `DISPATCH_GROUP_AND_FLOATING_TYPES` macro needs to accept `at::ScalarType` directly instead of going through the deprecated `::detail::scalar_type()` helper. In the macro definition, replace:

```diff
- #define DISPATCH_GROUP_AND_FLOATING_TYPES(GROUP, TYPE, NAME, ...)        \
-   [&] {                                                                  \
-     const auto& the_type = TYPE;                                         \
-     /* don't use TYPE again in case it is an expensive or side-effect op */ \
-     at::ScalarType _st = ::detail::scalar_type(the_type);               \
+ #define DISPATCH_GROUP_AND_FLOATING_TYPES(GROUP, SCALAR_TYPE, NAME, ...) \
+   [&] {                                                                  \
+     const at::ScalarType _st = SCALAR_TYPE;                              \
```

> **Note:** Do NOT change `.device().type()` calls in `lietorch.cpp` -- those check the device type (CPU vs CUDA) and are correct.

## 8. Compile CUDA Extensions (Requires GPU Node)

The compilation **must run on a GPU node** since it needs CUDA. Submit via Slurm:

```bash
srun -p <your_partition> --gres=gpu:1 --time=00:15:00 --mem=32G bash -c '
    conda activate mega_sam
    cd /path/to/mega-sam/base
    python setup.py install
'
```

Or if conda activate doesn't work inside srun (common issue), use the full path:

```bash
srun -p <your_partition> --gres=gpu:1 --time=00:15:00 --mem=32G bash -c '
    export PYTHONNOUSERSITE=1
    export PATH=/path/to/miniconda3/envs/mega_sam/bin:$PATH
    cd /path/to/mega-sam/base
    python setup.py install
'
```

This compiles two extensions:
- `droid_backends` -- DROID-SLAM correlation and altcorr kernels
- `lietorch_backends` -- Lie group operations for SE3/SO3

The `setup.py` targets compute capabilities 70, 75, 80, 86 (covers V100, T4, A100, RTX 3090). If you have newer GPUs (e.g., H100 = sm_90), add `-gencode=arch=compute_90,code=sm_90` to the nvcc args in `base/setup.py`.

## 9. Download Pretrained Checkpoints

### DepthAnything checkpoint

```bash
mkdir -p Depth-Anything/checkpoints
wget -O Depth-Anything/checkpoints/depth_anything_vitl14.pth \
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
```

### RAFT checkpoint

The RAFT checkpoint is hosted on Google Drive. Install `gdown` and download:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT" -O /tmp/raft_checkpoints
cp /tmp/raft_checkpoints/raft-things.pth cvd_opt/raft-things.pth
```

Or download `raft-things.pth` manually from the [RAFT Google Drive folder](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and place it at `cvd_opt/raft-things.pth`.

## 10. Verify the Installation

Run this on a GPU node to confirm everything works:

```bash
srun -p <your_partition> --gres=gpu:1 --time=00:05:00 --mem=16G bash -c '
    export PYTHONNOUSERSITE=1
    export PATH=/path/to/miniconda3/envs/mega_sam/bin:$PATH
    python -c "
import torch
print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.version.cuda, \"GPU:\", torch.cuda.get_device_name(0))
import torch_scatter; print(\"torch_scatter OK\")
import xformers; print(\"xformers OK\")
import droid_backends; print(\"droid_backends OK\")
import lietorch; print(\"lietorch OK\")
import kornia; print(\"kornia OK\")
import timm; print(\"timm OK\")
import einops; print(\"einops OK\")
print(\"All imports successful!\")
"'
```

Expected output:

```
PyTorch: 2.10.0+cu126 CUDA: 12.6 GPU: NVIDIA A100 80GB PCIe
torch_scatter OK
xformers OK
droid_backends OK
lietorch OK
kornia OK
timm OK
einops OK
All imports successful!
```

> **Note:** A torchvision warning about `Failed to load image Python extension` is harmless -- MegaSaM uses OpenCV for image I/O, not torchvision.

## Quick Reference: Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `libnvshmem_host.so.3` not found | Missing NVIDIA nvshmem library | `pip install nvidia-nvshmem-cu12` |
| `torch_scatter` import error with `undefined symbol` | torch-scatter compiled for different PyTorch | Reinstall: `pip install torch-scatter --no-build-isolation` |
| CUDA compilation fails with `.type()` error | Deprecated PyTorch C++ API | Apply source patches from Section 7 |
| `ModuleNotFoundError` for packages that are installed | User site-packages (`~/.local`) shadowing conda env | Use `PYTHONNOUSERSITE=1` prefix |
| `Cannot import 'setuptools.build_meta'` | pip version conflict between user/conda | `PYTHONNOUSERSITE=1 python -m pip install ...` |
