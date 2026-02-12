# MegaSaM (RLWRLD Fork)

[Project Page](https://mega-sam.github.io/index.html) | [Paper](https://arxiv.org/abs/2412.04463)

RLWRLD fork of MegaSaM with patched CUDA kernels for PyTorch 2.x + CUDA 12.x,
and an added [Viser-based 3D visualizer](docs/visualizer.md).

Based on the paper: **MegaSaM: Accurate, Fast and Robust Structure and Motion
from Casual Dynamic Videos** — Li et al., CVPR 2025.

## Clone

```bash
git clone --recursive git@github.com:RLWRLD/rlwrld_mega_sam.git
```

## Setup

### Python Environment

Tested with Python 3.10, CUDA 12.4, PyTorch 2.x on A100 GPUs.

1.  Create the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate mega_sam
    ```

2.  Install xformers (for the UniDepth model):
    ```bash
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
    ```

3.  Compile the CUDA extensions for camera tracking:
    ```bash
    cd base && python setup.py install && cd ..
    ```

For Slurm-specific setup details, see [docs/slurm_setup.md](docs/slurm_setup.md).

### Downloading pretrained checkpoints

1.  Download [DepthAnything checkpoint](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) to
    mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth

2.  Download and include [RAFT checkpoint](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) at mega-sam/cvd_opt/raft-things.pth

### Running MegaSaM on Sintel

1.  Download and unzip [Sintel data](https://drive.google.com/file/d/1bSGX7JY73M3HzMS6xsJizRkPH-NQLPOf/view?usp=sharing)

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_sintel.sh`

3.  Run camera tracking (Please modify DATA_PATH in the script. Adding
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_sintel.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script): `./cvd_opt/cvd_opt_sintel.sh`

5.  Evaluate camera poses and depths: \
    `python ./evaluations_poses/evaluate_sintel.py`

    `python ./evaluations_depth/evaluate_depth_ours_sintel.py`

### Running MegaSaM on DyCheck

1.  Download [Dycheck data](https://drive.google.com/drive/folders/1BHzjHo58nGAMvKMo_AS0_SwU2tJagXXx?usp=sharing)

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_dycheck.sh`

3.  Running camera tracking (Please modify DATA_PATH in the script. Add
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_dycheck.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script):
    `./cvd_opt/cvd_opt_dycheck.sh`

5.  Evaluate camera poses and depths: \
    `python ./evaluations_poses/evaluate_dycheck.py`

    `python ./evaluations_depth/evaluate_depth_ours_dycheck.py`

### Running MegaSaM on in-the-wild video, for example from DAVIS videos

1.  Download example [DAVIS data](https://drive.google.com/file/d/1y5XItnTTgZJqRSOpG48v1FuHvPgaAvw8/view?usp=sharing)

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_demo.sh`

3.  Running camera tracking (Please modify DATA_PATH in the script. Add
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_demo.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script):
    `./cvd_opt/cvd_opt_demo.sh`

### Visualizing Results

After running the pipeline, inspect the output interactively in 3D:

```bash
pip install viser
python tools/visualize_viser.py outputs_cvd/<your_output>.npz --port 8080
```

Then open `http://<hostname>:8080` in your browser. No GPU needed.

See [docs/visualizer.md](docs/visualizer.md) for full usage details.

### Contact

For any questions related to our paper, please send email to zl548@cornell.edu.


### Bibtex

```
@inproceedings{li2025megasam,
  title     = {{MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos}},
  author    = {Li, Zhengqi and Tucker, Richard and Cole, Forrester and Wang, Qianqian and Jin, Linyi and Ye, Vickie and Kanazawa, Angjoo and Holynski, Aleksander and Snavely, Noah},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

### Copyright

Copyright 2025 Google LLC  

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

