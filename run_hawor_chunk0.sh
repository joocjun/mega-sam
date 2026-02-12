#!/bin/bash
set -e

export PYTHONNOUSERSITE=1
export PATH=/rlwrld1/home/sejune/miniconda3/envs/mega_sam/bin:$PATH
cd /rlwrld1/home/sejune/mega-sam

SCENE=hawor_chunk0
DATA_DIR=/rlwrld1/home/sejune/mega-sam/video_data
CKPT_PATH=checkpoints/megasam_final.pth

echo "=========================================="
echo "MegaSaM Pipeline - Scene: $SCENE"
echo "=========================================="

# ---- Stage 1a: DepthAnything (already completed) ----
echo ""
echo "[Stage 1a] DepthAnything already completed, skipping."

# ---- Stage 1b: UniDepth ----
echo ""
echo "[Stage 1b] Running UniDepth..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"
CUDA_VISIBLE_DEVICES=0 python adapters/run_unidepth.py \
  --scene-name $SCENE \
  --img-path $DATA_DIR/$SCENE \
  --outdir UniDepth/outputs
echo "[Stage 1b] UniDepth done."

# ---- Stage 2: Camera Tracking ----
echo ""
echo "[Stage 2] Running camera tracking..."
CUDA_VISIBLE_DEVICES=0 python camera_tracking_scripts/test_demo.py \
  --datapath=$DATA_DIR/$SCENE \
  --weights=$CKPT_PATH \
  --scene_name $SCENE \
  --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
  --metric_depth_path $(pwd)/UniDepth/outputs \
  --disable_vis
echo "[Stage 2] Camera tracking done."

# ---- Stage 3: Optical Flow (RAFT) ----
echo ""
echo "[Stage 3] Running RAFT optical flow..."
CUDA_VISIBLE_DEVICES=0 python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_DIR/$SCENE \
  --model=cvd_opt/raft-things.pth \
  --scene_name $SCENE \
  --mixed_precision
echo "[Stage 3] Optical flow done."

# ---- Stage 4: Depth Optimization (CVD-Opt) ----
echo ""
echo "[Stage 4] Running CVD depth optimization..."
CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
  --scene_name $SCENE \
  --w_grad 2.0 --w_normal 5.0
echo "[Stage 4] Depth optimization done."

echo ""
echo "=========================================="
echo "Pipeline complete! Output at: outputs_cvd/${SCENE}_sgd_cvd_hr.npz"
echo "=========================================="
