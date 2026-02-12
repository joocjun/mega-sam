# 3D Visualizer for MegaSaM Output

Interactive web-based 3D viewer for inspecting MegaSaM results (point clouds + camera poses).
Uses [Viser](https://viser.studio/) — runs as a web server, so it works on headless Slurm nodes.

## Install

Inside the `mega_sam` conda environment:

```bash
conda activate mega_sam
pip install viser
```

That's the only extra dependency. Everything else (numpy, matplotlib) is already in the env.

## Running the Visualizer

### On the login node (recommended)

The visualizer is CPU-only (no GPU needed) — just numpy math + a lightweight web server.
You can run it directly on the login node:

```bash
conda activate mega_sam
python tools/visualize_viser.py outputs_cvd/hawor_chunk0_sgd_cvd_hr.npz --port 8080
```

Then open `http://<login-node-hostname>:8080` in your browser.

### On a Slurm compute node (if login node ports are blocked)

If the login node doesn't allow inbound connections on your chosen port,
run on a compute node and use SSH port forwarding:

```bash
# 1. Start on a compute node
srun --partition=rlwrld --gres=gpu:1 --time=01:00:00 \
    python tools/visualize_viser.py outputs_cvd/hawor_chunk0_sgd_cvd_hr.npz --port 8080

# 2. From your local machine, forward the port (replace <compute-node> with the
#    node name printed by srun, e.g. gpu01):
ssh -L 8080:<compute-node>:8080 sejune@<login-node>
```

Then open `http://localhost:8080` in your browser.

> Note: `--gres=gpu:1` is required because this cluster only has GPU nodes,
> even though the visualizer itself doesn't use the GPU.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `npz_path` | (required) | Path to MegaSaM `.npz` output file |
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` = accessible from other machines) |
| `--port` | `8080` | Web server port |
| `--subsample` | `4` | Pixel subsample factor (4 = every 4th pixel in x and y = ~1/16 of all pixels) |

## GUI Controls

Once the viewer opens in your browser:

| Folder | Control | What it does |
|--------|---------|-------------|
| **Frame Range** | Start / End / Frame Step | Which frames to show. Step=2 means every other frame. |
| **Point Cloud** | Pixel Subsample | Density of points. Lower = more points = slower. |
| | Point Size | Visual size of each point. |
| | Color Mode | RGB (original image colors) or Depth (turbo colormap). |
| **Cameras** | Show Frustums | Toggle camera wireframe pyramids on/off. |
| | Frustum Scale | Visual size of the frustum pyramids. |
| **Playback** | Current Frame | Scrub through frames. Shows a red highlighted frustum. |
| | Play | Auto-advance through frames. Only current frame's points are shown. |
| | FPS | Playback speed. |

Click any green frustum to snap the viewer camera to that frame's viewpoint.

## Expected `.npz` Format

The visualizer expects MegaSaM's output format:

| Key | Shape | Description |
|-----|-------|-------------|
| `images` | `(N, H, W, 3)` uint8 | RGB frames |
| `depths` | `(N, H, W)` float | Per-pixel depth in meters (distance along camera Z-axis) |
| `intrinsic` | `(3, 3)` float | Pinhole camera matrix K (shared across all frames) |
| `cam_c2w` | `(N, 4, 4)` float | Camera-to-world 4x4 transform per frame |

The intrinsic matrix K encodes focal lengths and principal point:
```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

These are used to:
1. Compute frustum FOV: `fov = 2 * arctan(H / (2 * fy))`
2. Unproject depth to 3D: `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`

## Troubleshooting

**Frustums look too small / too large relative to point cloud:**
Adjust the "Frustum Scale" slider. The default is auto-computed from scene size.

**Too slow / browser lagging:**
Increase "Pixel Subsample" (fewer points) and/or increase "Frame Step" (fewer frames).

**Port already in use:**
Use a different port: `--port 8081`

**Can't connect from browser:**
Make sure the port is not blocked by a firewall. Try SSH port forwarding (see above).
