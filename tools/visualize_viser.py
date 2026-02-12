#!/usr/bin/env python3
"""Interactive 3D visualizer for MegaSaM output using Viser.

Usage:
    python tools/visualize_viser.py outputs_cvd/hawor_chunk0_sgd_cvd_hr.npz --port 8080

No GPU needed — runs on the login node (CPU-only: numpy + viser web server).

=== How the data flows ===

MegaSaM outputs an .npz with 4 arrays:
  - images   (N, H, W, 3) uint8   — RGB frames
  - depths   (N, H, W)    float   — per-pixel depth in meters (distance along camera Z)
  - intrinsic (3, 3)      float   — pinhole camera matrix K (shared across all frames)
  - cam_c2w  (N, 4, 4)    float   — camera-to-world transform per frame

The intrinsic matrix K looks like:
    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
where (fx, fy) are focal lengths in pixels and (cx, cy) is the principal point.

=== Coordinate conventions ===

Camera frame (OpenCV convention, which viser also uses):
  +X = right, +Y = down, +Z = forward (into the scene)

For a pixel (u, v) with depth d, the 3D point in camera coords is:
  X = (u - cx) * d / fx
  Y = (v - cy) * d / fy
  Z = d

Each frame's point cloud is kept in camera-local coords and placed into the
world using the c2w rotation (wxyz quaternion) and translation — viser handles
the transform. This avoids manually multiplying R @ pts + t for every frame.

=== Frustum FOV from intrinsics ===

Viser's add_camera_frustum takes (fov, aspect) not raw intrinsics. We convert:
  fov   = 2 * arctan(H / (2 * fy))   — vertical field of view in radians
  aspect = W / H                       — width / height

=== Scene auto-scaling ===

GUI slider ranges (frustum scale, point size) are set relative to the scene
"extent" so they work for any dataset. The extent is:
  max(camera_translation_span, mean_depth)

We take the max because for handheld video the cameras may barely move (e.g.
13cm) while objects are much farther away (e.g. 75cm). If we only used the
camera span, frustums would be ~70x too small relative to the point cloud.
"""

import argparse
import time

import numpy as np
import viser


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_npz(path: str) -> dict:
    """Load MegaSaM .npz output. See module docstring for array shapes."""
    data = np.load(path)
    return dict(
        images=data["images"],                          # (N, H, W, 3) uint8
        depths=data["depths"].astype(np.float32),       # (N, H, W)
        intrinsic=data["intrinsic"].astype(np.float32), # (3, 3)
        cam_c2w=data["cam_c2w"].astype(np.float32),     # (N, 4, 4)
    )


# ─── Geometry helpers ─────────────────────────────────────────────────────────


def unproject(depth, fx, fy, cx, cy, sub):
    """Unproject a depth map into camera-local 3D points using the pinhole model.

    For each pixel (u, v) with depth d, the 3D point in the camera frame is:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth: (H, W) depth map in meters.
        fx, fy, cx, cy: pinhole intrinsic parameters from K.
        sub: subsample factor — take every `sub`-th pixel in both axes
             to reduce point count. sub=4 means ~1/16 of all pixels.

    Returns:
        pts: (M, 3) float32 — 3D points in camera-local coords.
        uv:  (M, 2) int — (row, col) pixel coordinates for color lookup.
    """
    H, W = depth.shape
    # Build a subsampled pixel grid: v=rows, u=cols
    v, u = np.mgrid[0:H:sub, 0:W:sub]
    d = depth[v, u].ravel()
    valid = (d > 0) & np.isfinite(d)
    u, v, d = u.ravel()[valid].astype(np.float32), v.ravel()[valid].astype(np.float32), d[valid]

    # Pinhole inverse projection: pixel + depth -> 3D camera coords
    pts = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=-1)
    uv = np.stack([v.astype(int), u.astype(int)], axis=-1)
    return pts, uv


def depth_colormap(vals, vmin, vmax):
    """Map depth values to RGB using the turbo colormap. Returns (N, 3) uint8."""
    import matplotlib.cm as cm
    norm = np.clip((vals - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return (cm.turbo(norm)[:, :3] * 255).astype(np.uint8)


def rot_to_wxyz(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion.

    Viser expects quaternions in wxyz order (scalar-first). We use Shepperd's
    method which is numerically stable for all rotation angles.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s], np.float32)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s], np.float32)
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s], np.float32)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s], np.float32)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Viser 3D visualizer for MegaSaM output")
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--subsample", type=int, default=4)
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading {args.npz_path}...")
    d = load_npz(args.npz_path)
    images, depths, K, c2ws = d["images"], d["depths"], d["intrinsic"], d["cam_c2w"]
    N, H, W = images.shape[:3]

    # Extract pinhole parameters from the intrinsic matrix K
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Convert intrinsics to fov/aspect for viser's frustum API.
    # fov = vertical field of view: the full angle subtended by the image height.
    #   Half the image (H/2 pixels) maps to half the fov, and tan(fov/2) = (H/2) / fy,
    #   so fov = 2 * arctan(H / (2*fy)).
    fov = float(2 * np.arctan(H / (2 * fy)))
    aspect = W / H
    print(f"  {N} frames, {H}x{W}, fov={np.degrees(fov):.1f}deg")

    # ── Scene auto-scaling ──
    # We need a reference "extent" to set sensible default slider ranges for
    # frustum scale, point size, etc.
    #
    # cam_extent = how far cameras move (diagonal of translation bounding box).
    # mean_depth = average distance from cameras to the scene content.
    #
    # We use max(cam_extent, mean_depth) because for handheld video the camera
    # motion can be very small (e.g. 13cm) while the scene is much farther away
    # (e.g. 75cm). Using only cam_extent would make frustums ~70x too small
    # relative to the point cloud.
    trans = c2ws[:, :3, 3]  # (N, 3) camera positions in world
    valid_d = depths[(depths > 0) & np.isfinite(depths)]
    mean_depth = float(valid_d.mean()) if len(valid_d) > 0 else 1.0
    cam_extent = max(np.linalg.norm(trans.max(0) - trans.min(0)), 1e-3)
    extent = max(cam_extent, mean_depth)
    print(f"  cam_extent={cam_extent:.4f}, mean_depth={mean_depth:.4f}, extent={extent:.4f}")

    # ── Up direction ──
    # In OpenCV camera convention, +Y points down. The second column of each
    # c2w matrix is the camera's Y-axis in world coords. Averaging and negating
    # gives us the scene's "up" direction, so viser orients the view correctly.
    avg_y = c2ws[:, :3, 1].mean(0)
    up = -avg_y / max(np.linalg.norm(avg_y), 1e-8)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction(tuple(up.tolist()))

    # ── GUI controls ──

    with server.gui.add_folder("Frame Range"):
        gui_start = server.gui.add_slider("Start", min=0, max=N-1, step=1, initial_value=0)
        gui_end = server.gui.add_slider("End", min=0, max=N-1, step=1, initial_value=N-1)
        gui_step = server.gui.add_slider("Frame Step", min=1, max=max(N//4, 1), step=1, initial_value=1)

    with server.gui.add_folder("Point Cloud"):
        # sub=4 means take every 4th pixel in both x and y, so ~1/16 of pixels.
        # For 328x584 images, that's ~12K points/frame.
        gui_sub = server.gui.add_slider("Pixel Subsample", min=1, max=16, step=1, initial_value=args.subsample)
        gui_pt_size = server.gui.add_slider("Point Size", min=extent*0.0005, max=extent*0.02,
                                             step=extent*0.0001, initial_value=extent*0.003)
        gui_color = server.gui.add_dropdown("Color Mode", options=["RGB", "Depth"], initial_value="RGB")

    with server.gui.add_folder("Cameras"):
        gui_show_frustums = server.gui.add_checkbox("Show Frustums", initial_value=True)
        gui_frust_scale = server.gui.add_slider("Frustum Scale", min=extent*0.01, max=extent*0.5,
                                                 step=extent*0.005, initial_value=extent*0.08)

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider("Current Frame", min=0, max=N-1, step=1, initial_value=0)
        gui_play = server.gui.add_checkbox("Play", initial_value=False)
        gui_fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)

    # ── Dirty flags ──
    # We use lists so that closures (lambdas) can mutate them.
    # need_rebuild: any slider/dropdown change triggers a full scene rebuild.
    # need_playback: only the "Current Frame" slider changed — just move the
    #   highlighted red frustum and show only that frame's point cloud.
    need_rebuild = [True]
    need_playback = [False]
    handles = []             # all scene handles (frustums + point clouds) for cleanup
    cloud_handles = {}       # frame_index -> point cloud handle, for per-frame visibility
    playback_handle = [None] # the single red "current frame" frustum

    for g in [gui_start, gui_end, gui_step, gui_sub, gui_pt_size, gui_color, gui_show_frustums, gui_frust_scale]:
        g.on_update(lambda _: need_rebuild.__setitem__(0, True))

    gui_frame.on_update(lambda _: need_playback.__setitem__(0, True))
    # Toggling play on/off should immediately show/hide per-frame clouds
    gui_play.on_update(lambda _: need_playback.__setitem__(0, True))

    def get_indices():
        """Return frame indices based on current start/end/step sliders."""
        return range(gui_start.value, min(gui_end.value + 1, N), gui_step.value)

    def rebuild():
        """Rebuild the entire scene: remove old handles, add frustums + point clouds.

        For each visible frame we add:
          1. A camera frustum at /frames/NNNN/frustum — shows the camera's position,
             orientation, and FOV as a wireframe pyramid with an image thumbnail.
          2. A point cloud at /frames/NNNN/points — the depth map unprojected into
             camera-local 3D coords.

        Both are placed in world space by setting wxyz (rotation) and position
        (translation) from the frame's c2w matrix. Viser applies this transform
        when rendering, so we don't need to manually multiply R @ pts + t.
        """
        # Clear previous scene objects
        for h in handles:
            h.remove()
        handles.clear()
        cloud_handles.clear()
        if playback_handle[0] is not None:
            playback_handle[0].remove()
            playback_handle[0] = None

        indices = get_indices()
        sub = gui_sub.value
        pt_size = gui_pt_size.value
        frust_scale = gui_frust_scale.value
        show_frustums = gui_show_frustums.value
        color_mode = gui_color.value

        # Pre-compute depth colormap range across all visible frames
        if color_mode == "Depth":
            all_d = depths[list(indices)]
            vmask = (all_d > 0) & np.isfinite(all_d)
            vmin = float(all_d[vmask].min()) if vmask.any() else 0.0
            vmax = float(all_d[vmask].max()) if vmask.any() else 1.0

        for i in indices:
            # Extract rotation R and translation t from the 4x4 c2w matrix
            R, t = c2ws[i, :3, :3], c2ws[i, :3, 3]
            wxyz = rot_to_wxyz(R)
            frame_name = f"/frames/{i:04d}"

            # --- Camera frustum ---
            if show_frustums:
                fh = server.scene.add_camera_frustum(
                    f"{frame_name}/frustum",
                    fov=fov, aspect=aspect, scale=frust_scale,
                    color=(100, 200, 100),
                    image=images[i][::4, ::4],  # 4x downsampled thumbnail
                    wxyz=wxyz, position=t,
                )

                # Click a frustum to snap the viewer camera to that frame's viewpoint
                @fh.on_click
                def _(event, idx=i):
                    c = c2ws[idx]
                    with event.client.atomic():
                        event.client.camera.wxyz = rot_to_wxyz(c[:3, :3])
                        event.client.camera.position = c[:3, 3]
                    gui_frame.value = idx

                handles.append(fh)

            # --- Per-frame point cloud ---
            # Points are in camera-local coords (Z forward). Viser transforms
            # them to world space using the wxyz/position we provide, which
            # matches this frame's c2w pose.
            pts_cam, uv = unproject(depths[i], fx, fy, cx, cy, sub)
            if len(pts_cam) > 0:
                if color_mode == "RGB":
                    colors = images[i][uv[:, 0], uv[:, 1]]
                else:
                    d_vals = depths[i][uv[:, 0], uv[:, 1]]
                    colors = depth_colormap(d_vals, vmin, vmax)

                ph = server.scene.add_point_cloud(
                    f"{frame_name}/points",
                    points=pts_cam, colors=colors,
                    point_size=pt_size, point_shape="rounded",
                    wxyz=wxyz, position=t,
                )
                handles.append(ph)
                cloud_handles[i] = ph

        # During playback, only the current frame's cloud is visible.
        # When not playing, all clouds are visible.
        print(f"  Built {len(list(indices))} frames, subsample={sub}, color={color_mode}")
        update_playback_frustum()
        update_cloud_visibility()

    def update_cloud_visibility():
        """When playing, show only the current frame's point cloud.
        When stopped, show all point clouds."""
        playing = gui_play.value
        current = gui_frame.value
        for idx, ph in cloud_handles.items():
            ph.visible = (not playing) or (idx == current)

    def update_playback_frustum():
        """Show a highlighted red frustum for the current playback frame.

        This is separate from the green per-frame frustums — it's a single
        indicator that moves with the "Current Frame" slider or auto-playback
        without rebuilding all the point clouds.
        """
        if playback_handle[0] is not None:
            playback_handle[0].remove()
            playback_handle[0] = None

        idx = gui_frame.value
        R, t = c2ws[idx, :3, :3], c2ws[idx, :3, 3]
        playback_handle[0] = server.scene.add_camera_frustum(
            "/cameras/_current",
            fov=fov, aspect=aspect,
            scale=gui_frust_scale.value * 1.3,  # slightly larger than normal frustums
            color=(255, 80, 80),                 # red to stand out
            image=images[idx][::4, ::4],
            wxyz=rot_to_wxyz(R), position=t,
        )

    # ── Main loop ──
    print(f"Server at http://{args.host}:{args.port}")
    rebuild()
    need_rebuild[0] = False
    last_tick = time.time()

    try:
        while True:
            if need_rebuild[0]:
                print("Rebuilding...")
                rebuild()
                need_rebuild[0] = False

            if need_playback[0]:
                update_playback_frustum()
                update_cloud_visibility()
                need_playback[0] = False

            # Auto-advance current frame during playback
            if gui_play.value:
                now = time.time()
                if now - last_tick >= 1.0 / gui_fps.value:
                    indices = list(get_indices())
                    if indices:
                        cur = gui_frame.value
                        nxt = [f for f in indices if f > cur]
                        gui_frame.value = nxt[0] if nxt else indices[0]  # wrap around
                        update_playback_frustum()
                        update_cloud_visibility()
                    last_tick = now

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
