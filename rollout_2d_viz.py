# rollout_2d_viz.py
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import imageio
except Exception:
    imageio = None

from envs.planar_nav import PlanarNavEnv
from cac.algorithms import CACAgent

def fig_to_rgb_array(fig):
    """Return (H,W,3) uint8 across common backends (TkAgg/Qt5Agg/Agg)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # 1) Agg / many backends
    try:
        s = fig.canvas.tostring_rgb()
        return np.frombuffer(s, dtype=np.uint8).reshape(h, w, 3)
    except Exception:
        pass
    # 2) TkAgg ARGB path
    try:
        s = fig.canvas.tostring_argb()
        argb = np.frombuffer(s, dtype=np.uint8).reshape(h, w, 4)
        return argb[:, :, 1:4]  # ARGB -> RGB
    except Exception:
        pass
    # 3) Renderer fallback RGBA -> RGB
    try:
        renderer = fig.canvas.get_renderer()
        rgba = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return rgba[:, :, :3]
    except Exception:
        pass
    # 4) buffer_rgba direct
    try:
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return rgba[:, :, :3]
    except Exception:
        pass
    raise RuntimeError("Could not capture figure pixels; try MPLBACKEND=Agg if recording only.")

def draw_world(ax, env, artist_cache, draw_border=True):
    L = env.L
    ax.clear()
    ax.set_aspect('equal', 'box')

    # World border (visible frame)
    if draw_border:
        ax.add_patch(patches.Rectangle((-L*0.5, -L*0.5), L, L, fill=False, linewidth=1.5))

    # Obstacles
    for cx, cy, h in env.obstacles:
        ax.add_patch(patches.Rectangle((cx-h, cy-h), 2*h, 2*h, alpha=0.30))

    # Goal
    goal = ax.plot(env.goal[0], env.goal[1], marker='*', markersize=12)[0]

    # Robot + heading
    robot = patches.Circle((env.p[0], env.p[1]), radius=env.R, alpha=0.8)
    ax.add_patch(robot)
    head = ax.plot([env.p[0], env.p[0] + np.cos(env.theta)*env.R*2],
                   [env.p[1], env.p[1] + np.sin(env.theta)*env.R*2])[0]

    # Rays
    ranges = env._rangefinder(env.p, env.theta)
    angles = env.theta + np.linspace(-env.fov/2, env.fov/2, env.n_rays)
    lines = []
    for r, ang in zip(ranges, angles):
        end = env.p + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        ln, = ax.plot([env.p[0], end[0]], [env.p[1], end[1]], linewidth=0.8, alpha=0.5)
        lines.append(ln)

    # Path
    path, = ax.plot([env.p[0]], [env.p[1]])

    artist_cache.update(dict(robot=robot, head=head, lines=lines, path=path, goal=goal))

def update_world(ax, env, artist_cache, path_pts):
    robot = artist_cache['robot']
    head  = artist_cache['head']
    lines = artist_cache['lines']
    path  = artist_cache['path']

    robot.center = (env.p[0], env.p[1])
    head.set_data([env.p[0], env.p[0] + np.cos(env.theta)*env.R*2],
                  [env.p[1], env.p[1] + np.sin(env.theta)*env.R*2])

    ranges = env._rangefinder(env.p, env.theta)
    angles = env.theta + np.linspace(-env.fov/2, env.fov/2, env.n_rays)
    for ln, r, ang in zip(lines, ranges, angles):
        end = env.p + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        ln.set_data([env.p[0], end[0]], [env.p[1], end[1]])

    path.set_data(path_pts[:,0], path_pts[:,1])

def follow_robot(ax, env, half_window):
    """Center a fixed-size window (2*half_window) around the robot."""
    x, y = env.p
    ax.set_xlim(x - half_window, x + half_window)
    ax.set_ylim(y - half_window, y + half_window)

def smart_pan(ax, env, margin=1.0):
    """Pan only when the robot nears the edge of the current view."""
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    x, y = env.p
    w = xmax - xmin; h = ymax - ymin
    moved = False
    if x < xmin + margin: xmin, xmax = x - margin, x - margin + w; moved = True
    if x > xmax - margin: xmin, xmax = x + margin - w, x + margin; moved = True
    if y < ymin + margin: ymin, ymax = y - margin, y - margin + h; moved = True
    if y > ymax - margin: ymin, ymax = y + margin - h, y + margin; moved = True
    if moved:
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

def run_episode(env, agent, deterministic=True, seed=None, save_writer=None,
                follow=False, view_half=None, smart=False):
    artist_cache = {}
    fig, ax = plt.subplots(figsize=(6,6))

    # Reset BEFORE first draw
    o, _ = env.reset(seed=seed)

    # Initial axes: whole world (static)
    L = env.L
    ax.set_xlim(-L*0.5, L*0.5)
    ax.set_ylim(-L*0.5, L*0.5)

    draw_world(ax, env, artist_cache, draw_border=True)
    plt.tight_layout()
    fig.canvas.draw(); plt.pause(0.001)

    done = False
    ep_ret = 0.0
    pts = [env.p.copy()]
    while not done:
        a = agent.act(o, deterministic=deterministic)
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc
        ep_ret += r
        pts.append(env.p.copy())

        path_pts = np.stack(pts, axis=0)
        update_world(ax, env, artist_cache, path_pts)

        # Keep the robot in view:
        if follow and view_half is not None:
            follow_robot(ax, env, view_half)
        elif smart:
            smart_pan(ax, env, margin=max(0.5, env.R*2))

        fig.canvas.draw(); plt.pause(0.001)

        if save_writer is not None:
            frame = fig_to_rgb_array(fig)
            save_writer.append_data(frame)

    return ep_ret, len(pts)-1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to saved model .pt')
    ap.add_argument('--episodes', type=int, default=1, help='How many episodes to play/record')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--stage', choices=['goal','safety'], default='goal', help='Which reward to report')
    ap.add_argument('--stochastic', action='store_true', help='Use sampling instead of deterministic mean action')
    ap.add_argument('--save', type=str, default=None, help='Optional path to .mp4 or .gif to record')
    ap.add_argument('--fps', type=int, default=30)

    # New: camera options
    ap.add_argument('--follow', action='store_true', help='Center a fixed-size view on the robot')
    ap.add_argument('--view', type=float, default=None, help='Half window size for --follow (e.g., 6.0)')
    ap.add_argument('--smart_pan', action='store_true', help='Pan only when robot nears the edge')

    args = ap.parse_args()

    # load agent + env
    agent, _ = CACAgent.load(args.ckpt)
    
    # Try to detect if this is our enhanced model by checking observation dimension
    # Our balanced model expects 26-dim observations, original expects 23-dim
    try:
        # Create test environment with enhanced observations
        test_env = PlanarNavEnv(seed=args.seed, safety_margin=0.25, alpha=0.93, 
                               enhanced_obs=True, n_rays=19, fov_deg=130)
        test_obs, _ = test_env.reset()
        agent.act(test_obs)  # This will fail if dimensions don't match
        
        # If we get here, use enhanced environment
        env = PlanarNavEnv(seed=args.seed, safety_margin=0.25, alpha=0.93, 
                          enhanced_obs=True, n_rays=19, fov_deg=130)
        print(f"✅ Using enhanced environment (obs_dim={test_obs.shape[0]})")
        
    except Exception:
        # Fall back to original environment
        env = PlanarNavEnv(seed=args.seed, enhanced_obs=False)
        print(f"📝 Using original environment (obs_dim={env.observation_space.shape[0]})")
    
    env.set_stage(args.stage)

    det = not args.stochastic

    # Prepare writer (if recording)
    writer = None
    if args.save is not None:
        if imageio is None:
            print("imageio not found; install imageio & imageio-ffmpeg to record video.", file=sys.stderr)
            sys.exit(2)
        try:
            writer = imageio.get_writer(args.save, fps=args.fps)
        except Exception as e:
            print(f"Failed to open video writer for '{args.save}': {e}", file=sys.stderr)
            sys.exit(2)

    # Auto default view for follow mode
    view_half = args.view
    if args.follow and view_half is None:
        # show most of the world but allow panning; tweak if you like
        view_half = max(env.L * 0.35, 5.0)

    try:
        R = []
        for ep in range(args.episodes):
            ret, length = run_episode(
                env, agent, deterministic=det, seed=args.seed+ep, save_writer=writer,
                follow=args.follow, view_half=view_half, smart=args.smart_pan
            )
            R.append(ret)
            print(f"Episode {ep:03d} | Return={ret:.3f}, Length={length}")
        if args.episodes > 1:
            print(f"\nAvg Return over {args.episodes} eps: {np.mean(R):.3f}")
    finally:
        if writer is not None:
            writer.close()

if __name__ == '__main__':
    main()
