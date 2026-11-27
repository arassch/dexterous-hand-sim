"""Join rendered clips into one video, in order.

    python scripts/concat_videos.py videos/pinch_grid.mp4 videos/rigid_vs_soft.mp4 \\
        videos/bar_rigid_vs_soft.mp4 --out videos/showreel.mp4
"""
import argparse

import imageio.v2 as imageio
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=0.5, help="seconds of black between clips")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    w = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"])
    shape = None
    for k, path in enumerate(args.clips):
        for frame in imageio.get_reader(path):
            shape = frame.shape
            w.append_data(frame)
        if k < len(args.clips) - 1:
            for _ in range(int(args.gap * args.fps)):
                w.append_data(np.zeros(shape, np.uint8))
    w.close()
    print(f"wrote {args.out}")
