"""Turn recorded viewer frames into media/replay.mp4 (full quality) and media/replay.gif (for the README).

  python scripts/encode_video.py /tmp/frames
"""

import glob
import os
import sys

import imageio.v2 as imageio
from PIL import Image

frames_dir = sys.argv[1] if len(sys.argv) > 1 else "frames"
files = sorted(glob.glob(os.path.join(frames_dir, "f*.jpg")))
os.makedirs("media", exist_ok=True)

# MP4: 1280x720, 30 fps, H.264 (CRF 27 keeps it small despite the rain)
with imageio.get_writer("media/replay.mp4", fps=30, codec="libx264", macro_block_size=16,
                        ffmpeg_params=["-crf", "27", "-preset", "slow", "-vf", "scale=1280:-2", "-movflags", "+faststart"]) as w:
    for f in files:
        w.append_data(imageio.imread(f))

# GIF: smaller and slower frame rate so GitHub shows it inline
W, STEP = 600, 4  # width in px, keep every 4th frame (7.5 fps)
gif_frames = []
for f in files[::STEP]:
    im = Image.open(f).convert("RGB")
    im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    gif_frames.append(im.quantize(colors=80, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
gif_frames[0].save("media/replay.gif", save_all=True, append_images=gif_frames[1:],
                   duration=int(1000 * STEP / 30), loop=0, optimize=True, disposal=1)

# a still for the top of the README
Image.open(files[len(files) // 6]).convert("RGB").save("media/replay_still.jpg", quality=90)

for f in ("media/replay.mp4", "media/replay.gif", "media/replay_still.jpg"):
    print(f, os.path.getsize(f) // 1024, "KB")
