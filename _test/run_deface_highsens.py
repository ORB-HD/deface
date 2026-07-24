import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run deface with high-sensitivity partial-face settings.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Output path. Defaults to <input>_deface_highsens.mp4.")
    parser.add_argument("--deface-bin", default=os.environ.get("DEFACE_BIN", "deface"))
    parser.add_argument("--threshold", default="0.05")
    parser.add_argument("--mask-scale", default="1.6")
    args, passthrough = parser.parse_known_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_deface_highsens{args.input.suffix}")
    subprocess.run(
        [
            args.deface_bin,
            str(args.input),
            "--output",
            str(output),
            "--thresh",
            args.threshold,
            "--mask-scale",
            args.mask_scale,
            "--ffmpeg-config",
            '{"codec":"libx264","macro_block_size":1}',
            *passthrough,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
