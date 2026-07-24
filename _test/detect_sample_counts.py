import argparse
from pathlib import Path

import imageio.v3 as iio

from deface.centerface import CenterFace


def parse_scale(value):
    if value == "full":
        return value, None
    width, height = value.lower().split("x", maxsplit=1)
    return value, (int(width), int(height))


def main():
    parser = argparse.ArgumentParser(description="Count deface detections across sampled image frames.")
    parser.add_argument("frame_dir", type=Path, help="Directory containing sampled image frames.")
    parser.add_argument("--pattern", default="*.jpg", help="Frame glob pattern. Default: *.jpg.")
    parser.add_argument("--threshold", default=[0.2, 0.1, 0.05], nargs="+", type=float)
    parser.add_argument("--scale", default=["1280x720", "full"], nargs="+")
    parser.add_argument("--backend", default="opencv", choices=["auto", "onnxrt", "opencv"])
    args = parser.parse_args()

    frames = sorted(args.frame_dir.glob(args.pattern))
    if not frames:
        raise SystemExit(f"No frames found in {args.frame_dir} matching {args.pattern}")

    for scale in args.scale:
        scale_name, in_shape = parse_scale(scale)
        detector = CenterFace(in_shape=in_shape, backend=args.backend)
        print(f"scale={scale_name}")
        for threshold in args.threshold:
            print(f"  threshold={threshold}")
            for frame_path in frames:
                image = iio.imread(frame_path)
                dets, _ = detector(image, threshold=threshold)
                scores = ", ".join(f"{score:.3f}" for score in dets[:, 4]) if len(dets) else "-"
                print(f"    {frame_path.name}: {len(dets)} detections; scores={scores}")


if __name__ == "__main__":
    main()
