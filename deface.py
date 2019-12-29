#!/usr/bin/env python3

import argparse
import tqdm
import json
import skimage.draw
import numpy as np

import cv2
from centerface import CenterFace


parser = argparse.ArgumentParser()
parser.add_argument('-i', default='0', help='Input file name or camera index')
parser.add_argument('-o', default='/tmp/deface-output.mkv', help='Output file name.')
parser.add_argument('-r', default='blur', choices=['solid', 'blur', 'none'], help='Anonymization filter mode for face regions')
# parser.add_argument('-c', default='red', help='Color hue of the overlays (boxes, texts)')
parser.add_argument('-l', default=False, action='store_true', help='Enable landmark visualization')
parser.add_argument('-q', default=False, action='store_true', help='Disable GUI')
parser.add_argument('-n', default='./centerface.onnx', help='Path to CenterFace ONNX model file')
parser.add_argument('-e', default=False, action='store_true', help='Disable detection enumeration')
parser.add_argument('-t', default=0.3, type=float, help='Detection threshold')
parser.add_argument('-m', default=False, action='store_true', help='Use ellipse masks instead of boxes')
parser.add_argument('-s', default=1.3, type=float, help='Scale factor for face masks (use high values to be on the safe side)')

args = parser.parse_args()

ipath = args.i
ipath = int(ipath) if ipath.isdigit() else ipath
opath = args.o
replacewith = args.r
draw_lms = args.l
show = not args.q
enumerate_dets = not args.e
threshold = args.t
ellipse = args.m
mask_scale = args.s
# ovcolor = colors.get(args.c, (0, 0, 0))

cam = isinstance(ipath, int)

ovcolor = (255, 0, 0)

if not opath.endswith('.mkv'):
    raise RuntimeError('Output path needs to end with .mkv due to OpenCV limitations.')


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def video_detect():
    cap = cv2.VideoCapture(ipath)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) if not cam else None
    bar = tqdm.tqdm(dynamic_ncols=True, total=nframes)
    if opath is not None:
        out = cv2.VideoWriter(opath,cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width,frame_height))
    ret, frame = cap.read()
    # h, w = frame.shape[:2]
    centerface = CenterFace()
    if not cam:
        jso = {'frames': {}}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets, lms = centerface(frame, threshold=threshold)
        if not cam:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            jso['frames'][frame_idx] = {'faces': {}}
        for i, det in enumerate(dets):
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)

            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)

            # Clip bb coordinates to valid frame region
            y1, y2 = max(0, y1), min(frame_height - 1, y2)
            x1, x2 = max(0, x1), min(frame_width - 1, x2)

            if replacewith == 'solid':
                cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
            elif replacewith == 'blur':
                bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
                blurred_box =  cv2.blur(
                    frame[y1:y2, x1:x2],
                    (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
                )
                if ellipse:
                    roibox = frame[y1:y2, x1:x2]
                    # Get y and x coordinate lists of the "bounding ellipse"
                    ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
                    roibox[ey, ex] = blurred_box[ey, ex]
                    frame[y1:y2, x1:x2] = roibox
                else:
                    frame[y1:y2, x1:x2] = blurred_box
            # elif mode == 'none':
                # pass
            if not cam:
                jso['frames'][frame_idx]['faces'][i] = {
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'score': float(round(score, 2))
                }
            if enumerate_dets:
                cv2.putText(
                    frame, f'{i + 1}: {score:.2f}', (x1 +0, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 128)
                )

        if draw_lms:
            for lm in lms:
                cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
        # hm = cv2.cvtColor(cv2.resize(hm, dsize=(frame_width, frame_height)), cv2.COLOR_GRAY2RGB)
        if opath is not None:
            out.write(frame)

        if show:
            cv2.imshow('out', frame)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.update()
    if not cam and opath is not None:
        with open(f'{opath}.json', 'w') as f:
            json.dump(jso, f)
    cap.release()
    out.release()
    bar.close()


if __name__ == '__main__':
    video_detect()
    # test_image()
