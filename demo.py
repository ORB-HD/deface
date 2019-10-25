import argparse
import tqdm

import cv2
from centerface import CenterFace


parser = argparse.ArgumentParser()
parser.add_argument('-i', default='0', help='Input file name or camera index')
parser.add_argument('-o', default='/tmp/deface-output.mp4', help='Output file name.')
parser.add_argument('-r', default='box', choices=['box', 'blur', 'none'], help='How to change face regions')
parser.add_argument('-l', default=False, action='store_true', help='Enable landmark visualization')
parser.add_argument('-q', default=False, action='store_true', help='Disable GUI')
parser.add_argument('-n', default='./centerface.onnx', help='Path to CenterFace ONNX model file')


args = parser.parse_args()

ipath = args.i
ipath = int(ipath) if ipath.isdigit() else ipath
opath = args.o
replacewith = args.r
draw_lms = args.l
show = not args.q


def video_detect():
    cap = cv2.VideoCapture(ipath)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not isinstance(ipath, int):
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        bar = tqdm.tqdm(total=nframes)
    else:
        bar = tqdm.tqdm()
    if opath is not None:
        out = cv2.VideoWriter(opath,cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width,frame_height))
    ret, frame = cap.read()
    # h, w = frame.shape[:2]
    centerface = CenterFace()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets, lms = centerface(frame, threshold=0.5)
        for det in dets:
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            if replacewith == 'box':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            elif replacewith == 'blur':
                frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (20, 20))
            # elif mode == 'none':
                # pass
        if draw_lms:
            for lm in lms:
                cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
                cv2.circle(frame, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
        
        if opath is not None:
            out.write(frame)
        if show:
            cv2.imshow('out', frame)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.update()
        
    cap.release()
    out.release()
    bar.close()


if __name__ == '__main__':
    video_detect()
    # test_image()
