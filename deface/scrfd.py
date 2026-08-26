import os

from functools import lru_cache

import numpy as np
import cv2


# Find file relative to the location of this code files
default_onnx_path = f'{os.path.dirname(__file__)}/scrfd_10g.onnx'


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert input image to RGB if it is in RGBA or L format"""
    if img.ndim == 2:  # 1-channel grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # 4-channel RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points: np.ndarray of shape (n, 2), [x, y] anchor centers.
        distance: np.ndarray of shape (n, 4), [left, top, right, bottom] distances.
        max_shape: optional (h, w) tuple to clip coords to image bounds.

    Returns:
        np.ndarray of shape (n, 4), [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints in flat [x1,y1,x2,y2,...] order.

    Args:
        points: np.ndarray of shape (n, 2), [x, y] anchor centers.
        distance: np.ndarray of shape (n, 10), alternating [dx, dy] per keypoint.
        max_shape: optional (h, w) tuple to clip coords to image bounds.

    Returns:
        np.ndarray of shape (n, 10), [x1,y1,x2,y2,...,x5,y5].
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, onnx_path=None, in_shape=None, backend='auto', override_execution_provider=None):
        self.in_shape = in_shape
        self.onnx_input_name = 'input.1'
        # 9 outputs in order: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
        self.onnx_output_names = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
        self._feat_stride_fpn = [8, 16, 32]
        self._fmc = 3
        self._num_anchors = 2
        self._use_kps = True
        self._nms_thresh = 0.4
        self.center_cache = {}

        if onnx_path is None:
            onnx_path = default_onnx_path

        if backend == 'auto':
            try:
                import onnx
                import onnxruntime
                backend = 'onnxrt'
            except:
                # TODO: Warn when using a --verbose flag
                backend = 'opencv'
        self.backend = backend

        if self.backend == 'opencv':
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
        elif self.backend == 'onnxrt':
            import onnx
            import onnxruntime

            # Silence warnings about unnecessary bn initializers
            onnxruntime.set_default_logger_severity(3)

            static_model = onnx.load(onnx_path)

            # onnxruntime.get_available_providers() returns a list of all
            # available providers in a reasonable ordering (GPU providers
            # first, then accelerated CPU providers like OpenVINO, then
            # CPUExecutionProvider as the last choice). In normal conditions,
            # overriding this choice won't be necessary.
            available_providers = onnxruntime.get_available_providers()
            if override_execution_provider is None:
                ort_providers = available_providers
            else:
                if override_execution_provider not in available_providers:
                    raise ValueError(f'{override_execution_provider=} not found. Available providers are: {available_providers}')
                ort_providers = [override_execution_provider]

            self.sess = onnxruntime.InferenceSession(static_model.SerializeToString(), providers=ort_providers)

            preferred_provider = self.sess.get_providers()[0]
            print(f'Running on {preferred_provider}.')

    def __call__(self, img, threshold=0.5):
        img = ensure_rgb(img)
        orig_shape = img.shape[:2]
        in_shape = orig_shape[::-1] if self.in_shape is None else self.in_shape

        w_new, h_new, det_scale = self.shape_transform(in_shape, orig_shape)

        # Resize keeping aspect ratio, then pad to in_shape
        resized_img = cv2.resize(img, (w_new, h_new))
        det_img = np.zeros((in_shape[1], in_shape[0], 3), dtype=np.uint8)
        det_img[:h_new, :w_new, :] = resized_img

        # SCRFD preprocessing: (img - 127.5) / 128. Input is RGB, so swapRB=False.
        blob = cv2.dnn.blobFromImage(
            det_img, scalefactor=1.0/128, size=(in_shape[0], in_shape[1]),
            mean=(127.5, 127.5, 127.5), swapRB=False, crop=False
        )

        if self.backend == 'opencv':
            self.net.setInput(blob)
            outs = self.net.forward(self.onnx_output_names)
        elif self.backend == 'onnxrt':
            outs = self.sess.run(self.onnx_output_names, {self.onnx_input_name: blob})
        else:
            raise RuntimeError(f'Unknown backend {self.backend}')

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx]
            bbox_preds = outs[idx + self._fmc] * stride
            kps_preds = outs[idx + self._fmc * 2] * stride

            height = -(-input_height // stride)
            width = -(-input_width // stride)
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Build anchor centers at (x, y) for each FPN cell, replicated for num_anchors
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self._use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        # No detections at any stride: return CenterFace-shaped empty arrays
        if sum(s.shape[0] for s in scores_list) == 0:
            return np.empty(shape=[0, 5], dtype=np.float32), np.empty(shape=[0, 10], dtype=np.float32)

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        # Sort by score, NMS, reorder landmarks to match
        order = scores.ravel().argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Flatten kpss from (N, 5, 2) to (N, 10) to match CenterFace's lms format
        lms = kpss.reshape((kpss.shape[0], -1)).astype(np.float32)

        return det, lms

    @staticmethod
    @lru_cache(maxsize=128)
    def shape_transform(in_shape, orig_shape):
        # in_shape: (W, H); orig_shape: (H, W)
        w_target, h_target = in_shape
        h_orig, w_orig = orig_shape
        im_ratio = h_orig / w_orig
        model_ratio = h_target / w_target
        if im_ratio > model_ratio:
            new_height = h_target
            new_width = int(new_height / im_ratio)
        else:
            new_width = w_target
            new_height = int(new_width * im_ratio)
        det_scale = new_height / h_orig
        return new_width, new_height, det_scale

    def nms(self, dets):
        thresh = self._nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
