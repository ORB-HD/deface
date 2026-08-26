"""Gradio v5 web UI for the deface face anonymization tool."""

import os
import tempfile

import imageio

try:
    import gradio as gr
except ImportError:
    raise SystemExit("Gradio is not installed. Install it with: pip install 'deface[ui]'")

from deface.centerface import CenterFace
from deface.scrfd import SCRFD

_detectors: dict = {}


def get_detector(detector_name, in_shape=None, backend='auto', execution_provider=None):
    key = (detector_name, in_shape, backend, execution_provider)
    detector = _detectors.get(key)
    if detector is None:
        cls = SCRFD if detector_name == 'scrfd' else CenterFace
        detector = cls(
            in_shape=in_shape,
            backend=backend,
            override_execution_provider=execution_provider,
        )
        _detectors[key] = detector
    return detector


def _load_replaceimg(replacewith, replaceimg_file):
    if replacewith == 'img' and replaceimg_file is not None:
        return imageio.imread(replaceimg_file)
    return None


def process_image(
    input_img, detector_name, replacewith, threshold, mask_scale,
    use_ellipse, draw_scores, mosaicsize, replaceimg_file,
):
    frame = input_img.copy()
    detector = get_detector(detector_name.lower())
    dets, _ = detector(frame, threshold=threshold)
    replaceimg_np = _load_replaceimg(replacewith, replaceimg_file)
    from deface.deface import anonymize_frame
    anonymize_frame(
        dets, frame, mask_scale=mask_scale, replacewith=replacewith,
        ellipse=use_ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg_np, mosaicsize=mosaicsize,
    )
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        opath = tmp.name
    imageio.imwrite(opath, frame)
    return input_img, frame, opath


def process_video(
    input_video, detector_name, replacewith, threshold, mask_scale,
    use_ellipse, draw_scores, keep_audio, mosaicsize, replaceimg_file,
):
    _, ext = os.path.splitext(input_video)
    with tempfile.NamedTemporaryFile(suffix=ext or '.mp4', delete=False) as tmp:
        opath = tmp.name
    detector = get_detector(detector_name.lower())
    replaceimg_np = _load_replaceimg(replacewith, replaceimg_file)
    from deface.deface import video_detect
    video_detect(
        ipath=input_video, opath=opath, detector=detector,
        threshold=threshold, enable_preview=False, cam=False, nested=False,
        replacewith=replacewith, mask_scale=mask_scale,
        ellipse=use_ellipse, draw_scores=draw_scores,
        ffmpeg_config={'codec': 'libx264'},
        replaceimg=replaceimg_np, keep_audio=keep_audio,
        mosaicsize=mosaicsize, disable_progress_output=True,
    )
    return opath, f'Done! Output saved to {opath}'


def update_visibility(mode):
    return gr.Group(visible=(mode == 'mosaic')), gr.Group(visible=(mode == 'img'))


with gr.Blocks(title='deface - Face Anonymization', theme=gr.themes.Soft()) as app:
    gr.Markdown('# 🎭 deface — Face Anonymization Tool')
    gr.Markdown('Upload an image or video to anonymize detected faces.')

    with gr.Tab('🖼️ Image'):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type='numpy', label='Upload Image')
                img_detector = gr.Radio(['SCRFD', 'CenterFace'], value='SCRFD', label='Face Detector')
                img_replacewith = gr.Dropdown(
                    ['blur', 'solid', 'mosaic', 'none', 'img'], value='blur', label='Anonymization Mode',
                )
                img_threshold = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label='Detection Threshold')
                img_mask_scale = gr.Slider(1.0, 2.0, value=1.3, step=0.1, label='Mask Scale')
                img_use_ellipse = gr.Checkbox(value=True, label='Use Ellipse Mask')
                img_draw_scores = gr.Checkbox(value=False, label='Draw Detection Scores')
                img_mosaicsize_group = gr.Group(visible=False)
                with img_mosaicsize_group:
                    img_mosaicsize = gr.Slider(5, 50, value=20, step=1, label='Mosaic Size')
                img_replaceimg_group = gr.Group(visible=False)
                with img_replaceimg_group:
                    img_replaceimg = gr.File(label='Replacement Image', file_types=['image'])
                img_btn = gr.Button('Anonymize Image', variant='primary')
            with gr.Column():
                img_orig_out = gr.Image(label='Original')
                img_anon_out = gr.Image(label='Anonymized')
                img_download = gr.File(label='Download Anonymized Image')

        img_btn.click(
            fn=process_image,
            inputs=[
                img_input, img_detector, img_replacewith, img_threshold, img_mask_scale,
                img_use_ellipse, img_draw_scores, img_mosaicsize, img_replaceimg,
            ],
            outputs=[img_orig_out, img_anon_out, img_download],
        )
        img_replacewith.change(
            fn=update_visibility,
            inputs=img_replacewith,
            outputs=[img_mosaicsize_group, img_replaceimg_group],
        )

    with gr.Tab('🎬 Video'):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label='Upload Video')
                vid_detector = gr.Radio(['SCRFD', 'CenterFace'], value='SCRFD', label='Face Detector')
                vid_replacewith = gr.Dropdown(
                    ['blur', 'solid', 'mosaic', 'none', 'img'], value='blur', label='Anonymization Mode',
                )
                vid_threshold = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label='Detection Threshold')
                vid_mask_scale = gr.Slider(1.0, 2.0, value=1.3, step=0.1, label='Mask Scale')
                vid_use_ellipse = gr.Checkbox(value=True, label='Use Ellipse Mask')
                vid_draw_scores = gr.Checkbox(value=False, label='Draw Detection Scores')
                vid_keep_audio = gr.Checkbox(value=False, label='Keep Audio')
                vid_mosaicsize_group = gr.Group(visible=False)
                with vid_mosaicsize_group:
                    vid_mosaicsize = gr.Slider(5, 50, value=20, step=1, label='Mosaic Size')
                vid_replaceimg_group = gr.Group(visible=False)
                with vid_replaceimg_group:
                    vid_replaceimg = gr.File(label='Replacement Image', file_types=['image'])
                vid_btn = gr.Button('Anonymize Video', variant='primary')
            with gr.Column():
                vid_out = gr.Video(label='Anonymized Video')
                vid_status = gr.Textbox(label='Status')

        vid_btn.click(
            fn=process_video,
            inputs=[
                vid_input, vid_detector, vid_replacewith, vid_threshold, vid_mask_scale,
                vid_use_ellipse, vid_draw_scores, vid_keep_audio, vid_mosaicsize, vid_replaceimg,
            ],
            outputs=[vid_out, vid_status],
        )
        vid_replacewith.change(
            fn=update_visibility,
            inputs=vid_replacewith,
            outputs=[vid_mosaicsize_group, vid_replaceimg_group],
        )


def main():
    app.launch(server_name='0.0.0.0', server_port=7860)


if __name__ == '__main__':
    main()
