Video anonymization by face detection
=====================================

    $ python3 deface.py -i <input.mp4> -o <output.mp4>

Dependencies (conda):

    conda install -c conda-forge imageio imageio-ffmpeg numpy tqdm scikit-image opencv-python


Credits:
- centerface.{onnx,py}: based on https://github.com/Star-Clouds/centerface (8c39a497)