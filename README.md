Deface: Video anonymization by face detection
=====================================

Installation
------------

Usage
-----

After installing the `deface` package:

    $ deface -i <input.mp4> -o <output.mp4>

Without installation:

    $ python3 -m deface.deface -i <input.mp4> -o <output.mp4>


Dependencies
------------

Via pip:

    $ pip install imageio imageio-ffmpeg numpy tqdm scikit-image opencv-python

Or if you prefer conda:

    $ conda install -c conda-forge imageio imageio-ffmpeg numpy tqdm scikit-image py-opencv

GPU acceleration
----------------

If you have a CUDA-capable GPU, you can enable automatic GPU acceleration by installing ONNX Runtime:

    $ pip install onnx onnxruntime-gpu

If it can be loaded sucessfully, it is automatically used for face detection.
s

Credits
-------

- centerface.{onnx,py}: based on https://github.com/Star-Clouds/centerface (8c39a497),
  MIT license (https://github.com/Star-Clouds/CenterFace/blob/36afed/LICENSE)