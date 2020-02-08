# Deface: Video anonymization by face detection

## Installation

From PyPI (official releases):

    $ pip install deface

Current git master:

    $ pip install 'git+https://github.com/mdraw/deface'

## Usage

Basic usage after installing the `deface` package:

    $ deface -i <input.mp4> -o <output.mp4>

Show help and all options:

    $ deface -h


## Hardware acceleration with [ONNX Runtime](https://microsoft.github.io/onnxruntime/)

### CUDA (on Nvidia GPUs)

If you have a CUDA-capable GPU, you can enable GPU acceleration by installing the relevant packages:

    $ pip install onnx onnxruntime-gpu

If the `onnxruntime-gpu` package is found and a GPU is available, the face detection network is automatically offloaded to the GPU.
This can significantly improve the overall processing speed.

### Other platforms

If your machine doesn't have a CUDA-capable GPU but you want to accelerate computation on another hardware platform (e.g. Intel CPUs), you can look into the available options in the [ONNX Runtime build matrix](https://microsoft.github.io/onnxruntime/).


## Credits

- `centerface.onnx` (original) and `centerface.py` (modified) are based on https://github.com/Star-Clouds/centerface (revision [8c39a49](https://github.com/Star-Clouds/CenterFace/tree/8c39a497afb78fb2c064eb84bf010c273bb7d3ce)),
  [released under MIT license](https://github.com/Star-Clouds/CenterFace/blob/36afed/LICENSE)
