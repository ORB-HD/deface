import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='deface',
    version='0.1.0',
    author='Martin Drawitsch',
    author_email='martin.drawitsch@gmail.com',
    description='Video anonymization by face detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/mdraw/deface',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': [
        'deface = deface.deface:main',
    ]},
    package_data={'deface': ['centerface.onnx']},
    include_package_data=True,
    install_requires=[
        'imageio',
        'imageio-ffmpeg',
        'numpy',
        'tqdm',
        'scikit-image',
        'opencv-python',
    ],
    extras_require={
        'gpu':  ['onnxruntime-gpu'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
