from setuptools import setup, find_packages

setup(
    name='VoiceExperiments',
    # py_modules=['VoiceExperiments'],
    packages=find_packages(
        include=["VoiceExperiments*"]
    ),
    version='0.0.1',
    description='A set of data processing, model training, and inference tools for voice experiments',
    author='Arsh Zahed',
    instal_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'librosa',
        'pyyaml',
        'easydict',
        'torchvision',
        'soundfile',
        'pytorch-lightning',
        'tensorboard',
        'vector-quantize-pytorch',
        'git+https://git@github.com/facebookresearch/encodec#egg=encodec', # Required for EnCodec model
    ],
    python_requires='>3.7'
)