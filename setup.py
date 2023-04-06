from setuptools import setup, find_namespace_packages

setup(
    name="FineVC",
    version='0.0.1',
    description='Controllable pitch, phoneme, style, and identity in Voice Conversion.',
    author='Arsh Zahed',
    instal_requires=[
        'numpy',
        'scipy',
        'librosa',
        'torchvision',
        'soundfile',
        'pytorch-lightning',
        'tensorboard',
    ],
    python_requires='>3.7'
)