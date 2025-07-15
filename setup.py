import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="your_project_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.1",
        "timm>=0.9.2",
        "transformers>=4.30.2",
        "mmengine>=0.8.5",
        "h5py>=3.9.0",
        "av>=10.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.1",
        "numpy>=1.24.3",
        "pyarrow>=13.0.0",
    ],
)    