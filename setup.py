from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("SAQNAgent", ["SAQNAgent.pyx"],
        include_dirs=["./"],
        #libraries=["tensorflow_framework"], 
        library_dirs=["./"]),
]

setup(
    name='SAQN Agent API',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
