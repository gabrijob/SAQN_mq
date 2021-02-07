from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("AgentAPI", ["AgentAPI.pyx"],
        include_dirs=["./venv/include", "venv/lib/python3.8/site-packages/tensorflow/include"],
        libraries=["tensorflow_framework"], 
        library_dirs=["./venv/lib/python3.8/site-packages/tensorflow", "./venv/lib"]),
]

setup(
    name='SAQN Agent API',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
