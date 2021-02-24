from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Datanog',
    ext_modules=cythonize("datanog.pyx"),
    zip_safe=False,
)