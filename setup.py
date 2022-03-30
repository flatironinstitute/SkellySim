from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='skelly_sim',
    version='0.9.6',
    description='Simulate cytoskeletal systems with full hydrodynamics',
    long_description=long_description,
    url='https://github.com/flatironinstitute/skelly_sim/',
    author='Robert Blackwell',
    author_email='rblackwell@flatironinstitute.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists/Mathematicians',
        'License :: Apache 2',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    scripts=['scripts/skelly_precompute'],
    install_requires=['numba',
                      'numpy',
                      'scipy',
                      'msgpack',
                      'sklearn',
                      'toml',
                      'packaging',
                      'matplotlib',
                      'dataclass_utils',
                      'function_generator@git+https://github.com/blackwer/function_generator@fda8b44b5edf15d1677ca7433139a43778d6659d'],
)
