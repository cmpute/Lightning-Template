from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FancyModel',
    version='0.1.0',
    description='A neural network library based on Pytorch Lightning',
    author='Jacob Zhong',
    author_email='cmpute@qq.com',
    license='Apache License 2.0',
    packages=find_packages(exclude=['tools', 'data', 'output']),
    cmdclass={
        'build_ext': BuildExtension,
    },
    ext_modules=[
        # Put your C++/CUDA extensions here
    ],
)
