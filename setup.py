
from setuptools import setup, find_packages
  
setup(
    name='PyBindGPU',
    version='0.1.0',
    author='Johannes Blaschke, Darren Hsu',
    packages=find_packages("."),
    url='https://github.com/JBlaschke/PybindGPU',
    license='Apache 2',
    description='Light-weight python bindings to control GPUs',
    long_description=open('README.md').read(),
)

