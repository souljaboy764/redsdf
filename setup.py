from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name='redsdf',
    version='1.0.0',
    keywords='manifold',
    description='a library to train manifold',
    license='MIT',
    url='https://git.ias.informatik.tu-darmstadt.de/puze_liu/human_manifold',
    packages=['redsdf'],
    install_requires=requires_list,
    include_package_data=True,
)
