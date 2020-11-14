import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='dst',
    version='v0.1.0',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['dst'],
    install_requires=required_packages,
    python_requires='>3.6.0',
    package_data={},

    entry_points={
        'console_scripts': [
            'dst = dst.run:entry_point'
        ]
    },
)
