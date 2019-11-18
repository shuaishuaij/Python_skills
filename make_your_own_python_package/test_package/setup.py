#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
from setuptools import setup, find_packages

setup(
        name='test_tools',
        version=0.1,
        description='testing',
        author='jwli9',
        author_email='jwli9@iflytek.com',
        url='xxx',
        packages=find_packages(),
        include_package_data=True,
        include_requires=[],
        classifiers=[
                "operating System :: OS Independent",
                "Intended Audience :: Developers and Researchers",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python",
        ],
)