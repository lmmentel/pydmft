#!/usr/bin/env python

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name = "pydmft",
    version = '0.1.0',
    description = "Python module for post-CI dmft calculations",
    author = "Lukasz Mentel",
    author_email = "lmmentel@gmail.com",
    packages = ["pydmft"],
    license = open('LICENSE.rst').read(),
    url = 'https://bitbucket.org/lukaszmentel/pydmft/',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: MIT',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords = 'density matrix functional theory quantum chemistry ',
)
