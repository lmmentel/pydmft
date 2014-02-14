#!/usr/bin/env python

from distutils.core import setup

setup(
    name = "pydmft",
    version = "0.1",
    description = "Python module for post-CI dmft calculations",
    author = "Lukasz Mentel",
    author_email = "lmmentel@gmail.com",
    py_modules = ["pydmft", "postcidmft", "pydmftScf"],
)
