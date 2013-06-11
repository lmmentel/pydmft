#!/bin/bash

f2py -m onelectron  --fcompiler=gfortran -c onelectron.f90
f2py -m twoelectron --fcompiler=gfortran -c twoelectron.f90

cp onelectron.so ..
cp twoelectron.so ..
