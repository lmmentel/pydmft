#!/bin/bash

f2py2.7 -m onelectron  --fcompiler=intelem -c onelectron.f90 --f90flags="-O2 -ftz -auto -assume byterecl -vec-report0 -w95 -cm"
f2py2.7 -m twoelectron --fcompiler=intelem -c twoelectron.f90 --f90flags="-O2 -ftz -auto -assume byterecl -vec-report0 -w95 -cm"

# for test purposes

#ifort -o onelectron.x -i8 -O2 -ftz -auto -assume byterecl -vec-report0 -w95 -cm onelectron.f90

cp onelectron.so ..
cp twoelectron.so ..
