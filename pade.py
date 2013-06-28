#!/usr/bin/env python

import argparse
import gamessus 
from integrals import factor, ijkl
import numpy as np
import matplotlib.pyplot as mplot 
import os
import re
import sys
from scipy.optimize import minimize

def qoly(poly, arg):
    return poly(arg)**2/(1.0 + poly(arg)**2)

def pade_ac3(a, b, arg):
    
    coeffs = a*np.array([1.0, -4.0*b/3, -0.5, b, 0.0])
    p      = np.poly1d(coeffs)
    return p(arg)**2/(1.0 + p(arg)**2)

def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("logfile",
#                        help = "gamess-us log file")
#    args = parser.parse_args()

    x0 = np.array([-46.71847635, -0.18830162])

    x = np.arange(-0.5, 0.51, 0.01)
    y = pade_ac3(x0[0], x0[1],x)
    

    mplot.plot(x,y)    
    mplot.show()
#================================
# example of post-ci optimization
#================================
#    x0 = np.array([a,b])
#    energies = get_fci()
#    print energies

#    res = minimize(get_error, x0, args=(energies,), method='Nelder-Mead')
#    print res.x

if __name__ == "__main__":
    main()
