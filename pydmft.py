#!/usr/bin/env python

import argparse
import gamessus 
from integrals import factor, ijkl
import numpy as np
import matplotlib.pyplot as mplot 
import os
import re
import sys

def get_coulomb_1rdm(occ, twoe):
    '''Calculate the exact coulomb energy and return the matrix already
        multiplied by a factor of 0.5.'''
    coulomb = np.zeros((occ.size, occ.size), dtype=float)

    for i in xrange(occ.size):
        for j in xrange(occ.size):
            coulomb[i, j] = occ[i]*occ[j]*twoe[ijkl(i,i,j,j)]

    return 0.5*coulomb

def get_exact_jk(rdm2, twoe, nbf):
    
    exact_j = np.zeros((nbf, nbf), dtype=float)
    exact_k = np.zeros((nbf, nbf), dtype=float)

    for i in xrange(nbf):
        exact_j[i, i] = rdm2[ijkl(i,i,i,i)]*twoe[ijkl(i,i,i,i)]
        for j in xrange(i):
            exact_j[i, j] = rdm2[ijkl(i,i,j,j)]*twoe[ijkl(i,i,j,j)] 
            exact_j[j, i] = exact_j[i, j]
            exact_k[i, j] = rdm2[ijkl(i,j,i,j)]*twoe[ijkl(i,j,i,j)]
            exact_k[j, i] = exact_k[i, j]
    return 0.5*exact_j, exact_k

def get_exact_nonjk(rdm2, twoe, nbf):
    
    exact_njk = np.zeros((nbf, nbf), dtype=float)

    for i in xrange(nbf):
        for k in xrange(nbf):
            for l in xrange(nbf):
                if k != l:
                    exact_njk[i, i] += rdm2[ijkl(i,i,k,l)]*twoe[ijkl(i,i,k,l)]
    for i in xrange(nbf):
        for j in xrange(nbf):
            if i != j:
                gam = 0.0
                exch = 2.0*rdm2[ijkl(i,j,i,j)]*twoe[ijkl(i,j,i,j)]
                for k in xrange(nbf):
                    for l in xrange(nbf):
                        gam += rdm2[ijkl(i,j,k,l)]*twoe[ijkl(i,j,k,l)]
                exact_njk[i, j] = gam - exch
    return 0.5*exact_njk
                        
def get_els_matrix(nocc, twoeno, homo):
    '''Calculate the energy matrix from the ELS functional.'''
    els = np.zeros((nocc.size, nocc.size), dtype=float)

    for i in xrange(nocc.size):
        els[i, i] = 0.5*nocc[i]*twoeno[ijkl(i,i,i,i)]
        for j in xrange(i):
            if i < homo and j < homo:
                els[i, j] = -0.25*nocc[i]*nocc[j]*twoeno[ijkl(i,j,i,j)] +\
                             0.5*nocc[i]*nocc[j]*twoemo[ijkl(i,i,j,j)]
                els[j, i] = els[i, j]
            elif j == homo and i > homo:
                els[i, j] = -0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] 
                els[j, i] = els[i, j]
            elif j > homo and i > homo:
                els[i, j] = 0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] 
                els[j, i] = els[i, j]
            else:
                els[i, j] = -0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] +\
                           0.5*nocc[i]*nocc[j]*twoeno[ijkl(i,i,j,j)] 
                els[j, i] = els[i, j]
    return els

def get_else_matrix(nocc, twoeno, homo, a, b):
    '''Calculate the energy matrix from the ELS functional.'''
    els = np.zeros((nocc.size, nocc.size), dtype=float)

    for i in xrange(nocc.size):
        els[i, i] = 0.5*nocc[i]*twoeno[ijkl(i,i,i,i)]
        for j in xrange(i):
            if i < homo and j < homo:
                els[i, j] = -0.25*nocc[i]*nocc[j]*twoeno[ijkl(i,j,i,j)] +\
                             0.5*nocc[i]*nocc[j]*twoemo[ijkl(i,i,j,j)]
                els[j, i] = els[i, j]
            elif j == homo and i > homo:
                els[i, j] = -0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] 
                els[j, i] = els[i, j]
            elif j > homo and i > homo:
                els[i, j] = 0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] 
                els[j, i] = els[i, j]
            else:
                els[i, j] = -0.5*np.sqrt(nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] +\
                             0.5*(np.sqrt(nocc[i]*nocc[j])-0.5*nocc[i]*nocc[j]) *\
                             twoeno[ijkl(i,j,i,j)]*pade_ac3(a,b,0.5*nocc[i]-0.5) +\
                           0.5*nocc[i]*nocc[j]*twoeno[ijkl(i,i,j,j)] 
                els[j, i] = els[i, j]
    return els


def decompose(ematrix, homo):
    diagonal = np.trace(ematrix)
    i_diagonal = np.trace(ematrix[:homo, :homo])
    o_diagonal = np.trace(ematrix[homo:, homo:])
    i_offdiagonal = np.sum(ematrix[:homo, :homo]) - np.trace(ematrix[:homo, :homo])
    o_offdiagonal = np.sum(ematrix[homo:, homo:]) - np.trace(ematrix[homo:, homo:])
    io = 2.0*np.sum(ematrix[homo:, :homo])
    total    = np.sum(ematrix)
    print "Inner Diagonal    : {0:15.10f}".format(i_diagonal)
    print "Inner Offdiagonal : {0:15.10f}".format(i_offdiagonal)
    print
    print "Outer Diagonal    : {0:15.10f}".format(o_diagonal)
    print "Outer Offdiagonal : {0:15.10f}".format(o_offdiagonal)
    print 
    print "Inner-Outer       : {0:15.10f}".format(io)
    print "{0:s}".format("-"*35)
    print "Diagonal          : {0:15.10f}".format(diagonal)
    print "Total             : {0:15.10f}".format(total)


def qoly(poly, arg):
    return poly(arg)**2/(1.0 + poly(arg)**2)

def pade_ac3(a, b, arg):
    
    coeffs = a*np.array([1.0, -4.0*b/3, -0.5, b, 0.0])
    p      = np.poly1d(coeffs)
    return p(arg)**2/(1.0 + p(arg)**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile",
                        help = "gamess-us log file with two electron integrals")
    args = parser.parse_args()

#    x  = np.arange(-0.5, 0.51, 0.01)
#    y  = pade_ac3(2.0, 3.0, x)
#    y1 = pade_ac3(16.8212438522, 2.61112559818e-04, x)
#    mplot.plot(x, y)
#    mplot.plot(x, y1)
#    mplot.show() 

    gamess = gamessus.GamessParser(args.logfile)

    twoemo = gamess.read_twoemo()
    rdm2   = gamess.read_rdm2()
    occ = gamess.read_occupations()

    nbf = gamess.get_number_of_mos()   
    
    a = 16.8212438522e00
    b = 2.61112559818e-04

    coulomb = get_coulomb_1rdm(occ, twoemo)
    els_ee  = get_els_matrix(occ, twoemo, 1)
    else_ee = get_else_matrix(occ, twoemo, 1, a, b)
    print "\n{0}\nExact Coulomb\n{0}\n".format("="*13) 
    decompose(coulomb, 1)
    print "\n{0}\nXC ELS\n{0}\n".format("="*6) 
    decompose(els_ee-coulomb, 1)
    print "\n{0}\nTotal ELS\n{0}\n".format("="*9) 
    decompose(els_ee, 1)
    print "\n{0}\nTotal ELSE\n{0}\n".format("="*10) 
    decompose(else_ee, 1)
    exact_j, exact_k = get_exact_jk(rdm2, twoemo, nbf)
    exact_njk = get_exact_nonjk(rdm2, twoemo, nbf)
    print "\n{0}\nExact non-JK\n{0}\n".format("="*12) 
    decompose(exact_j+exact_k+exact_njk, 1)
    

#    x = np.arange(nbf)
#    gamma = np.zeros(nbf, dtype=float)
#    for i in xrange(nbf):
#        print "{0:4d} {1:15.10f} {2:15.10f}".format(x[i], occ[i], rdm2[ijkl(i,i,i,i)]) 
 #       gamma[i] = rdm2[ijkl(i,i,i,i)]
      
#    mplot.plot(x, gamma-occ, 'bo')
#    mplot.grid(True)
#    mplot.rc('text', usetex=True)
#    mplot.rc('font', family='serif')
#    mplot.title(r'Difference n_i - Gamma_iiii')
#    mplot.xlabel(r'Orbital')
#    mplot.ylabel(r'Difference')
#    mplot.plot(x, gamma-np.multiply(occ,occ))
#    mplot.show() 

#    print "E_ee = {0:25.14f}".format(0.5*np.sum(np.multiply(twoemo, rdm2)))
if __name__ == "__main__":
    main()
