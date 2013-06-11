#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as mplot 
import os
import sys

# fortran interfaces fof reading files 
import onelectron
import twoelectron

class GamessFiles(object):
    '''Simple class for holding gamess filenames for a given job.'''
    def __init__(self, filename=""):
        if filename:
            self.filebase = os.path.splitext(filename)[0] 
            self.input      = self.filebase + ".inp"
            self.output     = self.filebase + ".log"
            self.twoeao     = self.filebase + ".F08"
            self.twoemo     = self.filebase + ".F09"
            self.dictionary = self.filebase + ".F10"
            self.rdm2       = self.filebase + ".F15"
        else:
            sys.exit('Gamess file not specified, exiting...') 

def pade(poly, arg):
    return poly(arg)**2/(1.0 + poly(arg)**2)

def pade_ac3(a, b, arg):
    
    coeffs = a*np.array([1.0, -4.0*b/3, -0.5, b, 0.0])
    p      = np.poly1d(coeffs)
    return p(arg)**2/(1.0 + p(arg)**2)

def factor(i,j,k,l):
    '''Based on the orbitals indices return the factor that takes into account 
       the index permutational symmetry.'''
    if i == j and k == l and i == k:
        fijkl = 1.0
    elif i == j and k == l:
        fijkl = 2.0
    elif (i == k and j == l) or (i == j and i == k) or (j == k and j == l) or (i == j or k == l):
        fijkl = 4.0
    else: 
        fijkl = 8.0
    return fijkl 

def ijkl(i,j,k,l):
    '''Based on the four orbital indices i,j,k,l return the address 
       in the 1d vector.'''
    ij = max(i, j)*(max(i, j) + 1)/2 + min(i, j)
    kl = max(k, l)*(max(k, l) + 1)/2 + min(k, l)
    return max(ij, kl)*(max(ij, kl) + 1)/2 + min(ij, kl)

def get_rdm2(filename, nb):
    '''Read the 2rdm from the gamess-us file'''
# calculate dimensions first 
    n1 = nb*(nb+1)/2
    n2 = n1*(n1+1)/2
# initialize numpy array to zeros
    rdm2 = np.zeros(n2, dtype=float)
# use gamess module to read the integrals from the file -filename-
    twoelectron.integrals.readinao(rdm2, filename)
    return rdm2

def get_motwoe(filename, nb):
    '''Read the tow electron integrals from the gamess-us file'''
# calculate dimensions first 
    n1 = nb*(nb+1)/2
    n2 = n1*(n1+1)/2
# initialize numpy array to zeros
    twoe = np.zeros(n2, dtype=float)
# use gamess module to read the integrals from the file -filename-
    twoelectron.integrals.readinmo(twoe, filename)
    return twoe

def get_occupations(gfiles, nb):
    '''Get the natural orbitla occupation numbers from the section 21 of the 
       gamess-us dictionary file.'''
    
    occ = np.zeros(nb, dtype=float)
    onelectron.dictionary.readreals(gfiles.dictionary, occ, 21)
    return occ

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nbf",
                        type = int,
                        help = "number of basis functions")
    parser.add_argument("logfile",
                        help = "gamess-us log file with two electron integrals")
    args = parser.parse_args()

    x  = np.arange(-0.5, 0.51, 0.01)
    y  = pade_ac3(2.0, 3.0, x)
    y1 = pade_ac3(16.8212438522, 2.61112559818e-04, x)
#    mplot.plot(x, y)
#    mplot.plot(x, y1)
#    mplot.show() 

    gfiles = GamessFiles(args.logfile)
    twoemo = get_motwoe(gfiles.twoemo, args.nbf)
    rdm2   = get_rdm2(gfiles.rdm2, args.nbf)
    nbf = args.nbf   
    
    occ = get_occupations(gfiles, nbf)
    for i in xrange(nbf):
        print "{0:4d} {1:10.6f}".format(i, occ[i]) 
        
    ij=0 
    for i in xrange(nbf):
        for j in xrange(i+1):
            ij += 1
            kl = 0
            for k in xrange(nbf):
                for l in xrange(k+1):
                    kl += 1
                    if ij >= kl:
                        twoemo[ijkl(i,j,k,l)] = factor(i,j,k,l)*twoemo[ijkl(i,j,k,l)]
                        if abs(twoemo[ijkl(i,j,k,l)]) > 1.0e-10: 
                            print "{0:3d}{1:3d}{2:3d}{3:3d} {4:>4d} {5:25.14f}{6:25.14f}".format(
                                i, j, k, l, int(factor(i,j,k,l)), twoemo[ijkl(i,j,k,l)], rdm2[ijkl(i,j,k,l)]) 

    print "E_ee = {0:25.14f}".format(0.5*np.sum(np.multiply(twoemo, rdm2)))
if __name__ == "__main__":
    main()
