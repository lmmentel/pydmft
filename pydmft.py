#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as mplot 
import os
import re
import sys

# fortran interfaces fof reading files 
import onelectron
import twoelectron

class Gamess(object):
    '''Simple class for holding gamess filenames for a given job.'''
    def __init__(self, filename=""):
        if filename:
            self.filebase   = os.path.splitext(filename)[0] 
            self.inputfile  = self.filebase + ".inp"
            self.outputfile = self.filebase + ".log"
            self.twoeaofile = self.filebase + ".F08"
            self.twoemofile = self.filebase + ".F09"
            self.dictionary = self.filebase + ".F10"
            self.rdm2file   = self.filebase + ".F15"
            self.aos        = self.get_number_of_aos()
            self.mos        = self.get_number_of_mos()
        else:
            sys.exit('Gamess file not specified, exiting...') 

    def get_number_of_aos(self):
        '''Get the number of primitive cartesian gaussian basis functions from 
           Gamess log file'''

        p = r'NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS =\s*(\d+)'
        cp= re.compile(p)

        with open(self.outputfile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            match = cp.search(line)
            if match:
                return int(match.group(1))

    def get_number_of_mos(self):
        '''Get the number of molecular orbitals from Gammess log file.'''

#   check if the calculation is done in spherical of cartesian gaussians
#   base on the value of ISPHER keyword
        ispher_patt    = r'ISPHER=\s*(\-?\d{1}).*'
        var_space_patt = r'.*VARIATION SPACE IS\s*(\d+).*'
        c_ispher = re.compile(ispher_patt)
        c_var_space = re.compile(var_space_patt)

        with open(self.outputfile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            match = c_ispher.search(line)
            if match:
                ispher = int(match.group(1)) 

        n_ao = self.get_number_of_aos()
        if ispher == -1: 
            n_mo = n_ao
            return n_mo
        elif ispher == 1:
            for line in lines:
                match = c_var_space.search(line)
                if match:
                    n_mo = int(match.group(1))
                    return n_mo
        else:
            sys.exit("ispher not found") 


    def get_onee_size(self):
        '''Get the size of the vector holding upper (or lower) triangle
           of a square matrix of size nmos.'''
        n = self.mos
        return n*(n+1)/2

    def get_twoe_size(self):
        '''Get the size of the 1d vector holding upper (or lower) triangle
           of a supermatrix of size nmos (2RDM and two-electrons integrals) .'''
        n = self.get_onee_size()
        return n*(n+1)/2

    def get_rdm2(self):
        '''Read the 2rdm from the gamess-us file'''
# initialize numpy array to zeros
        rdm2 = np.zeros(self.get_twoe_size(), dtype=float)
# use gamess module to read the integrals from the file -filename-
        if os.path.exists(self.rdm2file):
            twoelectron.integrals.readinao(rdm2, self.rdm2file)
            return rdm2
        else: 
            sys.exit("File '{0:s}' doesn't exist, exiting...".format(self.rdm2file))

    def get_motwoe(self):
        '''Read the tow electron integrals from the gamess-us file'''
# initialize numpy array to zeros
        twoe = np.zeros(self.get_twoe_size(), dtype=float)
# use gamess module to read the integrals from the file -filename-
        if os.path.exists(self.twoemofile):
            twoelectron.integrals.readinmo(twoe, self.twoemofile)
            return twoe
        else: 
            sys.exit("File '{0:s}' doesn't exist, exiting...".format(self.twoemofile))

    def get_occupations(self):
        '''Get the natural orbitla occupation numbers from the section 21 of the 
           gamess-us dictionary file.'''
# initialize numpy array to zeros
        occ = np.zeros(self.mos, dtype=float)
        onelectron.dictionary.readreals(self.dictionary, occ, 21)
        return occ

def print_twoe(twoe, nbf):  
    '''Print the two-electron values.'''
    ij=0 
    for i in xrange(nbf):
        for j in xrange(i+1):
            ij += 1
            kl = 0
            for k in xrange(nbf):
                for l in xrange(k+1):
                    kl += 1
                    if ij >= kl:
                        if abs(twoe[ijkl(i,j,k,l)]) > 1.0e-10: 
                            print "{0:3d}{1:3d}{2:3d}{3:3d} {4:25.14f}".format(
                                i, j, k, l, twoe[ijkl(i,j,k,l)]) 

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

    gamess = Gamess(args.logfile)

    twoemo = gamess.get_motwoe()
    rdm2   = gamess.get_rdm2()
    occ = gamess.get_occupations()

    nbf = gamess.mos   
    
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
