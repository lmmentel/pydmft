#!/usr/bin/env python2.7

import docopt
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import os
import re
import sys
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from subprocess import Popen
from gamessus import GamessReader, GamessParser
from operator import itemgetter
from collections import namedtuple

class Functional(object):
    '''
    Class for storing and manipulating the 1RDM functional objects
    '''

    def __init__(self, name=None):
        '''
        Initialize the class
        '''

        self.name = name

    def jkl_rdms(self):
        '''
        Get the J, K, L 1RDMS
        '''

        raise NotImplementedError('Should be available soon')

    def jkl_energy(self):
        '''
        Calculate three separate energy components corresponding to J, K and L
        terms.
        '''

        return np.sum(self.Jrdm), np.sum(self.Krdm), np.sum(self.Lrdm)

    def total_energy(self):
        '''
        Calculate the electron-electron potential energy corresponding to
        the functional.
        '''

        return np.sum(self.Jrdm) + np.sum(self.Krdm) + np.sum(self.Lrdm)

    def J_energy(self):
        '''
        Calculate the J-term contribution to the electron-electron potential
        energy corresponding to the functional.
        '''

        return np.sum(self.Jrdm)

    def K_energy(self):
        '''
        Calculate the K-term contribution to the electron-electron potential
        energy corresponding to the functional.
        '''

        return np.sum(self.Krdm)

    def L_energy(self):
        '''
        Calculate the L-term contribution to the electron-electron potential
        energy corresponding to the functional.
        '''

        return np.sum(self.Lrdm)


class BB(Functional):

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, **kwargs):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(nbf):
            ni = occ[i]
            for j in xrange(nbf):
                nj = occ[j]
                Jrdm[i, j] = ni*nj*Jints[i, j]
                Krdm[i, j] = -np.sqrt(ni*nj)*Kints[i, j]

        self.Jrdm = 0.5*Jrdm
        self.Krdm = 0.5*Krdm
        self.Lrdm = Lrdm

class PNOF4(Functional):
    '''
    Object representing PNOF4 functional,
    as presented in M. Piris et al. JCP 133, 111101 (2010)
    '''

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        Ne = np.sum(occ[homo+1:])

        for i in xrange(nbf):
            ni = occ[i]
            Jrdm[i, i] = 0.5*ni*Jints[i, i]
            for j in xrange(i):
                nj = occ[j]
                if i <= homo and j <= homo:
                    Jrdm[i, j] =  0.5*(ni*nj-(2.0-ni)*(2.0-nj))*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*(ni*nj-(2.0-ni)*(2.0-nj))*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5*np.sqrt((2.0-ni)*(2.0-nj))*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j > homo:
                    Lrdm[i, j] = 0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j <= homo:
                    Jrdm[i, j] =  0.5*(ni*nj-((2.0-Ne)/Ne)*ni*(2.0-nj))*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*(ni*nj-((2.0-Ne)/Ne)*ni*(2.0-nj))*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    t = (2.0-nj)*ni/Ne
                    Lrdm[i, j] = -0.5*np.sqrt(t*(nj-ni+t))*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

class DD(Functional):
    '''
    Object representing DMFT functional derived from CIDD ansatz,
    see E. J. Baerends, "Guidelines for DMFT... (05.02.2014)"
    '''

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        Ne = np.sum(occ[homo+1:])

        for i in xrange(nbf):
            ni = occ[i]
            Jrdm[i, i] = 0.5*ni*Jints[i, i]
            for j in xrange(i):
                nj = occ[j]
                if i <= homo and j <= homo:
                    Jrdm[i, j] =  (ni+nj-2.0)*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.5*(ni+nj-2.0)*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = 0.5*np.sqrt((2.0-ni)*(2.0-nj))*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j > homo:
                    Lrdm[i, j] = 0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j <= homo:
                    Jrdm[j, i] = 0.5*ni*(2.0-(2.0-nj)/Ne)*Jints[j, i]
                    Jrdm[i, j] = Jrdm[j, i]
                    Krdm[j, i] = -0.25*ni*(2.0-(2.0-nj)/Ne)*Kints[j, i]
                    Krdm[i, j] = Krdm[j, i]
                    Lrdm[j, i] = -np.sqrt((1.0-Ne)/Ne)*np.sqrt(ni*(2.0-nj))*Kints[j, i]
                    Lrdm[i, j] = Lrdm[j, i]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

class ELS1(Functional):
    '''
    Object representing ELS1 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    '''

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):
        '''
        Calculate the energy matrix from the ELS1 functional.
        '''

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5*ni*Jints[i, i]
            for j in xrange(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*ni*nj*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Lrdm[i, j] = -0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

class ELS2(Functional):
    '''
    Object representing ELS2 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    '''

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):
        '''
        Calculate the energy matrix from the ELS2 functional.
        '''

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5*ni*Jints[i, i]
            for j in xrange(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*ni*nj*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*ni*nj*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

class ELSPade(Functional):
    '''
    Object representing ELS DMFT functional that contains a parametrization in
    terms of a rational Pade approximant,
    see L. Mentel et al. "Consistent extension..." (2014)"
    '''

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None, a=None, b=None):
        '''
        Calculate the energy matrix from the ELS2 functional.
        '''


        # if not provide use old AC3 parameters
        if not a:
            a = 16.8212438522           # AC3 a1
        if not b:
            b =  2.61112559818          # AC3 b1
        #a = 19.569076362            # AC3 a2
        #b =  1.41823370195          # AC3 b2

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5*ni*Jints[i, i]
            for j in xrange(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*ni*nj*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5*np.sqrt(ni*nj)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j < homo and i > homo:
                    pade = self.pade_ac3(a, b, 0.5*ni-0.5)
                    Jrdm[i, j] = 0.5*ni*nj*Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25*ni*nj*pade*Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5*np.sqrt(ni*nj)*(1-pade)*Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

    def pade_ac3(a, b, arg):
        '''
        Pade approximant used in the intrpolation scheme.
        '''

        coeffs = a*np.array([1.0, -4.0*b/3, -0.5, b, 0.0])
        p      = np.poly1d(coeffs)
        return p(arg)**2/(1.0 + p(arg)**2)

class Exact(Functional):
    '''
    Exact functional based on the exact 2RDM
    '''

    def Coulomb(self, occ=None, Jints=None):
        '''
        Calculate the exact coulomb energy and return the matrix already multiplied
        by a factor of 0.5.
        '''

        coulomb = np.zeros((occ.size, occ.size), dtype=float)

        for i in xrange(occ.size):
            for j in xrange(occ.size):
                coulomb[i, j] = occ[i]*occ[j]*Jints[i, j]

        return 0.5*coulomb


    def jk_rdms(self, nbf=None, rdm2=None, Jints=None, Kints=None, homo=None):
        '''
        Calculate the primitive J and K rdms using exact 2RDM.
        '''

        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(nbf):
            Jrdm[i, i] = 0.5*rdm2[ijkl(i,i,i,i)]*Jints[i, i]
            for j in xrange(i):
                Jrdm[i, j] = 0.5*rdm2[ijkl(i,i,j,j)]*Jints[i, j]
                Jrdm[j, i] = Jrdm[i, j]
                Krdm[i, j] = rdm2[ijkl(i,j,i,j)]*Kints[i, j]
                Krdm[j, i] = Krdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

    def nonjk_rdm(self, nbf=None, rdm2=None, twoe=None):

        Njklrdm = np.zeros((nbf, nbf), dtype=float)

        for i in xrange(nbf):
            for k in xrange(nbf):
                for l in xrange(nbf):
                    if k != l:
                        Njklrdm[i, i] += rdm2[ijkl(i,i,k,l)]*twoe[ijkl(i,i,k,l)]
        for i in xrange(nbf):
            for j in xrange(nbf):
                if i != j:
                    gam = 0.0
                    exch = 2.0*rdm2[ijkl(i,j,i,j)]*twoe[ijkl(i,j,i,j)]
                    for k in xrange(nbf):
                        for l in xrange(nbf):
                            gam += rdm2[ijkl(i,j,k,l)]*twoe[ijkl(i,j,k,l)]
                    Njklrdm[i, j] = gam - exch

        self.Njklrdm = 0.5*Njklrdm


def get_else_matrix(nocc, twoeno, homo, a, b):
    '''Calculate the energy matrix from the ELS functional.'''
    els = np.zeros((nocc.size, nocc.size), dtype=float)

    for i in xrange(nocc.size):
        els[i, i] = 0.5*nocc[i]*twoeno[ijkl(i,i,i,i)]
        for j in xrange(i):
            if i < homo and j < homo:
                els[i, j] = -0.25*nocc[i]*nocc[j]*twoeno[ijkl(i,j,i,j)] +\
                             0.5*nocc[i]*nocc[j]*twoeno[ijkl(i,i,j,j)]
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
    return {"inner diagonal"    : i_diagonal,
            "inner offdiagonal" : i_offdiagonal,
            "outer diagonal"    : o_diagonal,
            "outer offdiagonal" : o_offdiagonal,
            "inner-outer"       : io,
            "total"             : total}

def print_components(comps_dict):
    for key in sorted(comps_dict.iterkeys()):
        print "{0:<25s} : {1:>15.10f}".format(key, comps_dict[key])



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

def contains(theString,theQuery):
    return theString.find(theQuery) > -1


def get_fci():

    filenames = []
    for (path, dirs, files) in os.walk(os.getcwd()):
        for fileItem in files:
            if contains(fileItem, 'NO.log'):
                filenames.append(os.path.join(path,fileItem))    
    energies = []
    for file in filenames:
        dir, log = os.path.split(file)
        os.chdir(dir)
        gamess = Gamess(log)
        print log, gamess.get_number_of_aos(), gamess.homo 
        twoemo = gamess.get_motwoe()
        rdm2   = gamess.get_rdm2()
        nbf    = gamess.mos
        exact_j, exact_k = get_exact_jk(rdm2, twoemo, nbf)
        exact_njk = get_exact_nonjk(rdm2, twoemo, nbf)
        dd = decompose(exact_j+exact_k+exact_njk, gamess.homo)
        energies.append({file : dd["inner-outer"]})
        print "\n{1}\nExact for {0}\n{1}\n".format(log, "="*(len(log)+10)) 
        print_components(dd)
    return energies

def get_error(x0, *args):

    jobs = args[0]
    diffs = []
    for job in jobs:
        filepath = job.keys()[0]
        energy   = job[filepath]
        dir, file = os.path.split(filepath)
        os.chdir(dir)
        gamess = Gamess(file)
        twoeno = gamess.get_motwoe()
        nbf    = gamess.mos
        nocc   = gamess.get_occupations()
        els    = get_else_matrix(nocc, twoeno, gamess.homo, x0[0], x0[1])
        dd     = decompose(els, gamess.homo)
        diffs.append(energy-dd["inner-outer"])
        print "\n{1}\nELS for {0}\n{1}\n".format(file, "="*(len(file)+8))
        print_components(dd)
    diffs = np.asarray(diffs)
    error = np.sqrt(np.add.reduce(diffs*diffs))
    print "Error = {e:>14.10f}  Parameters: {p:s}".format(e=error, p="  ".join(["{:>14.10f}".format(x) for x in x0]))
    print "-"*106
    return error


def splitlist(l, n):
    '''
    Split a list 'l' into lists of at most 'n' elements.
    '''

    if len(l) % n == 0:
        splits = len(l)/n
    elif len(l) % n != 0 and len(l) > n:
        splits = len(l)/n+1
    else:
        splits = 1

    for i in xrange(splits):
        yield l[n*i:n*i+n]

def get_error_scf(x0, *args):

    executable = "/home/lmentel/Source/dmft/dmft_code/dmft.x"
    renergy = r'\s*Electron Interaction Energy\s*(\-?\d+\.\d+)'
    jobs = args[0]
    diffs = []
    for jobbatch in splitlist(jobs, 6):
        processes = []
        for job in jobbatch:
            filepath = job.keys()[0]
            energy   = job[filepath]
            dir, file = os.path.split(filepath)
            os.chdir(dir)
            dmftinput = os.path.splitext(file)[0]+"_dmft.inp"
            write_dmft_input(file, dmftinput, functional=9, a1=x0[0], b1=x0[1])
            out = open(os.path.splitext(dmftinput)[0]+".out", 'w')
            p = Popen([executable, dmftinput], stdout=out, stderr=out)
            out.close()
            processes.append(p)

        for p in processes: p.wait()

    for job in jobs:
        filepath = job.keys()[0]
        energy   = job[filepath]
        dir, file = os.path.split(filepath)
        os.chdir(dir)
        dmftoutput = os.path.splitext(file)[0]+"_dmft.out"
        with open(dmftoutput, 'r') as f:
            contents = f.readlines()
        dmft_energy = get_info_float(contents, renergy)
        diffs.append(energy-dmft_energy)
        print "Processed {0}   Exact energy: {1:15.10f}   DMFT energy: {2:15.10f}".format(file, energy, dmft_energy)
    diffs = np.asarray(diffs)
    error = np.sqrt(np.add.reduce(diffs*diffs))
    print "Error = {e:>14.10f}  Parameters: {p:s}".format(e=error, p="  ".join(["{:>14.10f}".format(x) for x in x0]))
    print "-"*106
    return error

def get_jkints(twoe, nb):
    '''
    get the coulomb and exchange integrals as two index quantities.
    '''

    Jints = np.zeros((nb, nb), dtype=float)
    Kints = np.zeros((nb, nb), dtype=float)

    for i in xrange(nb):
        for j in xrange(nb):
            Jints[i, j] = twoe[ijkl(i, i, j, j)]
            Kints[i, j] = twoe[ijkl(i, j, j, i)]

    return Jints, Kints

def print_energies(fun_name, energies):


    print "\n{0:^118s}\n{1:^118s}\n{0:^118s}\n".format("="*len(fun_name), fun_name)

    print "{0:<30s}  {1:^6s}{2:^20s}{3:^20s}{4:^20s}{5:^20s}\n{6:s}".format("File",
            "R", "J energy", "K energy", "L energy", "Total E_ee", "="*118)
    for row in sorted(energies, key=itemgetter("dist")):
        print "{0:<30s}  {1:6.2f}{2:20.10f}{3:20.10f}{4:20.10f}{5:20.10f}".format(row["file"],
            row["dist"], row["j"], row["k"], row["l"], row["total"])
    print

def lplot(casesd, functs=[], save=False):

    x, te, onee, twoe, nucr, hf = np.loadtxt(casesd['tablefile'], dtype=float, comments="#", usecols=(0,1,2,3,4,5), unpack=True)

    te_s = interp1d(x, te, kind='cubic')
    hf_s = interp1d(x, hf, kind='cubic')
    xnew = np.linspace(min(x), max(x), 100)

    rc('font', size=18.0)
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('figure', autolayout=True)

    plt.figure(figsize=(12, 8))

    plt.title(casesd['name'])
    plt.xlabel(r'Internuclear distance $R$ [bohr]')
    plt.ylabel(r'Energy [hartree]')
    plt.plot(xnew, te_s(xnew), '-', linewidth='2.0', label="Exact")
    plt.plot(xnew, hf_s(xnew), '-', label="HF")
    for funct in functs:
        for name, energies in funct.items():
            fun_s = interp1d(x, onee+nucr+energies, kind='cubic')
            plt.plot(xnew, fun_s(xnew), label=name)

    plt.legend(loc="best", frameon=False)

    if save:
        figname = casesd['name']+".pdf"
        plt.savefig(figname)
    else:
        plt.show()


def main():
    '''
    Usage:
        postcidmft.py <molecule>

    Options:
        molecule  molecule to work with
    '''

    workdir = '/home/lmentel/work/jobs/dmft'
    cases = {'lih'  : {'name'     : r'LiH',
                       'workdir'  : os.path.join(workdir, 'LiH'),
                       'tablefile': os.path.join(workdir, 'LiH.tbl')},
             'li2'  : {'name'     : r'Li$_2$',
                       'workdir'  : os.path.join(workdir, 'Li2'),
                       'tablefile': os.path.join(workdir, 'Li2.tbl')},
             'behp' : {'name'     : r'BeH$^+$',
                      'workdir'   : os.path.join(workdir, 'BeHp'),
                      'tablefile' : os.path.join(workdir, 'BeHp.tbl')},
            }

    args = docopt.docopt(main.__doc__, help=True)
    if not args['<molecule>'] in cases.keys():
        sys.exit('wrong molecule: {0:s}, should be one of (lih, behp, li2)')


    functionals = [BB('BB'), PNOF4("PNOF4"), DD("DD"), ELS1("ELS1"), ELS2("ELS2")]

    data = {}
    for f in functionals:
        data[f.name] = []
    dist_re = re.compile(r'.*_(\d+\.\d+)_.*')

    for (path, dirs, files) in os.walk(cases[args['<molecule>']]['workdir']):
        for fileitem in files:
            if "_NO.log" in fileitem:
                log = os.path.join(path, fileitem)
                gp  = GamessParser(log)
                gr  = GamessReader(log)

                twoemo = gr.read_twoemo()
                occ    = np.abs(gr.read_occupations())
                Jints, Kints = get_jkints(twoemo, gp.get_number_of_mos())

                match = dist_re.search(fileitem)
                for f in functionals:
                    f.jkl_rdms(occ=occ, Jints=Jints, Kints=Kints, homo=gp.get_homo())

                for f in functionals:
                    data[f.name].append({"file"  : fileitem,
                                "dist"  : float(match.group(1)),
                                "j"     : f.J_energy(),
                                "k"     : f.K_energy(),
                                "l"     : f.L_energy(),
                                "total" : f.total_energy(),
                               })
    datap = []
    for funct, values in data.items():
        dde = [x["total"] for x in sorted(values, key=itemgetter("dist"))]
        datap.append({funct : dde})
        print_energies(funct, values)

    lplot(cases[args['<molecule>']], datap, save=False)


if __name__ == "__main__":
    main()
