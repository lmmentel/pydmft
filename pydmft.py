#!/usr/bin/env python

from integrals import factor, ijkl
from scipy.optimize import minimize
import argparse
import basis
import gamessus
import math
import matplotlib.pyplot as mplot
import molecule
import numpy as np
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
                           0.5*nocc[i]*nocc[j]*twoeno[ijkl(i,i,j,j)] 
                els[j, i] = els[i, j]
    return els

def get_elslin_matrix(nocc, twoeno, homo):
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
                els[i, j] = -0.25*nocc[i]*nocc[j]*twoeno[ijkl(i,j,i,j)] +\
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

def get_functional12_matrix(nocc, twoeno, homo, a, b):
    '''Calculate the energy matrix from the ELS12 functional.'''
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
                els[i, j] = (-a*0.5*np.sqrt(nocc[i]*nocc[j])-b*0.25*nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] +\
                           0.5*nocc[i]*nocc[j]*twoeno[ijkl(i,i,j,j)] 
                els[j, i] = els[i, j]
    return els

def get_functional12sin_matrix(nocc, twoeno, homo, a):
    '''Calculate the energy matrix from the ELS12 functional.'''
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
                els[i, j] = (-(1.0-math.sin(a)**2)*0.5*np.sqrt(nocc[i]*nocc[j])-math.sin(a)**2*0.25*nocc[i]*nocc[j])*twoeno[ijkl(i,j,i,j)] +\
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

def print_components(comps_dict, name=None):
    for key in sorted(comps_dict.iterkeys()):
        if name:
            print "{0:<30s} {1:<25s} : {2:>15.10f}".format(name, key, comps_dict[key])
        else:
            print "{0:<25s} : {1:>15.10f}".format(key, comps_dict[key])

def qoly(poly, arg):
    return poly(arg)**2/(1.0 + poly(arg)**2)

def pade_ac3(a, b, arg):
    
    coeffs = a*np.array([1.0, -4.0*b/3, -0.5, b, 0.0])
    p      = np.poly1d(coeffs)
    return p(arg)**2/(1.0 + p(arg)**2)

def contains(theString,theQuery):
    return theString.find(theQuery) > -1

def get_fci(testset=None):

    filenames = testset
    #    for (path, dirs, files) in os.walk(os.getcwd()):
    #        for fileItem in files:
    #            if contains(fileItem, 'NO.log'):
    #                filenames.append(os.path.join(path,fileItem))    
    energies = []
    for file in filenames:
        dir, log = os.path.split(file)
        os.chdir(dir)
        gamess = gamessus.GamessParser(log)
        print log, 
        twoemo = gamess.read_twoemo()
        rdm2   = gamess.read_rdm2()
        nbf    = gamess.get_number_of_mos()
        exact_j, exact_k = get_exact_jk(rdm2, twoemo, nbf)
        exact_njk = get_exact_nonjk(rdm2, twoemo, nbf)
        dd = decompose(exact_j+exact_k+exact_njk, gamess.get_homo())
        energies.append({file : dd["inner-outer"]})
        print "\n{1}\nExact for {0}\n{1}\n".format(log, "="*(len(log)+10)) 
        print_components(dd)
    return energies

def get_fci_energies(testset=None):


    energies = []
    for fileitem in testset:
        dir, log = os.path.split(fileitem)
        os.chdir(dir)
        gp = gamessus.GamessParser(log)
        energies.append([fileitem, gp.get_ci_ee_energy()])
    return energies

def get_error(x0, *args):
    
    jobs = args[0]
    diffs = []
    for job in jobs:
        filepath = job[0]
        energy   = job[1]
        dir, log = os.path.split(filepath)
        os.chdir(dir)
        gamess = gamessus.GamessParser(log)
        twoeno = gamess.read_twoemo()
        nocc   = gamess.read_occupations()
        els    = get_functional12sin_matrix(nocc, twoeno, gamess.get_homo(), x0[0])
        dd     = decompose(els, gamess.get_homo())
        diffs.append(energy-dd["total"])
        print "\n{1}\nELS for {0}\n{1}\n".format(log, "="*(len(log)+8)) 
        print "Exact:  {0:>14.10f}  Approximate:  {1:>14.10f}".format(energy, dd["total"])
        print_components(dd)
    diffs = np.asarray(diffs)
    error = np.sqrt(np.add.reduce(diffs*diffs))
    print "Error = {e:>14.10f}  Parameters: {p:s}".format(e=error, p="  ".join(["{:>14.10f}".format(x) for x in x0]))
    print "-"*106
    return error


def run_postci_pes(x0):

    bs = basis.Basis("/home/lmentel/work/Basis_Sets/EMSL", "cc-pvtz")

    lih_distances = [2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 
                 3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 
                 4.00, 4.50, 5.00, 5.50, 6.00, 7.00, 8.00, 9.00, 10.00]
    behp_distances = [1.80, 2.10, 2.30, 2.50, 2.70, 2.90, 3.20, 3.50, 4.00, 4.50, 
                  5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
    li2_distances = [2.50, 3.00, 3.50, 4.00, 4.50, 4.75, 5.05, 5.25, 5.50, 6.00,
                 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 14.00, 16.00]

    molecules = []
    jobs = []
    #for d in lih_distances:    
    #    molecules.append(molecule.Molecule('LiH', [(1, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, d))], unique=[0,1]))

    for d in li2_distances:    
        molecules.append(molecule.Molecule('Li2', [(3, (0.0, 0.0, -d/2.0)), (3, (0.0, 0.0, d/2.0))], unique=[0]))

    #for d in behp_distances:    
    #    molecules.append(molecule.Molecule('BeHp', [(4, (0.0, 0.0, -d/2.0)), (1, (0.0, 0.0, d/2.0))], unique=[0,1], charge=1))


    for mol in molecules:
        title = "_".join([mol.name, bs.name, str(mol.get_distance(0,1))]) + ".inp" 
        jobs.append(gamessus.Gamess(mol, bs, title))


    for job in jobs:
        os.chdir(job.filebase)
        gp = gamessus.GamessParser(job.outputfile.replace(".log", "_NO.log"))
        twoeno = gp.read_twoemo()
        nocc   = gp.read_occupations()
        #rdm2   = gp.read_rdm2()
        #nbf    = gp.get_number_of_mos()
        #exj, exk = get_exact_jk(twoeno, rdm2, nbf)
        #exnjk = get_exact_nonjk(twoeno, rdm2, gp.get_number_of_mos())
        em = get_functional12_matrix(nocc, twoeno, gp.get_homo(), x0[0], x0[1])
        dd = decompose(em, gp.get_homo())
        #dd = decompose(exj+exk+exnjk, gp.get_homo())
        print_components(dd, gp.logname)
        print "-"*100
        os.chdir("..")
#    for job in jobs:
#        os.chdir(job.filebase)
#        gp = gamessus.GamessParser(job.outputfile)
#        print gp.logname, "electron-electron energy : ", gp.get_ci_ee_energy()
#        os.chdir("..")

def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("logfile",
#                        help = "gamess-us log file")
#    args = parser.parse_args()
#
#    x0 = np.array([-46.71847635, -0.18830162])
#
#    gamess = gamessus.GamessParser(args.logfile)
#
#    twoemo = gamess.read_twoemo()
#    rdm2   = gamess.read_rdm2()
#    occ    = gamess.read_occupations()
#
#    nbf = gamess.get_number_of_mos()  
#    for i in xrange(nbf):
#        print "{0:5d} {1:24.14f} {2:24.14f} {3:24.14f}".format(i+1, rdm2[ijkl(i,i,i,i)], occ[i], occ[i]**2-occ[i])
#===============================

#    x0 = np.array([0.0034479770, -0.2030708415])
#    x1 = np.array([1.24272955, -0.65560724])
#    run_postci_pes(x1)

#================================
# example of post-ci optimization
#================================

    testset =[
        "/home/lmentel/work/jobs/dmft/LiH_cc-pvtz_3.0/LiH_cc-pvtz_3.0_NO.log",
        "/home/lmentel/work/jobs/dmft/LiH_cc-pvtz_5.0/LiH_cc-pvtz_5.0_NO.log",
        "/home/lmentel/work/jobs/dmft/LiH_cc-pvtz_9.0/LiH_cc-pvtz_9.0_NO.log",
        "/home/lmentel/work/jobs/dmft/BeHp_cc-pvtz_2.5/BeHp_cc-pvtz_2.5_NO.log",
        "/home/lmentel/work/jobs/dmft/BeHp_cc-pvtz_4.0/BeHp_cc-pvtz_4.0_NO.log",
        "/home/lmentel/work/jobs/dmft/BeHp_cc-pvtz_8.0/BeHp_cc-pvtz_8.0_NO.log",
        "/home/lmentel/work/jobs/dmft/Li2_cc-pvtz_5.05/Li2_cc-pvtz_5.05_NO.log",
        "/home/lmentel/work/jobs/dmft/Li2_cc-pvtz_8.0/Li2_cc-pvtz_8.0_NO.log",
        "/home/lmentel/work/jobs/dmft/Li2_cc-pvtz_12.0/Li2_cc-pvtz_12.0_NO.log"
        ]
#
    x0 = np.array([1.4])
    energies = get_fci_energies(testset)
    for e in energies:
        print e

#    get_error(x0, energies)

    res = minimize(get_error, x0, args=(energies,), method='BFGS', jac=False)
    print res.x

if __name__ == "__main__":
    main()
