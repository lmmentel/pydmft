#!/usr/bin/env python2.7

import argparse
import numpy as np
import matplotlib.pyplot as mplot 
import os
import re
import sys
from scipy.optimize import minimize
from subprocess import Popen

# fortran interfaces fof reading files 
import onelectron
import twoelectron


def get_info(lines, rege):
    '''Get the infromation form contents based on the regular expression.'''

    cp = re.compile(rege)

    for line in lines:
        match = cp.search(line)
        if match:
            return int(match.group(1))

def get_info_float(lines, rege):
    '''Get the infromation form contents based on the regular expression.'''

    cp = re.compile(rege)
    energy = 1000
    for line in lines:
        match = cp.search(line)
        if match:
            energy = float(match.group(1))
    return energy

def get_input_data(logfile, functional, a1, b1, print_level=1):
    '''Retrieve the available information form gamess-us logfile and put
    all the DMFT input keywords in a dictionary with some default values
    set.'''

    with open(logfile, 'r') as log:
        contents = log.readlines()

    n_aos  = r'NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS =\s*(\d+)'
    charge = r'CHARGE OF MOLECULE\s+=\s*(\d+)'
    nelecs = r'NUMBER OF ELECTRONS\s+=\s*(\d+)'

    input_dict = {
            "title"          : "'"+os.path.splitext(logfile)[0]+"'",
            "print_level"    : print_level,
            "nuclear_charge" : float(get_info(contents, charge)+get_info(contents, nelecs)),
            "total_charge"   : float(get_info(contents, charge)),
            "nbasis"         : get_info(contents, n_aos),
            "functional"     : functional,
            "restart"        : 0,
            "loadNOs"        : 1,
            "loadIntegrals"  : 0,
            "a1"             : a1,
            "b1"             : b1,
            "a2"             : 0.0,
            "b2"             : 0.0,
            "analyze"        : ".false.",
            "dictfile"       : "'"+os.path.splitext(logfile)[0]+".F10'",
            "twointfile"     : "'"+os.path.splitext(logfile)[0]+".F09'",
            "exportNOs"      : ".false.",
            "exportfile"     : "'"+os.path.splitext(logfile)[0]+"_dmft_NOs.vec'",
            }
    return input_dict


class Gamess(object):
    '''Simple class for holding gamess filenames for a given job.'''
    def __init__(self, filename=""):
        if filename:
            self.filebase   = os.path.splitext(filename)[0] 
            self.inputfile  = self.filebase + ".inp"
            self.outputfile = self.filebase + ".log"
            self.datfile    = self.filebase + ".dat"
            self.twoeaofile = self.filebase + ".F08"
            self.twoemofile = self.filebase + ".F09"
            self.dictionary = self.filebase + ".F10"
            self.rdm2file   = self.filebase + ".F15"
            self.aos        = self.get_number_of_aos()
            self.mos        = self.get_number_of_mos()
            self.homo       = self.get_homo()
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

    def get_homo(self):
        '''Get the orbital index of homo orbital (indexing starts from zero).'''
    
        with open(self.outputfile, 'r') as f:
            contents = f.read()

        nelecs = re.compile(r'NUMBER OF ELECTRONS\s+=\s*(\d+)')
    
        matchnelecs = nelecs.search(contents)
        if matchnelecs:
            if int(matchnelecs.group(1)) % 2 == 0:
                return int(matchnelecs.group(1))/2 - 1 
            else:
                sys.exit("open shell handling not implemented")
 
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
        onelectron.dictionary.readreals(self.dictionary, occ, occ.size, 21)
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

def get_elsl_matrix(nocc, twoeno, homo):
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

def write_dmft_input(glog, dmftinput, functional=9, a1=0.0, b1=0.0):
    '''Get system information from gamess log and write the dmft 
       input file.''' 
    input_dict = get_input_data(glog, functional, a1, b1) 
    inp = open(dmftinput, 'w')
    inp.write("&input\n")
    for key in sorted(input_dict.iterkeys()):
        inp.write("\t{0:s}={1:s}\n".format(key, str(input_dict[key])))
    inp.write("/\n")
    inp.close()

def splitlist(l, n): 
    if len(l) % n == 0:
        splits = len(l)/n
    elif len(l) % n != 0 and len(l) > n:
        splits = len(l)/n+1
    else:
        splits = 1 

    for i in xrange(splits):
        yield l[n*i:n*i+n]

def get_error_scf(x0, *args):
    
    executable = "/home/lmentel/Source/dmft/dmft/dmft_code/dmft.x"
    #renergy = r'\s*Electron Interaction Energy\s*(\-?\d+\.\d+)'
    renergy = r'\s*Total Energy\s*(\-?\d+\.\d+)'
    jobs = args[0]
    diffs = []
    for jobbatch in splitlist(jobs, 9):
        processes = []
        for job in jobbatch:
            filepath = job.keys()[0]
            energy   = job[filepath]
            dir, file = os.path.split(filepath)
            os.chdir(dir)
            dmftinput = os.path.splitext(file)[0]+"_dmft_fun9.inp"
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
        dmftoutput = os.path.splitext(file)[0]+"_dmft_fun9.out"
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


def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("logfile",
#                        help = "gamess-us log file")
#    args = parser.parse_args()

    a = 16.8212438522e00
    b = 2.61112559818e-04

#    x0 = np.array([-0.25, -0.18])
#    energies = get_fci()
    ee_energies = [
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_3.0/LiH_cc-pvtz_LiH_cc-pvtz_3.0_NO.log"    : 3.3893828641},
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_5.0/LiH_cc-pvtz_LiH_cc-pvtz_5.0_NO.log"    : 2.8860889858},
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_9.0/LiH_cc-pvtz_LiH_cc-pvtz_9.0_NO.log"    : 2.5753180377},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_2.5/BeHp_cc-pvtz_BeHp_cc-pvtz_2.5_NO.log" : 4.6469022574},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_4.0/BeHp_cc-pvtz_BeHp_cc-pvtz_4.0_NO.log" : 4.1448090544},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_8.0/BeHp_cc-pvtz_BeHp_cc-pvtz_8.0_NO.log" : 3.6801405114},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_5.05/Li2_cc-pvtz_Li2_cc-pvtz_5.05_NO.log"  : 6.3775221692},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_8.0/Li2_cc-pvtz_Li2_cc-pvtz_8.0_NO.log"    : 5.6241679747},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_12.0/Li2_cc-pvtz_Li2_cc-pvtz_12.0_NO.log"  : 5.2363070696}
    ]
    total_energies = [
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_3.0/LiH_cc-pvtz_LiH_cc-pvtz_3.0_NO.log"    : -8.0400255128},
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_5.0/LiH_cc-pvtz_LiH_cc-pvtz_5.0_NO.log"    : -7.9935680803},
    {"/home/lmentel/jobs/dmft/test/LiH_cc-pvtz_9.0/LiH_cc-pvtz_LiH_cc-pvtz_9.0_NO.log"    : -7.9500671056},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_2.5/BeHp_cc-pvtz_BeHp_cc-pvtz_2.5_NO.log" : -14.9078221449},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_4.0/BeHp_cc-pvtz_BeHp_cc-pvtz_4.0_NO.log" : -14.8421162935},
    {"/home/lmentel/jobs/dmft/test/BeHp_cc-pvtz_8.0/BeHp_cc-pvtz_BeHp_cc-pvtz_8.0_NO.log" : -14.7922544630},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_5.05/Li2_cc-pvtz_Li2_cc-pvtz_5.05_NO.log"  : -14.9371059764},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_8.0/Li2_cc-pvtz_Li2_cc-pvtz_8.0_NO.log"    : -14.9128113456},
    {"/home/lmentel/jobs/dmft/test/Li2_cc-pvtz_12.0/Li2_cc-pvtz_Li2_cc-pvtz_12.0_NO.log"  : -14.8994866925}
    ]
    for e in total_energies:
        print e
    
#    print get_error_scf(x0, energies)
    #x0 = np.array([0.50])
    # functional 9 parameters optimized with bfgs
    x0 = np.array([5.3133906858, 0.2362376247])
    res = minimize(get_error_scf, x0, args=(total_energies,), method='Nelder-Mead')
    print res.x

if __name__ == "__main__":
    main()