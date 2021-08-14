from abc import ABCMeta, abstractmethod
from matplotlib import rc
from operator import itemgetter
from scipy.interpolate import interp1d
from subprocess import Popen
import docopt
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

from chemtools.gamessus import GamessReader, GamessLogParser


def factor(i, j, k, l):
    """Based on the orbitals indices return the factor that takes into account
    the index permutational symmetry."""
    if i == j and k == l and i == k:
        return 1.0
    elif i == j and k == l:
        return 2.0
    elif (
        (i == k and j == l)
        or (i == j and i == k)
        or (j == k and j == l)
        or (i == j or k == l)
    ):
        return 4.0
    else:
        return 8.0


def ijkl(i, j, k, l):
    """Based on the four orbital indices i,j,k,l return the address
    in the 1d vector."""
    ij = max(i, j) * (max(i, j) + 1) / 2 + min(i, j)
    kl = max(k, l) * (max(k, l) + 1) / 2 + min(k, l)
    return max(ij, kl) * (max(ij, kl) + 1) / 2 + min(ij, kl)


def contains(theString, theQuery):
    return theString.find(theQuery) > -1


def get_fci():

    filenames = []
    for (path, dirs, files) in os.walk(os.getcwd()):
        for fileItem in files:
            if contains(fileItem, "NO.log"):
                filenames.append(os.path.join(path, fileItem))
    energies = []
    for file in filenames:
        dir, log = os.path.split(file)
        os.chdir(dir)
        gamess = Gamess(log)
        print((log, gamess.get_number_of_aos(), gamess.homo))
        twoemo = gamess.get_motwoe()
        rdm2 = gamess.get_rdm2()
        nbf = gamess.mos
        exact_j, exact_k = get_exact_jk(rdm2, twoemo, nbf)
        exact_njk = get_exact_nonjk(rdm2, twoemo, nbf)
        dd = decompose(exact_j + exact_k + exact_njk, gamess.homo)
        energies.append({file: dd["inner-outer"]})
        print(("\n{1}\nExact for {0}\n{1}\n".format(log, "=" * (len(log) + 10))))
        print_components(dd)
    return energies


def get_error(x0, *args):

    jobs = args[0]
    diffs = []
    for job in jobs:
        filepath = list(job.keys())[0]
        energy = job[filepath]
        dir, file = os.path.split(filepath)
        os.chdir(dir)
        gamess = Gamess(file)
        twoeno = gamess.get_motwoe()
        nbf = gamess.mos
        nocc = gamess.get_occupations()
        els = get_else_matrix(nocc, twoeno, gamess.homo, x0[0], x0[1])
        dd = decompose(els, gamess.homo)
        diffs.append(energy - dd["inner-outer"])
        print(("\n{1}\nELS for {0}\n{1}\n".format(file, "=" * (len(file) + 8))))
        print_components(dd)
    diffs = np.asarray(diffs)
    error = np.sqrt(np.add.reduce(diffs * diffs))
    print(
        (
            "Error = {e:>14.10f}  Parameters: {p:s}".format(
                e=error, p="  ".join("{:>14.10f}".format(x) for x in x0)
            )
        )
    )

    print(("-" * 106))
    return error


def splitlist(l, n):
    """
    Split a list 'l' into lists of at most 'n' elements.
    """

    if len(l) % n == 0:
        splits = len(l) / n
    elif len(l) > n:
        splits = len(l) / n + 1
    else:
        splits = 1

    for i in range(splits):
        yield l[n * i : n * i + n]


def get_error_scf(x0, *args):

    executable = "/home/lmentel/Source/dmft/dmft_code/dmft.x"
    renergy = r"\s*Electron Interaction Energy\s*(\-?\d+\.\d+)"
    jobs = args[0]
    diffs = []
    for jobbatch in splitlist(jobs, 6):
        processes = []
        for job in jobbatch:
            filepath = list(job.keys())[0]
            energy = job[filepath]
            dir, file = os.path.split(filepath)
            os.chdir(dir)
            dmftinput = os.path.splitext(file)[0] + "_dmft.inp"
            write_dmft_input(file, dmftinput, functional=9, a1=x0[0], b1=x0[1])
            with open(os.path.splitext(dmftinput)[0] + ".out", "w") as out:
                p = Popen([executable, dmftinput], stdout=out, stderr=out)
            processes.append(p)

        for p in processes:
            p.wait()

    for job in jobs:
        filepath = list(job.keys())[0]
        energy = job[filepath]
        dir, file = os.path.split(filepath)
        os.chdir(dir)
        dmftoutput = os.path.splitext(file)[0] + "_dmft.out"
        with open(dmftoutput, "r") as f:
            contents = f.readlines()
        dmft_energy = get_info_float(contents, renergy)
        diffs.append(energy - dmft_energy)
        print(
            (
                "Processed {0}   Exact energy: {1:15.10f}   DMFT energy: {2:15.10f}".format(
                    file, energy, dmft_energy
                )
            )
        )
    diffs = np.asarray(diffs)
    error = np.sqrt(np.add.reduce(diffs * diffs))
    print(
        (
            "Error = {e:>14.10f}  Parameters: {p:s}".format(
                e=error, p="  ".join("{:>14.10f}".format(x) for x in x0)
            )
        )
    )

    print(("-" * 106))
    return error


def get_jkints(twoe, nb):
    """
    get the coulomb and exchange integrals as two index quantities.
    """

    Jints = np.zeros((nb, nb), dtype=float)
    Kints = np.zeros((nb, nb), dtype=float)

    for i in range(nb):
        for j in range(nb):
            Jints[i, j] = twoe[ijkl(i, i, j, j)]
            Kints[i, j] = twoe[ijkl(i, j, j, i)]

    return Jints, Kints


def print_energies(fun_name, energies, components=False):

    print("\n{0:^118s}\n{1:^118s}\n{0:^118s}\n".format("=" * len(fun_name), fun_name))

    print(
        "{0:<30s}  {1:^6s}{2:^20s}{3:^20s}{4:^20s}{5:^20s}\n{6:s}".format(
            "File", "R", "J energy", "K energy", "L energy", "Total E_ee", "=" * 118
        )
    )
    for row in sorted(energies, key=itemgetter("dist")):
        print(
            "{0:<30s}  {1:6.2f}{2:20.10f}{3:20.10f}{4:20.10f}{5:20.10f}".format(
                row["file"], row["dist"], row["j"], row["k"], row["l"], row["total"]
            )
        )
    print()
    if components:
        for row in sorted(energies, key=itemgetter("dist")):
            print(
                "{0:<30s}  {1:6.2f}{2:20.10f}{3:20.10f}{4:20.10f}{5:20.10f}{6:20.10f}".format(
                    row["file"],
                    row["dist"],
                    row["jcomps"]["inner diagonal"],
                    row["jcomps"]["inner offdiagonal"],
                    row["jcomps"]["outer diagonal"],
                    row["jcomps"]["outer offdiagonal"],
                    row["jcomps"]["inner-outer"],
                )
            )


def lplot(casesd, functs=[], save=False):

    x, te, onee, twoe, nucr, hf = np.loadtxt(
        casesd["tablefile"],
        dtype=float,
        comments="#",
        usecols=(0, 1, 2, 3, 4, 5),
        unpack=True,
    )

    te_s = interp1d(x, te, kind="cubic")
    hf_s = interp1d(x, hf, kind="cubic")
    xnew = np.linspace(min(x), max(x), 100)

    rc("font", size=18.0)
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)
    rc("figure", autolayout=True)

    plt.figure(figsize=(12, 8))

    plt.title(casesd["name"])
    plt.xlabel(r"Internuclear distance $R$ [bohr]")
    plt.ylabel(r"Energy [hartree]")
    plt.plot(xnew, te_s(xnew), "-", linewidth="2.0", label="Exact")
    plt.plot(xnew, hf_s(xnew), "-", label="HF")
    for funct in functs:
        for name, energies in list(funct.items()):
            fun_s = interp1d(x, onee + nucr + energies, kind="cubic")
            plt.plot(xnew, fun_s(xnew), label=name)

    plt.legend(loc="best", frameon=False)

    if save:
        figname = casesd["name"] + ".pdf"
        plt.savefig(figname)
    else:
        plt.show()


def get_exact_jk():

    data = []

    exact = Exact("Exact")

    for (path, dirs, files) in os.walk(cases[args["<molecule>"]]["workdir"]):
        for fileitem in files:
            if "_NO.log" in fileitem:
                log = os.path.join(path, fileitem)
                gp = GamessParser(log)
                gr = GamessReader(log)

                twoemo = gr.read_twoemo()
                rdm2 = gr.read_rdm2()
                occ = np.abs(gr.read_occupations())
                Jints, Kints = get_jkints(twoemo, gp.get_number_of_mos())

                exact.jk_rdms(
                    occ.size, rdm2=rdm2, Jints=Jints, Kints=Kints, homo=gp.get_homo()
                )
                njk = exact.nonjk_rdm(occ.size, rdm2=rdm2, twoe=twoemo)
                # j, k, tot = exact.energy_ee(occ.size, twoemo, rdm2)
                data.append(
                    {
                        "file": fileitem,
                        "dist": float(match.group(1)),
                        "j": exact.J_energy(),
                        "k": exact.K_energy(),
                        "l": 0.0,
                        "total": njk,
                    }
                )

    print_energies("Exact JK", data)


def main():
    """
    Usage:
        postcidmft.py <molecule>

    Options:
        molecule  molecule to work with
    """

    workdir = "/home/lmentel/work/jobs/dmft"
    cases = {
        "lih": {
            "name": r"LiH",
            "workdir": os.path.join(workdir, "LiH"),
            "tablefile": os.path.join(workdir, "LiH.tbl"),
        },
        "li2": {
            "name": r"Li$_2$",
            "workdir": os.path.join(workdir, "Li2"),
            "tablefile": os.path.join(workdir, "Li2.tbl"),
        },
        "behp": {
            "name": r"BeH$^+$",
            "workdir": os.path.join(workdir, "BeHp"),
            "tablefile": os.path.join(workdir, "BeHp.tbl"),
        },
    }

    args = docopt.docopt(main.__doc__, help=True)
    if args["<molecule>"] not in cases:
        sys.exit(
            "wrong molecule: {0:s}, should be one of (lih, behp, li2)".format(
                args["<molecule>"]
            )
        )

    functionals = [
        ELS2("ELS2"),
        PNOF4("PNOF4"),
        ELS1mod("ELS1mod"),
        ELS1("ELS1"),
        DD("DD"),
    ]

    data = {f.name: [] for f in functionals}
    dist_re = re.compile(r".*_(\d+\.\d+)_.*")

    for (path, dirs, files) in os.walk(cases[args["<molecule>"]]["workdir"]):
        for fileitem in files:
            if "_NO.log" in fileitem:
                log = os.path.join(path, fileitem)
                gp = GamessParser(log)
                gr = GamessReader(log)

                twoemo = gr.read_twoemo()
                rdm2 = gr.read_rdm2()
                occ = np.abs(gr.read_occupations())
                Jints, Kints = get_jkints(twoemo, gp.get_number_of_mos())

                match = dist_re.search(fileitem)
                for f in functionals:
                    f.jkl_rdms(occ=occ, Jints=Jints, Kints=Kints, homo=gp.get_homo())

                for f in functionals:
                    data[f.name].append(
                        {
                            "file": fileitem,
                            "dist": float(match.group(1)),
                            "j": f.J_energy(),
                            "k": f.K_energy(),
                            "l": f.L_energy(),
                            "total": f.total_energy(),
                            "jcomps": f.decompose("J", gp.get_homo()),
                        }
                    )

    datap = []
    for funct, values in list(data.items()):
        dde = [x["total"] for x in sorted(values, key=itemgetter("dist"))]
        datap.append({funct: dde})
        print_energies(funct, values, components=False)

    lplot(cases[args["<molecule>"]], datap, save=False)


if __name__ == "__main__":
    main()
