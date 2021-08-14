import numpy as np
from abc import ABCMeta, abstractmethod


class Functional(object, metaclass=ABCMeta):
    """
    Class for storing and manipulating the 1RDM functional objects
    """

    def __init__(self, name=None):
        """
        Initialize the class
        """

        self.name = name

    @abstractmethod
    def jkl_rdms(self):
        """
        Get the J, K, L 1RDMS
        """

        raise NotImplementedError("Should be available soon")

    def jkl_energy(self):
        """
        Calculate three separate energy components corresponding to J, K and L
        terms.
        """

        return np.sum(self.Jrdm), np.sum(self.Krdm), np.sum(self.Lrdm)

    def total_energy(self):
        """
        Calculate the electron-electron potential energy corresponding to
        the functional.
        """

        return np.sum(self.Jrdm) + np.sum(self.Krdm) + np.sum(self.Lrdm)

    def J_energy(self):
        """
        Calculate the J-term contribution to the electron-electron potential
        energy corresponding to the functional.
        """

        return np.sum(self.Jrdm)

    def K_energy(self):
        """
        Calculate the K-term contribution to the electron-electron potential
        energy corresponding to the functional.
        """

        return np.sum(self.Krdm)

    def L_energy(self):
        """
        Calculate the L-term contribution to the electron-electron potential
        energy corresponding to the functional.
        """

        return np.sum(self.Lrdm)

    def decompose(self, mat, homo):

        if mat == "J":
            ematrix = self.Jrdm
        elif mat == "K":
            ematrix = self.Krdm
        elif mat == "L":
            ematrix = self.Lrdm
        diagonal = np.trace(ematrix)
        i_diagonal = np.trace(ematrix[:homo, :homo])
        o_diagonal = np.trace(ematrix[homo:, homo:])
        i_offdiagonal = np.sum(ematrix[:homo, :homo]) - np.trace(ematrix[:homo, :homo])
        o_offdiagonal = np.sum(ematrix[homo:, homo:]) - np.trace(ematrix[homo:, homo:])
        io = 2.0 * np.sum(ematrix[homo:, :homo])
        total = np.sum(ematrix)
        return {
            "inner diagonal": i_diagonal,
            "inner offdiagonal": i_offdiagonal,
            "outer diagonal": o_diagonal,
            "outer offdiagonal": o_offdiagonal,
            "inner-outer": io,
            "total": total,
        }

    def print_components(self, comps_dict):
        for key in sorted(comps_dict.keys()):
            print(("{0:<25s} : {1:>15.10f}".format(key, comps_dict[key])))


class BB(Functional):
    def jkl_rdms(self, occ=None, Jints=None, Kints=None, **kwargs):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(nbf):
            ni = occ[i]
            for j in range(nbf):
                nj = occ[j]
                Jrdm[i, j] = ni * nj * Jints[i, j]
                Krdm[i, j] = -np.sqrt(ni * nj) * Kints[i, j]

        self.Jrdm = 0.5 * Jrdm
        self.Krdm = 0.5 * Krdm
        self.Lrdm = Lrdm


class PNOF4(Functional):
    """
    Object representing PNOF4 functional,
    as presented in M. Piris et al. JCP 133, 111101 (2010)
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        Ne = np.sum(occ[homo + 1 :])

        for i in range(nbf):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i <= homo and j <= homo:
                    Jrdm[i, j] = 0.5 * (ni * nj - (2.0 - ni) * (2.0 - nj)) * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = (
                        -0.25 * (ni * nj - (2.0 - ni) * (2.0 - nj)) * Kints[i, j]
                    )
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5 * np.sqrt((2.0 - ni) * (2.0 - nj)) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo:
                    Jrdm[i, j] = (
                        0.5
                        * (ni * nj - ((2.0 - Ne) / Ne) * ni * (2.0 - nj))
                        * Jints[i, j]
                    )
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = (
                        -0.25
                        * (ni * nj - ((2.0 - Ne) / Ne) * ni * (2.0 - nj))
                        * Kints[i, j]
                    )
                    Krdm[j, i] = Krdm[i, j]
                    t = (2.0 - nj) * ni / Ne
                    Lrdm[i, j] = -0.5 * np.sqrt(t * (nj - ni + t)) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class DD(Functional):
    """
    Object representing DMFT functional derived from CIDD ansatz,
    see E. J. Baerends, "Guidelines for DMFT... (05.02.2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        Ne = np.sum(occ[homo + 1 :])

        for i in range(nbf):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i <= homo and j <= homo:
                    Jrdm[i, j] = (ni + nj - 2.0) * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.5 * (ni + nj - 2.0) * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = 0.5 * np.sqrt((2.0 - ni) * (2.0 - nj)) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif i > homo and j <= homo:
                    Jrdm[j, i] = 0.5 * ni * (2.0 - (2.0 - nj) / Ne) * Jints[j, i]
                    Jrdm[i, j] = Jrdm[j, i]
                    Krdm[j, i] = -0.25 * ni * (2.0 - (2.0 - nj) / Ne) * Kints[j, i]
                    Krdm[i, j] = Krdm[j, i]
                    Lrdm[j, i] = (
                        -np.sqrt((1.0 - Ne) / Ne)
                        * np.sqrt(ni * (2.0 - nj))
                        * Kints[j, i]
                    )
                    Lrdm[i, j] = Lrdm[j, i]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class ELS1(Functional):
    """
    Object representing ELS1 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):
        """
        Calculate the energy matrix from the ELS1 functional.
        """

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class ELS1mod(Functional):
    """
    Object representing ELS1 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):
        """
        Calculate the energy matrix from the ELS1 functional.
        """

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    # Lrdm[i, j] = -0.5*np.sqrt((2.0-ni)*(2.0-nj))*Kints[i, j]
                    # Lrdm[j, i] = Lrdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class ELS2(Functional):
    """
    Object representing ELS2 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None):
        """
        Calculate the energy matrix from the ELS2 functional.
        """

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if (
                    i < homo
                    and j < homo
                    or (j != homo or i <= homo)
                    and (j <= homo or i <= homo)
                ):
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo:
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class ELSlinear(Functional):
    """
    Object representing ELS2 DMFT functional,
    see L. Mentel et al. "Consistent extension..." (2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None, a=None):
        """
        Calculate the energy matrix from the ELS2 functional.
        """

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                else:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -a * 0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5 * (1.0 - a) * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm


class ELSPade(Functional):
    """
    Object representing ELS DMFT functional that contains a parametrization in
    terms of a rational Pade approximant,
    see L. Mentel et al. "Consistent extension..." (2014)"
    """

    def jkl_rdms(self, occ=None, Jints=None, Kints=None, homo=None, a=None, b=None):
        """
        Calculate the energy matrix from the ELS2 functional.
        """

        # if not provide use old AC3 parameters
        if not a:
            a = 16.8212438522  # AC3 a1
        if not b:
            b = 2.61112559818  # AC3 b1
        # a = 19.569076362            # AC3 a2
        # b =  1.41823370195          # AC3 b2

        nbf = occ.size
        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(occ.size):
            ni = occ[i]
            Jrdm[i, i] = 0.5 * ni * Jints[i, i]
            for j in range(i):
                nj = occ[j]
                if i < homo and j < homo:
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                elif j == homo and i > homo:
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j > homo and i > homo:
                    Lrdm[i, j] = 0.5 * np.sqrt(ni * nj) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]
                elif j < homo and i > homo:
                    pade = self.pade_ac3(a, b, 0.5 * ni - 0.5)
                    Jrdm[i, j] = 0.5 * ni * nj * Jints[i, j]
                    Jrdm[j, i] = Jrdm[i, j]
                    Krdm[i, j] = -0.25 * ni * nj * pade * Kints[i, j]
                    Krdm[j, i] = Krdm[i, j]
                    Lrdm[i, j] = -0.5 * np.sqrt(ni * nj) * (1 - pade) * Kints[i, j]
                    Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

    def pade_ac3(a, b, arg):
        """
        Pade approximant used in the intrpolation scheme.
        """

        coeffs = a * np.array([1.0, -4.0 * b / 3, -0.5, b, 0.0])
        p = np.poly1d(coeffs)
        return p(arg) ** 2 / (1.0 + p(arg) ** 2)


class Exact(Functional):
    """
    Exact functional based on the exact 2RDM
    """

    def Coulomb(self, occ=None, Jints=None):
        """
        Calculate the exact coulomb energy and return the matrix already multiplied
        by a factor of 0.5.
        """

        coulomb = np.zeros((occ.size, occ.size), dtype=float)

        for i in range(occ.size):
            for j in range(occ.size):
                coulomb[i, j] = occ[i] * occ[j] * Jints[i, j]

        return 0.5 * coulomb

    def jk_rdms(self, nbf=None, rdm2=None, Jints=None, Kints=None, homo=None):
        """
        Calculate the primitive J and K rdms using exact 2RDM.
        """

        Jrdm = np.zeros((nbf, nbf), dtype=float)
        Krdm = np.zeros((nbf, nbf), dtype=float)
        Lrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(nbf):
            Jrdm[i, i] = 0.5 * rdm2[ijkl(i, i, i, i)] * Jints[i, i]
            for j in range(i):
                Jrdm[i, j] = 0.5 * rdm2[ijkl(i, i, j, j)] * Jints[i, j]
                Jrdm[j, i] = Jrdm[i, j]
                Krdm[i, j] = rdm2[ijkl(i, j, j, i)] * Kints[i, j]
                Krdm[j, i] = Krdm[i, j]
                Lrdm[i, j] = rdm2[ijkl(i, j, i, j)] * Kints[i, j]
                Lrdm[j, i] = Lrdm[i, j]

        self.Jrdm = Jrdm
        self.Krdm = Krdm
        self.Lrdm = Lrdm

    def nonjk_rdm(self, nbf=None, rdm2=None, twoe=None):

        Njklrdm = np.zeros((nbf, nbf), dtype=float)

        for i in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    if k != l:
                        Njklrdm[i, i] += rdm2[ijkl(i, i, k, l)] * twoe[ijkl(i, i, k, l)]
        for i in range(nbf):
            for j in range(nbf):
                if i != j:
                    gam = 0.0
                    exch = 2.0 * rdm2[ijkl(i, j, i, j)] * twoe[ijkl(i, j, i, j)]
                    for k in range(nbf):
                        for l in range(nbf):
                            gam += rdm2[ijkl(i, j, k, l)] * twoe[ijkl(i, j, k, l)]
                    Njklrdm[i, j] = gam - exch

        self.Njklrdm = 0.5 * Njklrdm
        return np.sum(Njklrdm)

    def energy_ee(self, nbf, twoe, rdm2):
        """Print the two-electron integrals."""
        ij = 0
        ee = 0.00
        jrdm = 0.0
        krdm = 0.0
        for i in range(nbf):
            for j in range(i + 1):
                ij += 1
                kl = 0
                for k in range(nbf):
                    for l in range(k + 1):
                        kl += 1
                        if ij >= kl:
                            if i == j and k == l:
                                jrdm += (
                                    factor(i, j, k, l)
                                    * rdm2[ijkl(i, j, k, l)]
                                    * twoe[ijkl(i, j, k, l)]
                                )
                            if i == k and j == l:
                                krdm += (
                                    factor(i, j, k, l)
                                    * rdm2[ijkl(i, j, k, l)]
                                    * twoe[ijkl(i, j, k, l)]
                                )
                            # if abs(rdm2[ijkl(i,j,k,l)]) > 1.0e-10:
                            # print "{0:3d}{1:3d}{2:3d}{3:3d} {4:25.14f}".format(
                            # i, j, k, l, twoe[ijkl(i,j,k,l)])
                            ee += (
                                factor(i, j, k, l)
                                * rdm2[ijkl(i, j, k, l)]
                                * twoe[ijkl(i, j, k, l)]
                            )
        return 0.5 * jrdm, 0.5 * krdm, 0.5 * ee
