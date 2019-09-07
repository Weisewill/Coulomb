# Calculate the f-f and f-d two-body Coulomb interaction matrix elements.

from __future__ import print_function
from itertools import product
from scipy.misc import factorial as fact
from scipy.linalg import block_diag
import numpy as np
#import h5py
from sympy.physics.wigner import wigner_6j
from sympy.physics.wigner import wigner_3j

def U_matrix(mode, l, radial_integrals=None, U_int=None, J_hund=None, T=None):
    if 'slater' in mode:
        U_matrix = U_matrix_slater(l, radial_integrals, U_int, J_hund)
    elif 'kanamori' in mode:
        U_matrix = U_matrix_kanamori(2*l+1, U_int, J_hund)
    else:
        raise NameError(" unsupported mode!")
 #   u_avg, j_avg = get_average_uj(U_matrix)

    # add spin-components
    norb = U_matrix.shape[0]
    norb2 = norb*2
    Ufull_matrix = np.zeros((norb2,norb2,norb2,norb2), dtype=np.complex)
    if T is not None:
        # spin block
        Ufull_matrix[:norb,:norb,:norb,:norb] = U_matrix
        Ufull_matrix[norb:,norb:,norb:,norb:] = U_matrix
        Ufull_matrix[:norb,:norb,norb:,norb:] = U_matrix
        Ufull_matrix[norb:,norb:,:norb,:norb] = U_matrix
        print(" u-matrix: nnz in compl_sph_harm = {}".format(
                np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
        Ufull_matrix = unitary_transform_coulomb_matrix(Ufull_matrix, T)
    else: # spin index fast
        Ufull_matrix[::2,::2,::2,::2] = U_matrix  # up, up
        Ufull_matrix[1::2,1::2,1::2,1::2] = U_matrix # dn, dn
        Ufull_matrix[::2,::2,1::2,1::2] = U_matrix # up, dn
        Ufull_matrix[1::2,1::2,::2,::2] = U_matrix # dn, up

    print(" u-matrix: nnz in final basis = {}".format(
            np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
    return Ufull_matrix

# The interaction matrix in desired basis.
# U^{spherical}_{m1 m4 m2 m3} =
# \sum_{k=0}^{2l} F_k angular_matrix_element(l, k, m1, m2, m3, m4)
# H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
# a_{i \sigma}^\dagger a_{j \sigma'}^\dagger a_{l \sigma'} a_{k \sigma}.
def U_matrix_slater(l, radial_integrals=None, U_int=None, J_hund=None):
    r"""
    Calculate the full four-index U matrix being given either
    radial_integrals or U_int and J_hund.
    The convetion for the U matrix is that used to construct
    the Hamiltonians, namely:
    .. math:: H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
            a_{i \sigma}^\dagger a_{j \sigma'}^\dagger
            a_{l \sigma'} a_{k \sigma}.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    radial_integrals : list, optional
                       Slater integrals [F0,F2,F4,..].
                       Must be provided if U_int and J_hund are not given.
                       Preferentially used to compute the U_matrix
                       if provided alongside U_int and J_hund.
    U_int : scalar, optional
            Value of the screened Hubbard interaction.
            Must be provided if radial_integrals are not given.
    J_hund : scalar, optional
             Value of the Hund's coupling.
             Must be provided if radial_integrals are not given.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """

    # Check all necessary information is present and consistent
    if radial_integrals is None and (U_int is None and J_hund is None):
        raise ValueError("U_matrix: provide either the radial_integrals" +
                " or U_int and J_hund.")
    if radial_integrals is None and (U_int is not None and J_hund is not None):
        radial_integrals = U_J_to_radial_integrals(l, U_int, J_hund)
    if radial_integrals is not None and \
            (U_int is not None and J_hund is not None):
        if len(radial_integrals)-1 != l:
            raise ValueError("U_matrix: inconsistency in l" +
                    " and number of radial_integrals provided.")
        if not np.allclose(radial_integrals,
                U_J_to_radial_integrals(l, U_int, J_hund)):
            print(" Warning: U_matrix: radial_integrals provided\n"+
            " do not match U_int and J_hund.\n"+
            " Using radial_integrals to calculate U_matrix.")

    # Full interaction matrix
    # Basis of spherical harmonics Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}
    # U^{spherical}_{m1 m4 m2 m3} = \sum_{k=0}^{2l} F_k
    # angular_matrix_element(l, k, m1, m2, m3, m4)
    U_matrix = np.zeros((2*l+1,2*l+1,2*l+1,2*l+1),dtype=np.float)

    m_range = range(-l,l+1)
    for n, F in enumerate(radial_integrals):
        k = 2*n
        for m1, m2, m3, m4 in product(m_range,m_range,m_range,m_range):
            U_matrix[m1+l,m3+l,m2+l,m4+l] += \
                    F * angular_matrix_element(l,l,l,l,k,m1,m2,m3,m4)
    return U_matrix

def U_matrix_ffdd(l_1, l_2, radial_integrals, T_f, T_d):
    r"""
    Calculate the full four-index U matrix being given either
    radial_integrals or U_int and J_hund.
    The convetion for the U matrix is that used to construct
    the Hamiltonians, namely:
    .. math:: H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
            a_{i \sigma}^\dagger a_{j \sigma'}^\dagger
            a_{l \sigma'} a_{k \sigma}.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    radial_integrals : list, optional
                       Slater integrals [F0,F2,F4,..].
                       Must be provided if U_int and J_hund are not given.
                       Preferentially used to compute the U_matrix
                       if provided alongside U_int and J_hund.
    U_int : scalar, optional
            Value of the screened Hubbard interaction.
            Must be provided if radial_integrals are not given.
    J_hund : scalar, optional
             Value of the Hund's coupling.
             Must be provided if radial_integrals are not given.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """

    # Full interaction matrix
    # Basis of spherical harmonics Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}
    # U^{spherical}_{m1 m4 m2 m3} = \sum_{k=0}^{2l} F_k
    # angular_matrix_element(l, k, m1, m2, m3, m4)
    U_matrix = np.zeros((2*l_1+1,2*l_1+1,2*l_2+1,2*l_2+1),dtype=np.float)

    m_range_1 = range(-l_1,l_1+1)
    m_range_2 = range(-l_2,l_2+1)

    for n, F in enumerate(radial_integrals):
        k = 2*n
        for m1, m2, m3, m4 in product(m_range_1,m_range_2,m_range_1,m_range_2):
            U_matrix[m1+l_1,m3+l_1,m2+l_2,m4+l_2] += \
                    F * angular_matrix_element(l_1,l_2,l_1,l_2,k,m1,m2,m3,m4)


    # add spin-components
    norb1 = (2*l_1+1)
    norb2 = (2*l_2+1)
    norb1_full = (2*l_1+1)*2
    norb2_full = (2*l_2+1)*2
    Ufull_matrix = np.zeros((norb1_full,norb1_full,norb2_full,norb2_full),dtype=np.complex)
    if T_f is not None:
        # spin block
        Ufull_matrix[:norb1,:norb1,:norb2,:norb2] = U_matrix
        Ufull_matrix[norb1:,norb1:,norb2:,norb2:] = U_matrix
        Ufull_matrix[:norb1,:norb1,norb2:,norb2:] = U_matrix
        Ufull_matrix[norb1:,norb1:,:norb2,:norb2] = U_matrix
        print(" u-matrix: nnz in compl_sph_harm = {}".format(
                np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
        Ufull_matrix = unitary_transform_coulomb_matrix_fd(Ufull_matrix, T_f, T_d, mode='F')
    else: # spin index fast
        Ufull_matrix[::2,::2,::2,::2] = U_matrix  # up, up
        Ufull_matrix[1::2,1::2,1::2,1::2] = U_matrix # dn, dn
        Ufull_matrix[::2,::2,1::2,1::2] = U_matrix # up, dn
        Ufull_matrix[1::2,1::2,::2,::2] = U_matrix # dn, up

    print(" u-matrix: nnz in final basis = {}".format(
            np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
    return Ufull_matrix

def U_matrix_fddf(l_1, l_2, radial_integrals, T_f, T_d):
    r"""
    Calculate the full four-index U matrix being given either
    radial_integrals or U_int and J_hund.
    The convetion for the U matrix is that used to construct
    the Hamiltonians, namely:
    .. math:: H = \frac{1}{2} \sum_{ijkl,\sigma \sigma'} U_{ikjl}
            a_{i \sigma}^\dagger a_{j \sigma'}^\dagger
            a_{l \sigma'} a_{k \sigma}.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    radial_integrals : list, optional
                       Slater integrals [F0,F2,F4,..].
                       Must be provided if U_int and J_hund are not given.
                       Preferentially used to compute the U_matrix
                       if provided alongside U_int and J_hund.
    U_int : scalar, optional
            Value of the screened Hubbard interaction.
            Must be provided if radial_integrals are not given.
    J_hund : scalar, optional
             Value of the Hund's coupling.
             Must be provided if radial_integrals are not given.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """

    # Full interaction matrix
    # Basis of spherical harmonics Y_{-2}, Y_{-1}, Y_{0}, Y_{1}, Y_{2}
    # U^{spherical}_{m1 m4 m2 m3} = \sum_{k=0}^{2l} F_k
    # angular_matrix_element(l, k, m1, m2, m3, m4)
    U_matrix = np.zeros((2*l_1+1,2*l_2+1,2*l_2+1,2*l_1+1),dtype=np.float)

    m_range_1 = range(-l_1,l_1+1)
    m_range_2 = range(-l_2,l_2+1)

    for n, G in enumerate(radial_integrals):
        k = 2*n + 1
        for m1, m2, m3, m4 in product(m_range_1,m_range_2,m_range_2,m_range_1):
            U_matrix[m1+l_1,m3+l_2,m2+l_2,m4+l_1] += \
                    G * angular_matrix_element(l_1,l_2,l_2,l_1,k,m1,m2,m3,m4)


    # add spin-components
    norb1 = (2*l_1+1)
    norb2 = (2*l_2+1)
    norb1_full = (2*l_1+1)*2
    norb2_full = (2*l_2+1)*2
    Ufull_matrix = np.zeros((norb1_full,norb2_full,norb2_full,norb1_full),dtype=np.complex)
    if T_f is not None:
        # spin block
        Ufull_matrix[:norb1,:norb2,:norb2,:norb1] = U_matrix
        Ufull_matrix[norb1:,norb2:,norb2:,norb1:] = U_matrix
        Ufull_matrix[:norb1,:norb2,norb2:,norb1:] = U_matrix
        Ufull_matrix[norb1:,norb2:,:norb2,:norb1] = U_matrix
        print(" u-matrix: nnz in compl_sph_harm = {}".format(
                np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
        Ufull_matrix = unitary_transform_coulomb_matrix_fd(Ufull_matrix, T_f, T_d, mode='G')
    else: # spin index fast
        Ufull_matrix[::2,::2,::2,::2] = U_matrix  # up, up
        Ufull_matrix[1::2,1::2,1::2,1::2] = U_matrix # dn, dn
        Ufull_matrix[::2,::2,1::2,1::2] = U_matrix # up, dn
        Ufull_matrix[1::2,1::2,::2,::2] = U_matrix # dn, up

    print(" u-matrix: nnz in final basis = {}".format(
            np.count_nonzero(np.abs(Ufull_matrix)>1.e-10)))
    return Ufull_matrix

# Angular matrix elements of particle-particle interaction
# sqrt( (2l_1+1)*(2l_2+1) ) \sum_{q=-k}^{k} (-1)^{m1+m2+q}
#    ((l_1 0)   (k 0) (l_3 0))   * ((l_2 0)   (k 0)  (l_4 0))
#    ((l_1 -m1) (k q) (l_3 m3))  * ((l_2 -m2) (k -q) (l_4 m4))
def angular_matrix_element(l_1, l_2, l_3, l_4, k, m1, m2, m3, m4):
    r"""
    Calculate the angular matrix element
    .. math::
       (2l+1)^2
       \begin{pmatrix}
            l & k & l \\
            0 & 0 & 0
       \end{pmatrix}^2
       \sum_{q=-k}^k (-1)^{m_1+m_2+q}
       \begin{pmatrix}
            l & k & l \\
         -m_1 & q & m_3
       \end{pmatrix}
       \begin{pmatrix}
            l & k  & l \\
         -m_2 & -q & m_4
       \end{pmatrix}.
    Parameters
    ----------
    l : integer
    k : integer
    m1 : integer
    m2 : integer
    m3 : integer
    m4 : integer
    Returns
    -------
    ang_mat_ele : scalar
                  Angular matrix element.
    """
    ang_mat_ele = 0
    for q in range(-k,k+1):
        ang_mat_ele += three_j_symbol((l_1,-m1),(k,q),(l_3,m3))* \
                three_j_symbol((l_2,-m2),(k,-q),(l_4,m4))* \
                (-1.0 if (m1+q+m2) % 2 else 1.0)
    ang_mat_ele *= np.sqrt((2*l_1+1)*(2*l_2+1)*(2*l_3+1)*(2*l_4+1)) \
     * (three_j_symbol((l_1,0),(k,0),(l_3,0))*three_j_symbol((l_2,0),(k,0),(l_4,0)))
    return ang_mat_ele

# Wigner 3-j symbols
# ((j1 m1) (j2 m2) (j3 m3))
def three_j_symbol(jm1, jm2, jm3):
    r"""
    Calculate the three-j symbol
    .. math::
       \begin{pmatrix}
        l_1 & l_2 & l_3\\
        m_1 & m_2 & m_3
       \end{pmatrix}.
    Parameters
    ----------
    jm1 : tuple of integers
          (j_1 m_1)
    jm2 : tuple of integers
          (j_2 m_2)
    jm3 : tuple of integers
          (j_3 m_3)
    Returns
    -------
    three_j_sym : scalar
                  Three-j symbol.
    """
    j1, m1 = jm1
    j2, m2 = jm2
    j3, m3 = jm3

    if (m1+m2+m3 != 0 or
        m1 < -j1 or m1 > j1 or
        m2 < -j2 or m2 > j2 or
        m3 < -j3 or m3 > j3 or
        j3 > j1 + j2 or
        j3 < abs(j1-j2)):
        return .0

    three_j_sym = -1.0 if (j1-j2-m3) % 2 else 1.0
    three_j_sym *= np.sqrt(fact(j1+j2-j3)*fact(j1-j2+j3)* \
            fact(-j1+j2+j3)/fact(j1+j2+j3+1))
    three_j_sym *= np.sqrt(fact(j1-m1)*fact(j1+m1)*fact(j2-m2)* \
            fact(j2+m2)*fact(j3-m3)*fact(j3+m3))

    t_min = max(j2-j3-m1,j1-j3+m2,0)
    t_max = min(j1-m1,j2+m2,j1+j2-j3)

    t_sum = 0
    for t in range(t_min,t_max+1):
        t_sum += (-1.0 if t % 2 else 1.0)/(fact(t)*fact(j3-j2+m1+t)* \
                fact(j3-j1-m2+t)*fact(j1+j2-j3-t)*fact(j1-m1-t)*fact(j2+m2-t))

    three_j_sym *= t_sum
    return three_j_sym

# Convert U,J -> radial integrals F_k
def U_J_to_radial_integrals(l, U_int, J_hund):
    r"""
    Determine the radial integrals F_k from U_int and J_hund.
    Parameters
    ----------
    l : integer
        Angular momentum of shell being treated
        (l=2 for d shell, l=3 for f shell).
    U_int : scalar
            Value of the screened Hubbard interaction.
    J_hund : scalar
             Value of the Hund's coupling.
    Returns
    -------
    radial_integrals : list
                       Slater integrals [F0,F2,F4,..].
    """

    F = np.zeros((l+1),dtype=np.float)
    F[0] = U_int
    if l == 0:
        pass
    elif l == 1:
        F[1] = J_hund * 5.0
    elif l == 2:
        F[1] = J_hund * 14.0 / (1.0 + 0.625)
        F[2] = 0.625 * F[1]
    elif l == 3:
        F[1] = 6435.0 * J_hund / (286.0 + 195.0 * 0.668 + 250.0 * 0.494)
        F[2] = 0.668 * F[1]
        F[3] = 0.494 * F[1]
    else:
        raise ValueError(
            " U_J_to_radial_integrals: implemented only for l=0,1,2,3")
    return F

def unitary_transform_coulomb_matrix(a, u):
    '''Perform a unitary transformation (u) on the Coulomb matrix (a).
    '''
    a_ = np.asarray(a).copy()
    m_range = range(a.shape[0])
    for i,j in product(m_range, m_range):
        a_[i,j,:,:] = u.T.conj().dot(a_[i,j,:,:].dot(u))
    a_ = a_.swapaxes(0,2).swapaxes(1,3)
    for i,j in product(m_range, m_range):
        a_[i,j,:,:] = u.T.conj().dot(a_[i,j,:,:].dot(u))
    return a_

def unitary_transform_coulomb_matrix_fd(a, u_f, u_d, mode):
    '''
    Perform a unitary transformation on the Coulomb matrix (a).
    mode == 'F': transform U_ffdd
    mode == 'G': transform U_fddf
    '''

    m_range_f = range(14)
    m_range_d = range(10)

    if mode == 'F':

        a_ = np.asarray(a).copy()
        for i,j in product(m_range_f, m_range_f):
            a_[i,j,:,:] = u_d.T.conj().dot(a_[i,j,:,:].dot(u_d))
        a_ = a_.swapaxes(0,2).swapaxes(1,3)
        for i,j in product(m_range_d, m_range_d):
            a_[i,j,:,:] = u_f.T.conj().dot(a_[i,j,:,:].dot(u_f))
        a_ = a_.swapaxes(0,2).swapaxes(1,3)
        return a_

    if mode == 'G':

        a_ = np.asarray(a).copy()
        for i,j in product(m_range_f, m_range_d):
            a_[i,j,:,:] = u_d.T.conj().dot(a_[i,j,:,:].dot(u_f))
        a_ = a_.swapaxes(0,2).swapaxes(1,3)
        for i,j in product(m_range_d, m_range_f):
            a_[i,j,:,:] = u_f.T.conj().dot(a_[i,j,:,:].dot(u_d))
        a_ = a_.swapaxes(0,2).swapaxes(1,3)
        return a_

def trans_JJ_to_CH_sup_sdn(L):
    '''
    trafoso provides transformation matrices from
    |L,1/2,mL,mS> (L=0,1,2,3, mS=-1/2,1/2) basis to
    basis |J,L,S,mJ>, J=L-1/2, L+1/2
    J. J. Sakurai 'Modern Quantum Mechanics 2nd ed.'
    eq (3.8.63).
    ordering because of the convention used in WIEN is:
                       mS=1/2        mS=-1/2
                     -L .......L  -L ...... L     (2*(2L+1) columns)
            -(L-1/2)
               .
    J=L-1/2    .
               .
             (L-1/2)
             -L-1/2
               .
    J=L+1/2    .
               .
              L+1/2
    '''
    cf = np.zeros((2 * (2 * L + 1), 2 * (2 * L + 1)))
    if L == 0:
        cf[0, 1] = 1.0
        cf[1, 0] = 1.0
    else:
        k1 = -1
        for ms in range(-1, 2, 2):
            ams = -ms / 2.
            for ml in range(-L, L + 1):
                k1 = k1 + 1
                k2 = -1
                for mj in range(-2 * L + 1, 2 * L, 2):  # L-1/2 states
                    amj = mj / 2.
                    k2 = k2 + 1
                    d = amj - ml - ams
                    if abs(d) < 0.0001:
                        if ms == 1:
                            cf[k2, k1] =   np.sqrt((L + 0.5 + amj) / (2 * L + 1))
                        else:
                            cf[k2, k1] = - np.sqrt((L + 0.5 - amj) / (2 * L + 1))
            

                for mj in range(-2 * L - 1, 2 * L + 2, 2):  # L+1/2 states
                    amj = mj / 2.
                    k2 = k2 + 1
                    d = amj - ml - ams
                    if abs(d) < 0.0001:
                        if ms == 1:
                            cf[k2, k1] = np.sqrt((L + 0.5 - amj) / (2 * L + 1))
                        else:
                            cf[k2, k1] = np.sqrt((L + 0.5 + amj) / (2 * L + 1))
    return cf


def test():
# Test
    l_1 = 3
    l_2 = 2
    m_range_1 = range(2*l_1+1)
    m_range_2 = range(2*l_2+1)

    m_range_full = range( (2*l_1+1)*2 )

    F=np.zeros(3)
    F[0] = 1; F[1] = 0.5; F[2] = 0.2;
    G=np.zeros(3)
    G[0] = 1; G[1] = 0.5; G[2] = 0.2;
    U_int=4.5
    J_hund=0.5

    ryd2ev = 13.605698065894
    with h5py.File('GPARAM.h5') as f:
        imp=1
        utrans=f['/IMPURITY_{}/DB_TO_SAB'.format(imp)][...].T
        print(utrans)

        v2e = f['IMPURITY_1/V2E'][...].T


        utrans = np.zeros((14,14))
        for i in range(14):
            utrans[i][i] = 1.0

        iso = f['iso'][0]
        if (iso == 2):
            u_cmplx_harm_to_rel_harm = trans_JJ_to_CH_sup_sdn(l_1).T
            utrans = u_cmplx_harm_to_rel_harm.dot(utrans)

        print(utrans)

    #l_list = [3]
    #_, utrans = get_JU_relat_sph_harm_cg(l_list)

    coulomb_matrix = U_matrix(mode="slater",l=l_1,U_int=U_int,J_hund=J_hund,T=utrans)
    coulomb_matrix = coulomb_matrix/ryd2ev

    with open('coulomb_matrix_ffff_full.txt','w') as f:
        for m1,m2,m3,m4 in product(m_range_full,m_range_full,m_range_full,m_range_full):
            if np.abs(coulomb_matrix[m1,m2,m3,m4]) < 1.e-8:
                continue
            f.write('{} {} {} {}  {}\n'.format(m1+1,m2+1,m3+1,m4+1, \
                    coulomb_matrix[m1,m2,m3,m4]))


    diff = mat_diff(coulomb_matrix,v2e)
    print(diff)


def mat_diff(mat1,mat2):
    diff = 0.0
    n_range = range(14)
    for i, j, k, l in product(n_range,n_range,n_range,n_range):
        diff = np.sqrt((mat1[i][j][k][l] - mat2[i][j][k][l])**2)
    return diff

def calculate_U_fd():
    '''
    Input parameters in jj basis
    '''
    l_1 = 3
    l_2 = 2

    ryd2ev = 13.605698065894

# Pu from Cowan's code

    F=np.zeros(3)
    F[0] = 0; F[1] = 0.36*ryd2ev*0.8; F[2] = 0.14*ryd2ev*0.8;

    G=np.zeros(3)
    G[0] = 0.062*ryd2ev*0.8; G[1] = 0.067*ryd2ev*0.8; G[2] = 0.057*ryd2ev*0.8;

    U_int=4.5
    J_hund=0.5


    '''
    '''

    m_range_1 = range(2*l_1+1)
    m_range_2 = range(2*l_2+1)

    m_range_f = range( (2*l_1+1)*2 )
    m_range_d = range( (2*l_2+1)*2 )

    # Transform to jj basis
    utrans_f = np.zeros((14,14))
    for i in range(14):
        utrans_f[i][i] = 1.0

    u_cmplx_harm_to_rel_harm = trans_JJ_to_CH_sup_sdn(l_1).T
    utrans_f = u_cmplx_harm_to_rel_harm.dot(utrans_f)

    utrans_d = np.zeros((10,10))
    for i in range(10):
        utrans_d[i][i] = 1.0

    u_cmplx_harm_to_rel_harm_d = trans_JJ_to_CH_sup_sdn(l_2).T
    utrans_d = u_cmplx_harm_to_rel_harm_d.dot(utrans_d)

    coulomb_matrix_ffff = U_matrix(mode="slater",l=l_1,U_int=U_int,J_hund=J_hund,T=utrans_f)
    coulomb_matrix_ffdd = U_matrix_ffdd(l_1,l_2,F,T_f=utrans_f,T_d=utrans_d)
    coulomb_matrix_fddf = U_matrix_fddf(l_1,l_2,G,T_f=utrans_f,T_d=utrans_d)

    #coulomb_matrix_ffff /= ryd2ev

    with open('coulomb_matrix_ffff.txt','w') as f:
        for m1,m2,m3,m4 in product(m_range_f,m_range_f,m_range_f,m_range_f):
            if np.abs(coulomb_matrix_ffff[m1,m2,m3,m4]) < 1.e-8:
                continue
            f.write('{} {} {} {}  {} {}\n'.format(m1+1,m2+1,m3+1,m4+1, \
                    np.real(coulomb_matrix_ffff[m1,m2,m3,m4]),np.imag(coulomb_matrix_ffff[m1,m2,m3,m4])) )
    with open('coulomb_matrix_ffdd.txt','w') as f:
        for m1,m2,m3,m4 in product(m_range_f,m_range_f,m_range_d,m_range_d):
            if np.abs(coulomb_matrix_ffdd[m1,m2,m3,m4]) < 1.e-8:
                continue
            f.write('{} {} {} {}  {} {}\n'.format(m1+1,m2+1,m3+1,m4+1, \
                    np.real(coulomb_matrix_ffdd[m1,m2,m3,m4]),np.imag(coulomb_matrix_ffdd[m1,m2,m3,m4])) ) 
    with open('coulomb_matrix_fddf.txt','w') as f:
        for m1,m2,m3,m4 in product(m_range_f,m_range_d,m_range_d,m_range_f):
            if np.abs(coulomb_matrix_fddf[m1,m2,m3,m4]) < 1.e-8:
                continue
            f.write('{} {} {} {}  {} {}\n'.format(m1+1,m2+1,m3+1,m4+1, \
                    np.real(coulomb_matrix_fddf[m1,m2,m3,m4]),np.imag(coulomb_matrix_fddf[m1,m2,m3,m4])) ) 


def get_T_fd_element(j_c, j_v, m_c, m_v, q, l_c, l_v):
    ele = ( 1.0 if np.abs(j_c-j_v)%2 == 0 else -1.0) * np.sqrt(2*j_c+1)*np.sqrt(2*l_c+1)*np.sqrt(2*l_v+1) \
        * wigner_6j(j_c,1,j_v,l_v,0.5,l_c) * wigner_3j(j_c,1,j_v,m_c,q,-m_v)
    return ele


def fd_transition_matrix():

    l_1 = 3 # f-orbital, valence
    l_2 = 2 # d-orbital, core

    jv_1 = l_1 - 1/2
    jv_2 = l_1 + 1/2

    jc_1 = l_2 - 1/2
    jc_2 = l_2 + 1/2

    T_fd = np.zeros(( (2*l_1+1)*2, (2*l_2+1)*2 ), dtype=float)
    T_q = np.zeros(( (2*l_1+1)*2, (2*l_2+1)*2, 3 ),dtype=float)

    q=-1
    while (q<=1):
        k=q+1
        for i in range( (2*l_1+1)*2 ):
            # jv_1 case
            if(i<(2*jv_1+1)):
                m_v = -jv_1 + i
                for j in range( (2*l_2+1)*2 ):
                    # jc_1 case
                    if(j<(2*jc_1+1)):
                        m_c = -jc_1 + j
                        T_q[i][j][k]=get_T_fd_element(jc_1, jv_1, m_c, m_v, q, l_2, l_1)
                    # jc_2 case
                    else:
                        m_c = -jc_2 + j - (2*jc_1+1)
                        T_q[i][j][k]=get_T_fd_element(jc_2, jv_1, m_c, m_v, q, l_2, l_1)    
            # jv_2 case
            else:
                m_v = -jv_2 + i - (2*jv_1+1)
                for j in range( (2*l_2+1)*2 ):
                    # jc_1 case
                    if(j<(2*jc_1+1)):
                        m_c = -jc_1 + j
                        T_q[i][j][k]=get_T_fd_element(jc_1, jv_2, m_c, m_v, q, l_2, l_1)
                    # jc_2 case
                    else:
                        m_c = -jc_2 + j - (2*jc_1+1)
                        T_q[i][j][k]=get_T_fd_element(jc_2, jv_2, m_c, m_v, q, l_2, l_1)

        q+=1
    # End of while loop

    for i_q in range(3):
        with open('T_fd_{}.txt'.format(i_q-1),'w') as f:
            for m1, m2 in product(range(2*(2*l_1+1)), range(2*(2*l_2+1)) ):
                f.write('{} {} {}\n'.format(m1+1, m2+1, T_q[m1,m2,i_q]) )
    #return T_fd

if __name__ == '__main__':

    #test()
    calculate_U_fd()
    fd_transition_matrix()

    print('Done...')
