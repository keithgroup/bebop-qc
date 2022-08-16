# MIT License
# 
# Copyright (c) 2022, Barbaro Zulueta
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Parameters for the BEBOP code (version 1.0.0)

BEBOP_Pair :: BEBOP atom-pair parameters (i.e., beta, zeta, R_sigma and D_AB)
BEBOP_Atom :: single atom parameters (i.e., n_2s, CBS-QB3 excitation energies from 1s to 2s)
"""

import numpy as np

def BEBOP_Pair(Atom1, Atom2, res_strain=False):
    """Fitted atom-pair parameters used in the BEBOP equation (i.e., beta, zeta,
    R_sigma and D_AB).

    Parameters
    ----------
    Atom1 : :obj:'str'
        Name of the atom (i.e, 'H','He',etc.)
    Atom2 : :obj:'str'
        Name of the atom interacting with Atom1 (i.e,'H','He',etc.)
    res_strain : :obj:'bool', optional
        Need the beta parameter only for resonance and strain calculations
    
    Returns 
    -------
    beta : :obj:`np.float`
        Fixed parameter used compute the extended-Hückel bond energy (with ZPE)  
    zeta : :obj:`np.float`
        Slater-type exponential parameter for the short-range repulsion
    R_sigma : :obj:`np.float`
        Classical turning distance (i.e., :math:`E_tot \approx E_short`)  
    D_AB : :obj:`np.float`
        Bond dissociation energy parameter (with ZPE) 
    """

    AtomN = {'H': 0,
             'He': 1,
             'Li': 2,
             'Be': 3,
             'B': 4,
             'C': 5,
             'N': 6,
             'O': 7,
             'F': 8}
    
            #                  H      He       Li      Be       B       C       N       O       F
    beta = {'H':np.array([  144.77,  0.000,  119.91, 143.19, 168.44, 178.45, 192.39, 258.42, 372.61]),
            'He':np.array([  0.000,  0.000,   0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]),
            'Li':np.array([ 119.91,  0.000,   43.41,  86.36, 121.82, 178.60, 207.00, 342.56, 760.99]),
            'Be':np.array([ 143.19,  0.000,   86.36, 122.27, 147.88, 193.93, 217.45, 298.97, 474.24]),
            'B':np.array([  168.44,  0.000,  121.82, 147.88, 160.70, 201.30, 233.62, 316.25, 468.04]),
            'C':np.array([  178.45,  0.000,  178.60, 193.93, 201.30, 225.87, 233.16, 291.28, 403.33]),
            'N':np.array([  192.39,  0.000,  207.00, 217.45, 233.62, 233.16, 215.55, 271.38, 322.98]),
            'O':np.array([  258.42,  0.000,  342.56, 298.97, 316.25, 291.28, 271.38, 257.25, 252.10]),
            'F':np.array([  372.61,  0.000,  760.99, 474.24, 468.04, 403.33, 322.98, 252.10, 289.78])}
    
            #                 H     He     Li     Be     B      C      N      O       F
    zeta = {'H':np.array([ -8.19, -9.55, -2.71, -5.31, -5.92, -7.17, -7.81, -8.68,  -9.56]),
            'He':np.array([-9.55, -9.55, -9.55, -9.55, -9.55, -9.55,- 9.55, -9.55,  -9.55]),
            'Li':np.array([-2.71, -9.55, -3.26, -2.85, -5.77, -7.02, -4.85, -2.18,  -1.24]),
            'Be':np.array([-5.31, -9.55, -2.85, -4.28, -4.27, -4.26, -4.25, -4.25,  -4.99]),
            'B':np.array([ -5.92, -9.55, -5.77, -4.27, -6.73, -6.68, -6.64, -5.57,  -4.61]),
            'C':np.array([ -7.17, -9.55, -7.02, -4.26, -6.68, -7.53, -7.44, -7.30,  -7.57]),
            'N':np.array([ -7.81, -9.55, -4.85, -4.25, -6.64, -7.44, -13.84, -8.99, -7.50]),
            'O':np.array([ -8.68, -9.55, -2.18, -4.25, -5.57, -7.30, -8.99, -8.42, -10.91]),
            'F':np.array([ -9.56, -9.55, -1.24, -4.99, -4.61, -7.57, -7.50, -10.91, -1.17])}
    
            #                H       He     Li     Be     B      C      N      O      F
    R_eq = {'H':np.array([ 0.654,  1.000, 1.593, 1.327, 1.190, 1.091, 1.016, 0.962, 0.920]),
            'He':np.array([1.000,  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]),
            'Li':np.array([1.593,  1.000, 2.705, 2.399, 2.183, 1.677, 1.569, 1.570, 1.560]),
            'Be':np.array([1.327,  1.000, 2.399, 2.080, 1.867, 1.673, 1.495, 1.518, 1.373]),
            'B':np.array([ 1.190,  1.000, 2.183, 1.867, 1.727, 1.553, 1.388, 1.271, 1.324]),
            'C':np.array([ 1.091,  1.000, 1.677, 1.673, 1.553, 1.532, 1.392, 1.200, 1.389]),
            'N':np.array([ 1.016,  1.000, 1.569, 1.495, 1.388, 1.392, 1.095, 1.148, 1.430]),
            'O':np.array([ 0.962,  1.000, 1.570, 1.518, 1.271, 1.200, 1.148, 1.207, 1.434]),
            'F':np.array([ 0.920,  1.000, 1.560, 1.373, 1.324, 1.389, 1.430, 1.434, 1.408])}

            #                H      He      Li      Be      B       C       N       O       F
    D =    {'H':np.array([104.45, 0.000,  55.68,  92.28, 104.54, 103.60, 105.93, 117.73, 136.01]),
            'He':np.array([0.000, 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]),
            'Li':np.array([55.68, 0.000,  24.03,  42.30,  44.48,  46.45,  72.02, 102.88, 136.27]),
            'Be':np.array([92.28, 0.000,  42.30,  71.67,  82.44,  92.03, 120.86, 147.18, 176.64]),
            'B':np.array([104.54, 0.000,  44.48,  82.44, 136.32, 143.94, 177.14, 219.44, 169.55]),
            'C':np.array([103.60, 0.000,  46.45,  92.03, 143.94, 226.85, 157.24, 179.47, 110.06]),
            'N':np.array([105.93, 0.000,  72.02, 120.86, 177.14, 157.24, 122.65, 150.43,  69.31]),
            'O':np.array([117.73, 0.000, 102.88, 147.18, 219.44, 179.47, 150.43, 119.71,  48.44]),
            'F':np.array([136.01, 0.000, 136.27, 176.64, 169.55, 110.06,  69.31,  48.44,  37.44])}
    
    betaAB = beta[Atom1][AtomN[Atom2]] # in kcal/mol
    if res_strain == True:
        return betaAB
    else:
        zetaAB = zeta[Atom1][AtomN[Atom2]] # in \AA^{-1}
        R_sigma = R_eq[Atom1][AtomN[Atom2]] / np.sqrt(2) # in \AA
        D_AB = D[Atom1][AtomN[Atom2]] # in kcal/mol
        return (betaAB, zetaAB, R_sigma, D_AB)

def BEBOP_Atom(Atom):
    """Single-atom parameters used in the BEBOP equation (i.e., n_2s, excitation
    energies from 2s to 2p)
    
    Parameters
    ----------
    Atom : :obj:'str'
        The name of the atom (i.e, 'H','He',etc.)  
    
    Returns
    -------
    :obj:`tuple`
        Hybrid[Atom]: (number of 2s electrons, UCBS-QB3 excitation energy from
        2s to 2p) in Atom object.
    """
    
    Hybrid = {'H':(0.000,0.0000),
              'He':(0.000,0.000),
              'Li':(1.000,42.479),
              'Be':(2.000,65.008),
              'B':(2.000,85.177),
              'C':(2.000,98.098),
              'N':(2.000,134.827),
              'O':(2.000,172.055),
              'F':(2.000,209.517)}
    return Hybrid[Atom] # units for the tuple: (unitless, in kcal/mol)
