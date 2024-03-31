# MIT License
# 
# Copyright (c) 2024, Barbaro Zulueta
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


"""Parameters for the BEBOP-2 code
   bebop_pair :: BEBOP atom-pair parameters (i.e., beta_sig, beta_pi,zeta_sig, etc.)
   bebop_hybrid :: hybridization parameters (i.e., n_2s, W1BD excitation energies from 1s to 2s, damping parameter)
   bebop_electronegativity_hardness :: Mulliken electronegativities and chemical hardness 
   """

import numpy as np
import json

def bebop_pair(pair_atoms, parameter_folder_path, anti_corr, res_strain = False):
    """Fitted atom-pair parameters used in the BEBOP equation (i.e., beta, zeta, R_sigma and D_AB)
    
       Parameters
       ----------
       pair_atoms: :obj:'np.ndarray'
           Name of the two-pair atoms 
       parameter_folder_path: :obj:'str'
           Name of the parameter folder path
       anti_corr: :obj:'int'
           Conditional protocol
       res_strain: :obj:'bool', optional
           Need the beta parameter only for resonance and strain calculations
       
       Returns 
       -------
       beta_sig: :obj:'np.float'
           Fixed parameter used compute the extended-Hückel bond energy for sigma bonding  
       beta_pi: :obj:'np.float'
           Fixed parameter used compute the extended-Hückel bond energy for pi bonding  
       zeta_sig: :obj:'np.float'
           Slater-type exponential parameter for the short-range repulsion for sigma bonding
       zeta_pi: :obj:'np.float'
           Slater-type exponential parameter for the short-range repulsion for pi bonding
       R_sig: :obj:'np.float'
           Classical turning distance for sigma bonding 
       R_pi: :obj:'np.float'
           Classical turning distance for pi bonding
       D_sig: :obj:'np.float'
           Bond dissociation energy parameter for sigma bonding
       D_pi: :obj:'np.float'
           Bond dissociation energy parameter for pi bonding
       xi_1: :obj:'np.float'
           Damping parameter for charge fluctuation model
       xi_2: :obj:'np.float'
           Damping parameter for point-charge model
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
    
    #               H       He      Li     Be      B      C      N      O      F
    files = {'H':[  'H~H', 'NaN','Li~H', 'Be~H', 'B~H', 'C~H', 'N~H', 'O~H', 'H~F'],
             'He':[ 'NaN', 'NaN','NaN',  'NaN',  'NaN', 'NaN', 'NaN', 'NaN', 'NaN'],
             'Li':['Li~H', 'NaN','Li~Li','Be~Li','B~Li','C~Li','N-Li','O~Li','Li~F'],
             'Be':['Be~H', 'NaN','Be~Li','Be~Be','B~Be','C~Be','N~Be','O~Be','Be~F'],
             'B':[  'B~H', 'NaN','B~Li', 'B~Be', 'B~B', 'C~B', 'N~B', 'O~B', 'B~F'],
             'C':[  'C~H', 'NaN','C~Li', 'C~Be', 'C~B', 'C~C', 'N~C', 'O~C', 'C~F'],
             'N':[  'N~H', 'NaN','N~Li', 'N~Be', 'N~B', 'N~C', 'N~N', 'O~N', 'N~F'],
             'O':[  'O~H', 'NaN','O~Li', 'O~Be', 'O~B', 'O~C', 'O~N', 'O~O', 'O~F'],
             'F':[  'H~F', 'NaN','Li~F', 'Be~F', 'B~F', 'C~F', 'N~F', 'O~F', 'F~F']}
    
    opt_file = files[pair_atoms[0]][AtomN[pair_atoms[1]]]
    with open(parameter_folder_path + f'{opt_file}/{opt_file}_param_opt.json', "r") as openfile:
        parameter = json.load(openfile)
        openfile.close()
    if anti_corr == 0:
        # for O~O and N~N bond only
        beta_sig = parameter['beta_sig'] # in kcal/mol
        zeta_sig = parameter['zeta_sig']
        D_sig = parameter['D_sig'] # in kcal/mol
        xi_1 = parameter['xi_param_1'] # in \AA^-1
        xi_2 = parameter['xi_param_2'] # in \AA^-1
        R_sig = parameter['R_sig'] # in \AA
        R_pi = parameter['R_pi'] # in \AA
        beta_pi = parameter[f'k_cov'] * parameter['beta_pi'] # in kcal/mol
        zeta_pi = parameter['zeta_cor'] * parameter['zeta_pi'] # in \AA^{-1}
        D_pi = parameter[f'k_rep'] * parameter['D_pi'] # in kcal/mol
    elif anti_corr == 1:
        # for Li~Li bond only 
        beta_sig = parameter[f'k_cov'] * parameter['beta_sig'] # in kcal/mol
        beta_pi =  parameter['beta_pi'] # in kcal/mol
        zeta_sig = parameter['zeta_cor'] * parameter['zeta_sig'] # in \AA^{-1}
        zeta_pi = parameter['zeta_pi'] # in \AA^{-1}
        D_sig = parameter['D_sig'] # in kcal/mol
        D_pi =  parameter['D_pi'] # in kcal/mol
        xi_1 = parameter['xi_param_1'] # in \AA^-1
        xi_2 = parameter['xi_param_2'] # in \AA^-1
        R_sig = parameter['R_sig'] # in \AA
        R_pi = parameter['R_pi'] # in \AA
    elif anti_corr == 2:
        # for all other bonds
        beta_sig = parameter['beta_sig'] # in kcal/mol
        beta_pi = parameter['beta_pi'] # in kcal/mol
        zeta_sig = parameter['zeta_sig'] # in \AA^{-1}
        zeta_pi = parameter['zeta_pi'] # in \AA^{-1}
        R_sig = parameter['R_sig'] # in \AA
        R_pi = parameter['R_pi'] # in \AA
        D_sig = parameter['D_sig'] # in kcal/mol
        D_pi = parameter['D_pi'] # in kcal/mol
        xi_1 = parameter['xi_param_1'] # in \AA^-1
        xi_2 = parameter['xi_param_2'] # in \AA^-1
    if res_strain == False:    
        return (beta_sig, beta_pi, zeta_sig, zeta_pi, R_sig, R_pi, D_sig, D_pi, xi_1, xi_2)
    else:
        return (beta_sig, beta_pi)

def bebop_hybrid(Atom):
    """Element_wise parameters used in the BEBOP equation (i.e., n_2s, excitation energies from 2s to 2p)
    
       Parameters
       ----------
       Atom: :obj:'str'
           The name of the atom (i.e, 'H','He',etc.) 
       Returns
       -------
       Hybrid[Atom]: :obj:'tuple'
           (number of 2s electrons, W1BD excitation energy from 2s to 2p) in Atom object
       """
    
    Hybrid = {'H':(0.000,0.0000),
              'He':(0.000,0.000),
              'Li':(1.000,42.542),
              'Be':(2.000,62.943),
              'B':(2.000,82.590),
              'C':(2.000,95.993),
              'N':(2.000,133.155),
              'O':(2.000,171.599),
              'F':(2.000,210.582)}
    
    return Hybrid[Atom] # units for the tuple: (unitless, kcal/mol)

def bebop_electronegativity_hardness(Atom):
    """Element_wise parameters used in the BEBOP equation (i.e., n_2s, excitation energies from 2s to 2p)
    
       Parameters
       ----------
       Atom: :obj:'str'
           The name of the atom (i.e, 'H','He',etc.)  
       Returns
       -------
       xi: :obj:'np.float'
           Mulliken electronegativies (in Hartrees/e)
       eta: :obj:'np.float'
           Chemical hardness (in Hartrees/e^2)
       """
    
    w1bd_energies = {'H':   -0.499994, # energies (in Hartrees) of neutral,
                     'H+':   0.000000, # cationic, and anionic elements from W1BD
                     'H-':  -0.510658,
                     'Li':  -7.472052,
                     'Li+': -7.274087,
                     'Li-': -7.494796,
                     'Be': -14.663427,
                     'Be+':-14.320843,
                     'Be-':-14.651308,
                     'B':  -24.653730,
                     'B+': -24.349567,
                     'B-': -24.663269,
                     'C':  -37.852873,
                     'C+': -37.439425,
                     'C-': -37.898986,
                     'N':  -54.611188,
                     'N+': -54.076852,
                     'N-': -54.604713,
                     'O':  -75.111239,
                     'O+': -74.611882,
                     'O-': -75.164169,
                     'F':  -99.811418,
                     'F+': -99.171493,
                     'F-': -99.937161}
    
    xi = (w1bd_energies[Atom + '+'] - w1bd_energies[Atom + '-']) / 2  # Mulliken electronegativities
    eta = (w1bd_energies[Atom + '+'] + w1bd_energies[Atom + '-'] - 2 * w1bd_energies[Atom]) / 2 # chemical hardness
    return (xi, eta) 