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


"""Computes the BEBOP-2 energy functional and/or bond energy contributions
   E_cov :: covalent energy (for sigma and pi)
   E_rep :: short range repulsion energy (for sigma and pi)
   E_hyb_2s :: hybridization energies
   E_qeq :: paired charge fluctuation energies using modified QEq equation
   E_ion :: ionic interaction equation using damped Coulombs law
   conditional_statements :: conditional protocols
   bebop2 :: total bebop2 atomization energy in kcal/mol
   bebop2_bond_energy :: bond energies in kcal/mol (i.e., gross and net bond energies)
   resonance :: resonance energy in kcal/mol
   strain :: strain energy in kcal/mol
   """

import numpy as np
from . import bebop2_params as param
from .roothan import overlap_integrals
from scipy.special import logsumexp

# BEBOP sub-equations
def E_cov(bo_sig, bo_pi, beta_sig, beta_pi):
    """Calculate the sigma and pi covalent 
       extended-Hückel theory bond energy contributions
    
       Parameters
       ----------
       bo_sig: :obj:'numpy.ndarray'
           sigma bond orders
       bo_pi: :obj:'numpy.ndarray'
           pi bond orders
       beta_sig: :obj:'numpy.float64'
           Extend-Hückel theory parameters
           for sigma bond
       beta_pi: :obj:'numpy.float64'
           Extended-Hückel theory parameters
           for pi bond
       
       Returns
       -------
       E_sig: :obj:'numpy.ndarray'     
           sigma bond energies
       E_pi: :obj:'numpy.ndarray'
           pi bond energies
       """
    
    E_sig = 2 * beta_sig * bo_sig 
    E_pi = 2 * beta_pi * bo_pi
    return (E_sig, E_pi)

def E_rep(R, D_sig, D_pi, zeta_sig, zeta_pi, R_sig, R_pi):
    """Calculate the sigma and pi short-range repulsion terms 
    
       Parameters
       ----------
       R: :obj:'numpy.ndarray'
           inter-atomic distances between 
           pair of atoms 
       D_sig: :obj:'numpy.float64'
           pre-exponential sigma constant 
           for the short-range repulsion term  
       D_pi: :obj:'numpy.float64'
           pre-exponential pi constant 
           for the short-range repulsion term 
       zeta_sig: :obj:'numpy.float64'
           exponential sigma constant 
           for the short-range repulsion term  
       zeta_pi: :obj:'numpy.float64'
           exponential pi constant 
           for the short-range repulsion term 
       R_sig: :obj:'numpy.float64'
           classical turning point distance for the sigma bond
       R_pi: :obj:'numpy.float64'
           classical turning point distance for the pi bond
           
       Returns
       -------
       E_sig: :obj:'numpy.ndarray'     
           sigma short-range repulsion energies
       E_pi: :obj:'numpy.ndarray'
           pi short-range repulsion energies
       """
    
    E_sig = D_sig * np.exp(-zeta_sig * (R - R_sig))
    E_pi = D_pi * np.exp(-zeta_pi * (R - R_pi))
    return (E_sig, E_pi)

def E_hyb_2s(atom, occ_2s):
    """Calculate the hybridization energy going from 2s -> 2p
    
       Parameters
       ----------
       atom: :obj:'numpy.ndarray'
           all atoms present in the molecule
       occ_2s: :obj:'numpy.float64'
           occupation value of the 2s electrons
           
       Returns
       -------
       E_hyb: :obj:'numpy.ndarray'     
           hybridization energy going from 2s -> 2p
       """
    
    n2s, E2s2p = param.bebop_hybrid(atom)
    E_hyb = E2s2p * (n2s - occ_2s)
    return E_hyb

def E_qeq(pair_atoms, pair_charges, xi_damp_1, R):
    """Calculate the charge transfer energy using 
       the bond-charge equilibration model
    
       Parameters
       ----------
       pair_atoms: :obj:'numpy.ndarray'
           array showing the two pair of atoms
       pair_charges: :obj:'numpy.ndarray'
           array showing the two pair charges
           from each respective atoms
       xi_damp_1: :obj:'numpy.float64'
           damping parameter for the charge equilibration model
       R: :obj:'numpy.float64'
           distance between the two atoms in the molecule
           
       Returns
       -------
       energy: :obj:'numpy.ndarray'     
           bond-charge equilibration model energy
       """
    QEq = 0 
    for l in range(pair_atoms.shape[0]):
        xi_Mulliken, eta = param.bebop_electronegativity_hardness(pair_atoms[l]) 
        QEq += xi_Mulliken * pair_charges[l] + eta * pair_charges[l]**2
    energy = QEq * np.exp(-xi_damp_1 * R) * 627.5098 # in kcal/mol 
    return energy

def E_ion(pair_charges, xi_damp_2, R, pair_atoms):
    """Calculate the point-charge pontential energy equation
    
       Parameters
       ----------
       pair_charges: :obj:'numpy.ndarray'
           array showing the two pair charges
           from each respective atoms
       xi_damp_2: :obj:'numpy.float64'
           damping parameter for the coulomb model
       R: :obj:'numpy.float64'
           distance between the two atoms in the molecule
       pair_atoms: :obj:'numpy.ndarray'
           array showing the two pair of atoms
           
       Returns
       -------
       energy: :obj:'numpy.ndarray'     
           bond-charge equilibration model energy
       """
    R_amu = R / 0.52917721092 # from Ansgtroms to Bohr
    energy = np.prod(pair_charges) *  np.exp(-xi_damp_2 * R)/ R_amu * 627.5098 # in kcal/mol
    return energy

def conditional_statements(pair_atoms, bo_sig_pair, bo_pi_pair):
    """Condition protocols for computing bond energies
    
       Parameters
       ----------
       pair_atoms: :obj:'numpy.ndarray'
           array showing the two pair of atoms
       bo_sig_pair: :obj:'numpy.float64'
           sigma bond order
       bo_pi_pair: :obj:'numpy.float64'
           pi bond order
           
       Returns
       -------
       corrections: :obj:'int'     
           the correction value: 
               0 -> bo_sig > 0 and bo_pi < 0
               1 -> bo_pi > 0
               2 -> no conditions
       """
    # Conditional protocols for computing bond energies
    bond_pairs_1 = np.array(['O~O','N~N','O~N','N~O','C~C','O~C','C~O','C~N','N~C','B~B','B~O','O~B','C~B','B~C'])
    bond_pairs_2 = np.array(['Li~Li'])
    if (f'{pair_atoms[0]}~{pair_atoms[1]}' in bond_pairs_1) and (bo_sig_pair > 0 and 0 > bo_pi_pair):
        if (np.abs(bo_pi_pair) > np.abs(bo_sig_pair)):
            corrections = 2
        else:
            corrections = 0
    elif (f'{pair_atoms[0]}~{pair_atoms[1]}' in bond_pairs_2) and (bo_pi_pair > 0):
        corrections = 1
    elif (f'{pair_atoms[0]}~{pair_atoms[1]}' in bond_pairs_1) and (bo_sig_pair > 0 and 0 > bo_pi_pair and 0 > bo_sig_pair + bo_pi_pair):
        corrections = 2
    else:
        corrections = 2
    return corrections

# MAIN BEBOP EQUATIONS
def bebop2(distance_matrix, atoms, charges, mol_occ_2s, bo_sig, bo_pi, 
           parameter_folder):
    """Calculate BEBOP-2 energy and energy terms 
    
       Parameters
       ----------
       distance_matrix: :obj:'numpy.ndarray'
           distance matrix in Angstroms
       atoms: :obj:'numpy.ndarray'
           array of atoms
       charges: :obj:'numpy.ndarray'
           array of atomic charges
       mol_occ_2s: :obj:'numpy.ndarray'
           fractional amount of 2s electrons 
           in each atom
       bo_sig: :obj:'numpy.ndarray'
           sigma bond order matrix of the molecule
       bo_pi: :obj:'numpy.ndarray'
           pi bond order matrix of the molecule
       parameter_folder: :obj:'numpy.ndarray'
           name of the path to the parameter folder
           
       Returns
       -------
       bebop2: :obj:'numpy.float64'
           total energy of BEBOP-2
       E_cov_sig: :obj:'numpy.ndarray'
           covalent sigma bond energy contributions
       E_cov_pi: :obj:'numpy.ndarray'
           covalent pi bond energy contributions
       E_rep_sig: :obj:'numpy.ndarray'
           short-range sigma repulsion bond energy contributions'
       E_rep_pi: :obj:'numpy.ndarray'
           short-range pi repulsion bond energy contributions
       E_fluc: :obj:'numpy.ndarray'
           charge fluctuation bond energy contributions calculated 
           from a modified QEq equation
       E_coulomb: :obj:'numpy.ndarray'
           ionic coulomb bond energy contributions
       E_hybridization: :obj:'numpy.ndarray' 
           orbital hybridization energy contribution (2s -> 2p) 
           for each atom
       """     
    size = atoms.shape[0]
    E_cov_sig = np.zeros((size,size), dtype = np.float64) # covalent energy for sigma
    E_cov_pi = np.zeros((size,size), dtype = np.float64) # covalent energy for pi
    E_rep_sig = np.zeros((size,size), dtype = np.float64) # short-range repulsion for sigma
    E_rep_pi = np.zeros((size,size), dtype = np.float64) # short-range repulsion for pi
    E_fluc = np.zeros((size,size), dtype = np.float64) # charge fluctuation using modified QEq equation
    E_coulomb = np.zeros((size,size), dtype = np.float64) # coulombs law for ionic interactions
    
    for l in range(1,size):
        for n in range(l):
            # important properties
            index = (l,n)
            pair_atoms = np.array([atoms[l], atoms[n]])
            pair_charges = np.array([charges[l], charges[n]])
            R = distance_matrix[l][n]
            bo_sig_pair = bo_sig[l][n]
            bo_pi_pair = bo_pi[l][n]
            
            anti_corr = conditional_statements(pair_atoms, bo_sig_pair, bo_pi_pair)
            
            # fitting parameters
            parameters = param.bebop_pair(pair_atoms, parameter_folder, anti_corr) # BEBOP atom-paired parameters
            beta_sig, beta_pi, zeta_sig, zeta_pi, R_sig, R_pi, D_sig, D_pi, xi_damp_1, xi_damp_2 = parameters

            # extended Hückel covalent bonding for sigma and pi bonding 
            sig_contr, pi_contr = E_cov(bo_sig_pair, bo_pi_pair, beta_sig, beta_pi)
            E_cov_sig[index] = sig_contr
            E_cov_pi[index] = pi_contr  
            
            # Short-range repulsion wall
            sig_contr, pi_contr = E_rep(R, D_sig, D_pi, zeta_sig, zeta_pi, R_sig, R_pi)
            E_rep_sig[index] = sig_contr
            if bo_pi_pair == 0:
                E_rep_pi[index] = 0 
            else:
                E_rep_pi[index] = pi_contr
            
            if pair_atoms[0] == pair_atoms[1] and pair_atoms[0] in np.array(['H','F']):
                continue
            else:
                E_fluc[index] = E_qeq(pair_atoms, pair_charges, xi_damp_1, R) # Charge fluctuation using the modified QEq equation
                E_coulomb[index] = E_ion(pair_charges, xi_damp_2, R, pair_atoms) # Ionic coulomb interactions
                
    # Calculate the hybridization energy
    E_hybridization = np.zeros(size, dtype = np.float64)
    for l in range(size):
        if atoms[l] in np.array(['H','He','Li','F']):
            continue # skip if 'H' and/or 'He' is present (no hybridization!)
        else:
            E_hybridization[l] = E_hyb_2s(atoms[l], mol_occ_2s[l])
                
    # Calculate total bebop2 energy
    paired_matrix = E_cov_sig + E_cov_pi + E_rep_sig + E_rep_pi + E_fluc + E_coulomb
    bebop2 = np.sum(paired_matrix) + np.sum(E_hybridization)
    return (bebop2, E_cov_sig, E_cov_pi, E_rep_sig, E_rep_pi, E_fluc, 
            E_coulomb, E_hybridization)

def bebop2_bond_energy(bebop2, E_cov_sig, E_cov_pi, E_rep_sig, E_rep_pi, E_fluc, 
                       E_coulomb, E_hybrid):
    """Calculate the individual bond energy contributions
    
       Parameters
       ----------
       bebop2: :obj:'numpy.float64'
           total energy of BEBOP-2
       E_cov_sig: :obj:'numpy.ndarray'
           covalent sigma bond energy contributions
       E_cov_pi: :obj:'numpy.ndarray'
           covalent pi bond energy contributions
       E_rep_sig: :obj:'numpy.ndarray'
           short-range sigma repulsion bond energy contributions'
       E_rep_pi: :obj:'numpy.ndarray'
           short-range pi repulsion bond energy contributions
       E_fluc: :obj:'numpy.ndarray'
           charge fluctuation bond energy contributions calculated 
           from a modified QEq equation
       E_coulomb: :obj:'numpy.ndarray'
           ionic coulomb bond energy contributions
       E_hybrid: :obj:'numpy.ndarray' 
           orbital hybridization energy contribution (2s -> 2p) 
           for each element
       
       Returns
       -------
       E_sig: :obj:'numpy.ndarray'     
           Gross sigma bond energies
       E_pi: :obj:'numpy.ndarray' 
           Gross pi bond energies
       E_gross: :obj:'numpy.ndarray' 
           Gross covalent bond energies (i.e., E_gross = E_sig + E_pi)
       E_gross_charge: :obj:'numpy.ndarray'
           Gross covalent bond energies with charge transfer 
           and electrostatics (i.e., E_gross_charge = E_gross + E_fluc + E_coulomb)
       E_net_sig: :obj:'numpy.ndarray'
           Net sigma bond energies with charge transfer, electrostatics, and hybridization
       E_net_pi: :obj:'numpy.ndarray'
           Net pi bond energies with charge transfer, electrostatics, and hybridization
       E_net: :obj:'numpy.ndarray'
           Net bond energies (i.e., E_net = E_net_sig + E_net_pi + E_fluc + E_coulomb + E_hyb)
       CompositeTable: :obj:'numpy.ndarray'
           Matrix showing hybridization energy(diagonal elements),
           gross covalent bond energies(upper diagonal elements), 
           and net bond energies (lower diagonal elements)
       E_charge: :obj:'numpy.float64', 
           Total energy due to charge fluctuation and electrostatics
       """
    
    size = E_hybrid.shape[0]
    # Gross charge bond energy terms
    E_sig = E_cov_sig + E_rep_sig
    E_pi = E_cov_pi + E_rep_pi
    E_gross = E_sig + E_pi
    E_charge = E_fluc + E_coulomb
    E_gross_charge = E_gross + E_charge
    
                
    E_net_sig = np.zeros((size,size), dtype = np.float64)
    E_net_pi = np.zeros((size,size), dtype = np.float64)
    E_net = np.zeros((size, size), dtype = np.float64) # calculate pi net bond energies
    TBE = np.zeros(size, dtype = np.float64)
    # Calculate total net bond energy
    for i in range(1,size):
        for j in range(i):
            TBE[i] += E_gross[i][j]
            TBE[j] += E_gross[i][j]
    
    for i in range(1,size):
        for j in range(i):
            Factor = 1.0 + E_hybrid[i] / TBE[i] + E_hybrid[j] / TBE[j] # scaling factor
            E_net[i][j] = E_gross[i][j] * Factor + E_charge[i][j]  # total net sigma and net pi energies contributions
            E_net_sig[i][j] = E_sig[i][j] * Factor + E_charge[i][j] * np.divide(E_sig[i][j], E_gross[i][j],out=np.zeros_like(E_sig[i][j]),where= E_gross[i][j]!=0) # total net sigma bond energies (including charge contr.)
            E_net_pi[i][j] = E_pi[i][j] * Factor + E_charge[i][j] * np.divide(E_pi[i][j], E_gross[i][j],out=np.zeros_like(E_pi[i][j]),where= E_gross[i][j]!=0)  # total net pi bond energies (including charge contr.)
    
    CompositeTable = np.zeros((size,size), dtype = np.float64)     # Composite Table: Eii (hybrid),Eji (gross),
                                                                   #                  Eij (net)   ,Ejj (hybrid) 
    
    for i in range(1,size):
        for j in range(i):
            CompositeTable[j][i] = E_gross_charge[i][j] # gross energies for 
                                                        # the upper diagonal matrices
            CompositeTable[i][j] = E_net[i][j] # net bond energies for
                                               # the lower diagonal elements
    np.fill_diagonal(CompositeTable, E_hybrid) # Fill the diagonal elements of the CompositeTable with the Ehybrid contributions       
    return (E_sig, E_pi, E_gross, E_gross_charge, E_net_sig, E_net_pi, E_net, CompositeTable, E_charge)
    

def resonance(Data, parameter_folder):
    """Compute the resonance energy of the aromatic system (in kcal/mol)
    
        Parameters
        ----------
        Data: :obj:'dict'
            A dictionary containing the following inside the tuple: (total number of bond orders, 
            total number of unique bonds, reference bond orders, atoms arrays) for each reference
            bond
        
        Return
        ------
        res_E: :obj:'np.float'
            Return the total resonance energy change
    """
    res_E = 0
    for l in Data.keys():
            bo_sigma, bo_pi, n, ref_BO_sigma, ref_BO_pi, atoms = Data[l] # unpack the data
            anti_corr = conditional_statements(atoms, bo_sigma, bo_pi)
            beta_sig, beta_pi = param.bebop_pair(atoms, parameter_folder, anti_corr, res_strain = True) # get beta parameter
            res_E_sigma = (bo_sigma - np.int(n) * ref_BO_sigma) * beta_sig
            res_E_pi = (bo_pi - np.int(n) * ref_BO_pi) * beta_pi
            res_E += res_E_sigma + res_E_pi
    return res_E
    
def strain(Data, parameter_folder):
    """Compute the ring strain energy for a ring (in kcal/mol)
    
    Parameters
    ----------
    Data: :obj:'dict'
        A dictionary containing the following inside the tuple: (sigma bond order, pi bond order
        total number of unique bonds, reference bond orders, atoms arrays) for each reference
        bond
        
    Return
    ------
    res_E: :obj:'np.float'
        Return the total resonance energy change
    """    
    strain_E = 0
    for l in Data.keys():
        atom_pairs = np.array(['C','C'])
        bo_tot_sig, bo_tot_pi, n, ref_bo_sig, ref_bo_pi = Data[l] # unpack the data
        anti_corr = conditional_statements(atom_pairs, bo_tot_sig, bo_tot_pi)
        beta_sig, beta_pi = param.bebop_pair(atom_pairs, parameter_folder, anti_corr, res_strain = True) # get beta parameter
        strain_E += (np.int(n) * ref_bo_sig - bo_tot_sig) * beta_sig + (np.int(n) * ref_bo_pi - bo_tot_pi) * beta_pi
        
    return strain_E
