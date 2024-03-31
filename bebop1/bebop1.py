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

"""Main subroutine that computes all BEBOP-1 calculations

total_E:: compute the BEBOP1 atomization energy with zero-point vibrational energy (ZPVE) at 0 K
bond_E:: compute bond energy tables
sort_bondE:: sort the bond energies in order
resonance_E:: get the computed resonance energy
strain_E:: get the computed strain energy
"""

from . import bebop1_equation as bebop_eq 
from . import read_output as ro
from . import roothan
from . import spatial_geom as sg
import numpy as np

class BEBOP:
    
    def __init__(self, name):
        """Give the name of the ROHF output file to get the bond energy data.
        
        Parameters
        ----------
        name: :obj:`str`
            Name of the ROHF/CBSB3 output file.
        """
        
        self.name = name 
        nAtoms, XYZ, CiCjAlpha, CiCjBeta, PopMatrix, NISTBF, Occ2s, Mulliken = ro.read_entire_output(name) # get all data from the output file
        CiCj = CiCjAlpha + CiCjBeta # sum of the population matrix for up and down spins
        self.mol = nAtoms # array containing the atoms within the molecule
        self.mol2s = Occ2s # array or float of molecular occupation of 2s electrons
        self.mulliken = Mulliken # Mulliken MBS bond orders condensed to atoms
        self.CiCj = CiCj 
        self.pop = PopMatrix # Mulliken MBS population matrix condenced to orbitals
        self.elec_Occ = NISTBF # array showing where the orbitals begin and end for an atom (important for roothan package) 
        DistanceMatrix, Angles = sg.Spatial_Properties(XYZ) # distance matrix, the projection angles
        self.distance_matrix = DistanceMatrix 
        self.angles = Angles 
        
        return None
    
    def total_E(self):
        """Compute the total atomization energy in kcal/mol
        
        Output
        ------
        total : :obj:`np.float`
            Total BEBOP atomization energy including zero-point vibrational energy (ZPVE) at 0 K
        """

        total = bebop_eq.BEBOP1(self.distance_matrix, self.mol, self.mol2s, self.mulliken)
        return total 
        
    def bond_E(
        self, NetBond=True, GrossBond=False, Composite=False, TotalE=False,
        sig_pi=False
    ):
        """Compute all bond energies in kcal/mol.All energies are printed by
        default.

        User may select which energies will be returned by selecting 'True'. 
        
        New users are advise to use the '.keys()' method on the output variables
        to check the name of the keys in the dictionaries.
        
        Parameters
        ----------
        NetBond : obj:'bool',optional
            Generate the bond energy to include repulsion and hybridization corrections only
            Returns net covalent bond energies (Enet) if 'NetBond = True' and the net pi covalent bond
            energies (Enetpi) and the net sigma bond energies (Enetsig) if 'sig_pi = False'. 
        GrossBond : obj:'bool', optional
            Generate the bond energy to include repulsion corrections only 
            Returns covalent bond energies (Ecov) if 'GrossBond = True' and the gross pi covalent bond
            energies (Epi) and the gross sigma bond energies (Esig) if 'sig_pi = False'. 
        Composite : obj:'bool', optional
            Return the composite table (CompositeTable) containing the net bond energies (lower diagonal elements), 
            the hybridization energies (diagonal elements), and the gross bond energies (upper diagonal elements)
        TotalE : obj:'bool', optional
            Return the BEBOP total atomization energy with ZPVE at 0 K 
        sig_pi : obj:'bool', optional
            Return sigma and pi bonding energy decompositions for Enet and Ecov. 
            If 'sig_pi = True', output will be a tuple (first DictionaryTotal and second DictionaryDecomp)
        
        Returns
        -------
        :obj:`dict`
            DictionaryTotal: Dictionary containing Enet (key: 'NetBond'), Ecov (key: 'GrossBond'), 
            and CompositeTable (key: 'CompositeTable')
        :obj:`dict`, optional
            Dictionary containing Esig (key: 'Egross_sigma'), Epi (key: 'Egross_pi'), 
            Enetsig (key: 'Enet_sigma'), and Enetpi (key: 'Enet_pi')
        :obj:`np.float64`, optional
            Return the BEBOP total atomization energy with ZPVE at 0 K
        """
        
        AllTotalEnergies = [] # stored data for the total energies (defined by the user)
        AllDecompEnergies = [] # stored data for the decomposed energies (defined by the user)
        keysTotalE = [] # name of the keys for the type of energy 
        keysDecompE = [] # name of the keys for the bond energy types 
        Input = (self.CiCj, self.pop, self.mulliken, self.angles, self.distance_matrix, self.elec_Occ, self.mol)
        Sigma, Pi = roothan.SigmaPiBondOrders(*Input) # get the sigma and pi bond orders from Roothan's expression
        Data = bebop_eq.BEBOPBondEnergies(Sigma, Pi, self.mulliken, self.distance_matrix, self.mol, self.mol2s) 
        Esig, Epi, Ecov, Enetsig, Enetpi, Enet, CompositeTable, BEBOP = Data # all bond energy and total energy data
        
        # store all of the data in a list
        if GrossBond == True:
            AllTotalEnergies += Ecov,
            AllDecompEnergies += Esig, Epi,
            keysTotalE += 'GrossBond',
            keysDecompE += 'Egross_sigma','Egross_pi',
        if NetBond == True:
            AllTotalEnergies += Enet,
            AllDecompEnergies += Enetsig, Enetpi,
            keysTotalE += 'NetBond',
            keysDecompE += 'Enet_sigma','Enet_pi',
        if Composite == True:
            AllTotalEnergies += CompositeTable,
            keysTotalE += 'CompositeTable',
            
        # get the names of the keys, and bring arrays to its respective key
        DictionaryTotal = {key: value for key, value in zip(keysTotalE, AllTotalEnergies)}
        DictionaryDecomp = {key: value for key, value in zip(keysDecompE, AllDecompEnergies)}

        if TotalE == False:
            if sig_pi == True:
                return (DictionaryTotal, DictionaryDecomp) 
            else:
                return DictionaryTotal
        else:
            if sig_pi == True:
                return (DictionaryTotal, DictionaryDecomp, BEBOP) 
            else:
                return (DictionaryTotal, BEBOP) 
        
    def sort_bondE(self, Bond_Energies, rel=False, with_anti=False, with_number=True):
        """Sort the relative bond energies from strongest to weakest in energy.

        User can request not to have absolute by 'rel= False'.
        Also, user can request whether they would like relative or absolute anti-bonding energies.
        
        Parameters
        ----------
        Bond_Energies : :obj:`np.ndarray`
            Any bond energy array from bond_E(). This subroutine will not work for composite methods. 
        rel : obj:'bool', optional
            Return relative bond energies  
        with_anti : obj:'bool', optional
            Return antibonding energies
        with_number : obj:'bool', optional
            Put the atom number to distinguish the bond energy
        
        Returns
        -------
        :obj:`dict` or `np.ndarray`
            sort_bonds: The bonding (key: 'bonding') and/or antibonding(key:'antibonding') indentity in the molecule
            (i.e., gives the bond between two atoms with/without the atom number shown in the ROHF input)
        :obj:`dict` or :obj:`np.ndarray`
            sort_BE: Values of the bonding (key: 'bonding') and/or anti-bonding (key:'antibonding') arrays
            
        """
        
        size = self.mol.shape[0]
        if with_anti == False:
            sort_bonds = np.array([])
            sort_BE = np.array([])
        else:
            sort_bonds = {'bonding': np.array([]),
                          'antibonding': np.array([])
                         }

            sort_BE = {'bonding': np.array([]),
                       'antibonding': np.array([])
                      }
        if with_number == True: # user wants to distinguish the identity of the bond
            position = np.array(np.arange(1,size + 1), dtype=np.str) # indicate the position of the atoms 
            newAtoms = np.core.defchararray.add(self.mol, position) 
        else: # user does not want to distinguish bonds
            newAtoms = self.mol
        if with_anti == False:
            for l in range(1, size):
                for n in range(l):
                    if Bond_Energies[l][n] < 0:
                        if np.abs(Bond_Energies[l][n]) < 10: # this is for cases when the bond energy is small
                            continue
                        else:
                            sort_bonds = np.concatenate((sort_bonds,np.array([newAtoms[n] + '-' + newAtoms[l]])))
                            sort_BE = np.concatenate((sort_BE,np.array([Bond_Energies[l][n]])))
                    else:
                        continue
        else:
            for l in range(1, size):
                for n in range(l):
                    if Bond_Energies[l][n] < 0:
                        if np.abs(Bond_Energies[l][n]) < 1: # this is for cases when the bond energy is small
                            continue
                        else:
                            sort_bonds['bonding'] = np.concatenate((sort_bonds['bonding'],np.array([newAtoms[n] + '-' + newAtoms[l]])))
                            sort_BE['bonding'] = np.concatenate((sort_BE['bonding'],np.array([Bond_Energies[l][n]])))
                    elif Bond_Energies[l][n] > 0:
                        if np.abs(Bond_Energies[l][n]) < 1: # this is for cases when the bond energy is small
                            continue
                        else:
                            sort_bonds['antibonding'] = np.concatenate((sort_bonds['antibonding'],np.array([newAtoms[n] + '-' + newAtoms[l]])))
                            sort_BE['antibonding'] = np.concatenate((sort_BE['antibonding'],np.array([Bond_Energies[l][n]])))
                    else:
                        continue
        
        if with_anti == True:
            # Sort the bond energies and bonding index
            for l in np.array(['bonding','antibonding']):
                n = np.argsort(sort_BE[l])
                sort_BE[l] = np.sort(sort_BE[l])
                sort_bonds[l] = np.array([sort_bonds[l][t] for t in n])
        else:
            # Sort the bond energies and bonding index
            n = np.argsort(sort_BE)
            sort_BE = np.sort(sort_BE)
            sort_bonds = np.array([sort_bonds[t] for t in n])
        
        # Sort the bond type
        if rel == True:
            if with_anti == True:
                for l in np.array(['bonding','antibonding']):
                    sort_BE[l] = sort_BE[l] - sort_BE[l][0]   
            else:
                sort_BE = sort_BE - sort_BE[0]
            return (sort_bonds, sort_BE)
        else:
            return (sort_bonds, sort_BE)
        
    def resonance_E(
        self, Atom_Positions, CC_single=0.7874, CC_double=1.1958,
        CN_star=0.6941, CN_double=1.0699, double_CN=0.7644
    ):
        """Calculates the resonance energy of molecules in an aromatic ring, containing C-C, C=C,
           C-N*, C=N, and =C-N: bonds.
            
        Parameters
        ----------
        Atom_Positions : :obj:`dict`
            A dictionary containing the positions of the atoms in C-C, C=C,
            C-N*, C=N, and =C-N: as [(Atom1,Atom2),...,(AtomN,AtomM)]
        CC_single : obj:'int',optional
            The C-C reference bond order. The default value comes from
            1,3-butadiene twisted 90°
        CC_double : :obj:`float`,optional
            The C=C reference bond order. The default value comes from 
            cis-2-butene
        CN_star : :obj:`float`,optional
            The C-N* reference bond order. The default value comes from
            H2C=CH-N=CH2 twisted 90° 
        CN_double : :obj:`float`,optional
            The C=N reference bond order. The default value comes from
            cis-CH3CH=NCH3 
        double_CN : :obj:`float`, optional
            The =C-N: reference bond order. The default value comes from
            from planar vinyl amine, H2C=CH-NH2 
            
        Returns
        -------
        :obj:`np.float`
            res_E: The resonance energy in kcal/mol
        """
        
        ref = {'C-C':CC_single, # reference bond energies
               'C=C':CC_double,
               'C-N*': CN_star,
               'C=N':CN_double,
               '=C-N:':double_CN}
        
        Allkeys = np.array([]) # get all the keys
        BO_tot = np.array([]) # get the total amount of bond orders
        ref_BO = np.array([]) # get the reference BO
        num_BO = np.array([]) # number of unique bond orders
        bond_Atoms = np.array([]) # get the types of atoms in the bond

        for l in Atom_Positions.keys():
            Allkeys = np.concatenate((Allkeys,np.array([l])))
            ref_BO = np.concatenate((ref_BO, np.array([ref[l]])))
            if (l == 'C-C') or (l == 'C=C'):
                bond_Atoms = np.concatenate((bond_Atoms,np.array(['CC'])))
            elif (l == 'C-N*') or (l == 'C=N') or (l == '=C-N:'):
                bond_Atoms = np.concatenate((bond_Atoms,np.array(['CN'])))  

            n = 0
            allBO = 0 # sum the bond orders
            for Atom1, Atom2 in Atom_Positions[l]:
                allBO += 2 * self.mulliken[Atom1-1][Atom2-1] # Python n-1 rule for matrices applies here
                n += 1
                
            BO_tot = np.concatenate((BO_tot, np.array([allBO]))) 
            num_BO = np.concatenate((num_BO, np.array([n])))
            
        # dictionary containing the key, total number of unique bond orders, reference bond orders, and bonded atoms respectively   
        Data = {key: (total_bo, num_bo, ref_bo, bond_atoms) 
                for key, total_bo, num_bo, ref_bo, bond_atoms  
                in zip(Allkeys, BO_tot, num_BO, ref_BO, bond_Atoms)}
        
        res_E = bebop_eq.resonance(Data) # get resonance energy
        
        return res_E
    
    def strain_E(self, Atom_Positions, CC_single = 0.733556, CC_anti = -0.046488):
        """Calculate the strain energy of molecules in a free-strained ring, containing C-C bonding.
        
        The user must differentiate bonds participating in a three-membered ring (key: '3-ring') 
        and those that are not (key: 'normal')
        
        Parameters
        ----------
        Atom_Positions : :obj:`dict`
            A dictionary containing the atom numbers in a C-C three membered ring and/or 
            non-three membered rings as [(Atom1,Atom2),...,(AtomN,AtomM)]
        CC_single : :obj:`float`, optional
            The C-C reference bond order for strain energy. The default value comes from cyclohexane
        CC_anti : :obj:`float`, optional
            The C...C 1,3 anti-bonding bond order for strain energy. The default value comes from cyclohexane
            
        Returns
        -------
        :obj:`float`
            strain_DE: The train energy in kcal/mol
        """
        ref = {'3-ring':np.float(CC_single + 2 * CC_anti),'normal': CC_single} # {total bond order for a three membered ring,
                                                                               #  normal C-C bond for non-three membered ring}
        
        Allkeys = np.array([]) # get all the keys
        BO_tot = np.array([]) # get the total bond order 
        num_BO = np.array([]) # number of unique bond orders or 3-membered bond order
        ref_BO = np.array([]) # reference bond order 
        
        for l in Atom_Positions.keys():
            Allkeys = np.concatenate((Allkeys,np.array([l])))
            ref_BO = np.concatenate((ref_BO, np.array([ref[l]])))
            n = 0
            allBO = 0 # sum the bond orders

            for Atom1, Atom2 in Atom_Positions[l]:
                allBO += 2 * self.mulliken[Atom1-1][Atom2-1] # Python n-1 rule for matrices applies here 
                n += 1

            BO_tot = np.concatenate((BO_tot, np.array([allBO]))) 
            num_BO = np.concatenate((num_BO, np.array([n])))
            
        # dictionary containing the key, total number of unique bond orders (or 3-membered rings),
        # reference bond orders, and bonded atoms respectively
        Data = {key: (total_bo, num_bo, bond_ref) 
                for key, total_bo, num_bo, bond_ref  
                in zip(Allkeys, BO_tot, num_BO, ref_BO)}
        
        strain_DE = bebop_eq.strain(Data) # get resonance energy
        return strain_DE
