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


"""Main subroutine that does all calculation
   total_E:: compute the BEBOP-2 electronic atomization energy
   bond_E:: compute bond energy tables
   sort_bondE:: sort the bond energies in order
   ionic_character:: calculate the percent of ionicty of a bond
   resonance_E:: calculate resonance energy for C~C and C~N bond
   strain_E:: calculate the strain energy for any C~C bond
   """

from . import bebop2_equation as bebop_eq 
from . import read_output as ro
from . import roothan 
from . import spatial_geom as sg
import numpy as np

class BEBOP: # BEBOP class
    
    def __init__(self, name, parameter_folder='opt_parameters'):
        """Give the name of the ROHF output file to get the bond energy data.
        
        Parameters
        ----------
        name: :obj:'str'
            Name of the ROHF/CBSB3 output file
        parameter_folder: :obj:'str', optional
            Name of the parameter folder
        """
        
        if parameter_folder[-1] != '/':
            self.parameter_folder = parameter_folder + '/'
        else:
            self.parameter_folder = parameter_folder 
        
        self.name = name # name(or path+name) of the Gaussian output file
        nAtoms, XYZ, CiCjAlpha, CiCjBeta, PopMatrix, NISTBF, Occ2s, Mulliken, charges = ro.read_entire_output(name) # get all data 
                                                                                                                    # from the output file
        CiCj = CiCjAlpha + CiCjBeta # this is the sum of the population matrix
        self.mol = nAtoms # array containing the elements within the molecule
        self.mol2s = Occ2s # array or float of molecular occupation 2s
        self.mulliken = Mulliken # Mulliken MBS bond orders condensed to atoms
        self.CiCj = CiCj # this is the sum of the population matrix
        self.pop = PopMatrix # Mulliken MBS population matrix condenced to orbitals
        self.elec_Occ = NISTBF # array showing where the orbitals begin and end for each atom (important for sigma and pi bonding)
        distance_matrix, angles = sg.Spatial_Properties(XYZ) # returns the distance matrix and the projection angles
        self.distance_matrix = distance_matrix # save the distance matrix
        self.angles = angles # save the projection angles
        self.charges = charges # save the charges for each species
        input_data = (self.CiCj, self.pop, self.mulliken, self.angles, self.distance_matrix, self.elec_Occ, self.mol)
        sigma, pi = roothan.sigma_pi_bond_orders(*input_data) # get the sigma and pi bond orders
        self.bo_sig = sigma
        self.bo_pi = pi
        return None
    
    def total_E(self):
        """Compute the total atomization energy in kcal/mol
        
        Output
        ------
        self.bebop2_elec: :obj:'np.float'
            Total bebop2 electronic atomization energy 
            
        """
        
        data = bebop_eq.bebop2(self.distance_matrix,self.mol,self.charges, 
                               self.mol2s, self.bo_sig, self.bo_pi, self.parameter_folder)
        
        self.all_data = data
        self.bebop2_elec = data[0]
        self.E_cov_sig = data[1]
        self.E_cov_pi = data[2]
        self.E_rep_sig = data[3]
        self.E_rep_pi = data[4]
        self.E_fluc = data[5]
        self.E_coulomb = data[6]
        self.E_hyb = data[7]
        return self.bebop2_elec
        
    def bond_E(self, NetBond = True, GrossBond = False, Composite = False, sig_pi = False):
        """Compute all bond energies in kcal/mol.All energies are printed by default.
           User may select which energies will be returned by selecting 'True'. 
           
           New users are advise to use the '.keys()' method on the output variables to check 
           the name of the keys in the dictionaries.
           
           
           Parameters
           ----------
           NetBond: :obj:'bool',optional
               Generate the bond energy to include repulsion and hybridization corrections only
               Returns net covalent bond energies (Enet) if 'NetBond = True' and the net pi covalent bond
               energies (Enetpi) and the net sigma bond energies (Enetsig) if 'sig_pi = False'. 
           GrossBond: :obj:'bool', optional
               Generate the bond energy to include repulsion corrections only 
               Returns covalent bond energies (Ecov) if 'GrossBond = True' and the gross pi covalent bond
               energies (Epi) and the gross sigma bond energies (Esig) if 'sig_pi = False'. 
           Composite: :obj:'bool', optional
               Return the composite table (CompositeTable) containing the net bond energies (lower diagonal elements), 
               the hybridization energies (diagonal elements), and the gross bond energies (upper diagonal elements)
           sig_pi: :obj:'bool', optional
               Return sigma and pi bonding energy decompositions for Enet and Ecov. 
               If 'sig_pi = True', output will be a tuple (first DictionaryTotal and second DictionaryDecomp)
           
           Returns
           -------
           DictionaryTotal: :obj:'dict'
               Dictionary containing Enet (key: 'NetBond'), Ecov (key: 'GrossBond'), 
               and CompositeTable (key: 'Composite')
           DictionaryDecomp: :obj:'dict', optional
                Dictionary containing Esig (key: 'Egross_sigma'), Epi (key: 'Egross_pi'), 
                Enetsig (key: 'Enet_sigma'), and Enetpi (key: 'Enet_pi')
           BEBOP: :obj:'np.float64', optional
               Return the BEBOP total atomization energy with ZPVE at 0 K
               
           """
        AllTotalEnergies = []
        AllDecompEnergies = []
        keysTotalE = []
        keysDecompE = []
        Data = bebop_eq.bebop2_bond_energy(*self.all_data) # get all the data
        self.E_sig = Data[0] 
        self.E_pi = Data[1]
        self.E_gross = Data[2]
        self.E_gross_charge = Data[3]
        self.E_net_sig = Data[4]
        self.E_net_pi = Data[5]
        self.E_net = Data[6]
        self.CompositeTable = Data[7] 
        self.E_charge = Data[8]
        
        # store all of the data in a list
        if GrossBond == True:
            AllTotalEnergies += self.E_gross, self.E_gross_charge, 
            AllDecompEnergies += self.E_sig, self.E_pi, self.E_coulomb, self.E_fluc, self.E_charge,
            keysTotalE += 'GrossBond', 'GrossChargeBond'
            keysDecompE += 'Egross_sigma','Egross_pi', 'E_coulomb', 'E_fluc', 'E_charge',
        
        if NetBond == True:
            AllTotalEnergies += self.E_net,
            AllDecompEnergies += self.E_net_sig, self.E_net_pi, 
            keysTotalE += 'NetBond',
            keysDecompE += 'Enet_sigma','Enet_pi',
            
        if Composite == True:
            AllTotalEnergies += self.CompositeTable,
            keysTotalE += 'CompositeTable',
            
        # get the names of the keys, and bring arrays to its respective key
        
        DictionaryTotal = {key: value for key, value in zip(keysTotalE, AllTotalEnergies)}
        DictionaryDecomp = {key: value for key, value in zip(keysDecompE, AllDecompEnergies)}
        if sig_pi == True:
            return (DictionaryTotal, DictionaryDecomp) 
        else:
            return DictionaryTotal 
        
    def ionic_character(self):
        """Compute the ionic character of each bond and molecule
        
        Returns
        -------
        molecule_ionic: :obj:'numpy.darray'
            Overall molecular ionicity
        bonds_ionic: :obj:'numpy.darray'
            Ionicity per each bond in molecule
        """
        
        ionic_contr = self.E_coulomb # terms reflecting ionic effects (neglect hybridization) 
        molecule_ionic = np.sum(ionic_contr) / self.bebop2_elec * 100 # percentage of ionicity for molecule
        if molecule_ionic < 0: #negative % ionicity = no ionic
            molecule_ionic = 0
        bonds_ionic = np.divide(ionic_contr, self.E_net, out = np.zeros_like(ionic_contr), where= self.E_net!= 0) * 100 # percentage of ioniocity per bonds
        bonds_ionic[bonds_ionic < 0] = 0 #negative % bond ionicity = no ionic
        return (molecule_ionic, bonds_ionic)
        
    def sort_bondE(self,Bond_Energies,rel= False, with_anti= False, with_number= True):
        """Sort the relative bond energies from strongest to weakest in energy.
           User can request not to have absolute by 'rel= False'.
           Also, user can request whether they would like relative or absolute anti-bonding energies.
           
           Parameters
           ----------
           Bond_Energies: :obj:'np.ndarray'
               Any bond enery array from bond_E(). This subroutine will not work for composite methods. 
           rel: :obj:'bool', optional
               Return relative bond energies  
           with_anti: :obj:'bool', optional
               Return antibonding energies
           with_number: :obj:'bool', optional
               Put the atom number to distinguish the bond energy
           
           Returns
           -------
           sort_bonds: :obj:'dict' or 'np.ndarray'
                The bonding (key: 'bonding') and/or antibonding(key:'antibonding') indentity in the molecule
                (i.e., gives the bond between two atoms with/without the atom number shown in the ROHF input)
           sort_BE: :obj:'dict' or np.darray
                Values of the bonding (key: 'bonding') and/or anti-bonding (key:'antibonding') arrays
                
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
        else: # user does not want to distinguish bons
            newAtoms = self.mol
        if with_anti == False:
            for l in range(1, size): 
                for n in range(l):
                    if Bond_Energies[l][n] < 0:
                        if np.abs(Bond_Energies[l][n]) < 1: # this is for cases when the bond energy is small
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
        
    def resonance_E(self,Atom_Positions, CC_single = [0.766595, 0.023907], CC_double = [0.811649, 0.386923],
                    CN_star = [0.640604, 0.052418], CN_double = [0.773745, 0.305153], double_CN = [0.686184, 0.060264]):
        """Calculates the resonance energy of molecules in an aromatic ring, containing C-C, C=C,
           C-N*, C=N, and =C-N: bonds.
            
        Parameters
        ----------
        Atom_Positions: :obj:'dict'
            A dictionary containing the positions of the atoms in C-C, C=C,
            C-N*, C=N, and =C-N: as [(Atom1,Atom2),...,(AtomN,AtomM)]
        CC_single: :obj:'int',optional
            The C-C reference bond order (array formulated as [sigma BO, pi BO]). 
            The default value comes from 1,3-butadiene twisted 90° 
        CC_double: :obj:'float',optional
            The C=C reference bond order (array formulated as [sigma BO, pi BO]). 
            The default value comes from cis-2-butene
        CN_star: :obj:'float',optional
            The C-N* reference bond order (array formulated as [sigma BO, pi BO]). 
            The default value comes from H2C=CH–N=CH2 twisted 90° 
        CN_double: :obj:'float',optional
            The C=N reference bond order (array formulated as [sigma BO, pi BO]). 
            The default value comes from cis-CH3CH=NCH3 
        double_CN: :obj:'float', optional
            The =C-N: reference bond order (array formulated as [sigma BO, pi BO]).
            The default value comes from from planar vinyl amine, H2C=CH-NH2             
        Returns
        -------
        res_E: :obj:'np.float'
            The resonance energy in kcal/mol
        """        
        ref_sigma = {'C-C':CC_single[0],
                     'C=C':CC_double[0],
                     'C-N*': CN_star[0],
                     'C=N':CN_double[0],
                     '=C-N:':double_CN[0]}
        
        ref_pi = {'C-C':CC_single[1],
                  'C=C':CC_double[1],
                  'C-N*': CN_star[1],
                  'C=N':CN_double[1],
                  '=C-N:':double_CN[1]} 
        
        Allkeys = np.array([]) # get all the keys
        BO_sigma = np.array([]) # unreferenced sigma BO
        BO_pi = np.array([]) # unreferenced pi BO
        ref_BO_sig = np.array([]) # reference sigma BO
        ref_BO_pi = np.array([]) # reference pi BO
        num_BO = np.array([]) # number of unique bond orders
        bond_Atoms = np.array([]) # get the types of atoms in the bond
        for l in Atom_Positions.keys():
            Allkeys = np.concatenate((Allkeys,np.array([l])))
            ref_BO_sig = np.concatenate((ref_BO_sig, np.array([ref_sigma[l]])))
            ref_BO_pi = np.concatenate((ref_BO_pi, np.array([ref_pi[l]])))
            if (l == 'C-C') or (l == 'C=C'):
                bond_Atoms = np.concatenate((bond_Atoms,np.array(['CC'])))
            elif (l == 'C-N*') or (l == 'C=N') or (l == '=C-N:'):
                bond_Atoms = np.concatenate((bond_Atoms,np.array(['CN'])))  
            n = 0
            allBO_sig = 0 # sum of the sigma bond orders
            allBO_pi = 0 # sum of the pi bond orders 
            for Atom1, Atom2 in Atom_Positions[l]: 
                allBO_sig += 2 * self.bo_sig[Atom1-1][Atom2-1] 
                allBO_pi += 2 * self.bo_pi[Atom1-1][Atom2-1]
                n += 1
            BO_sigma = np.concatenate((BO_sigma, np.array([allBO_sig]))) 
            BO_pi = np.concatenate((BO_pi, np.array([allBO_pi]))) 
            num_BO = np.concatenate((num_BO, np.array([n])))

      # create a dictionary containing the key, total number of unique bond orders, reference bond orders, and bonded atoms respectively
        Data = {key: (bo_sigma, bo_pi, num_bo, ref_bo_sigma, ref_bo_pi, bond_atoms) 
                for key, bo_sigma, bo_pi, num_bo, ref_bo_sigma, ref_bo_pi, bond_atoms  
                in zip(Allkeys, BO_sigma, BO_pi, num_BO, ref_BO_sig, ref_BO_pi, bond_Atoms)}
        
        res_E = bebop_eq.resonance(Data, self.parameter_folder) # get resonance energy
        return res_E
    
    def strain_E(self, Atom_Positions, CC_single = [0.658778, 0.076644], CC_anti = [-0.045618, -0.00149]):
        """Calculate the strain energy of molecules in a free-strained ring, containing C-C bonding.
           The user must differentiate bonds participating in a three-membered ring (key: '3-ring') 
           and those that are not (key: 'normal')
           
           Parameters
           ----------
           Atom_Positions: :obj:'dict'
               A dictionary containing the atom numbers in a C-C three membered ring and/or 
               non-three membered rings as [(Atom1,Atom2),...,(AtomN,AtomM)]
           CC_single: :obj:'float', optional
                The C-C reference bond order for strain energy. The default value comes from cyclohexane.
           CC_anti: :obj:'float', optional
               The C...C 1,3 anti-bonding bond order for strain energy. The default value comes from cyclohexane.
               
           Returns
           -------
           strain_DE: :obj:'float', optional
               The train energy in kcal/mol
           """
        
        ref_sig = {'3-ring_sigma':np.float64(CC_single[0] + 2 * CC_anti[0]), # {sigma bond order for a three-membered ring, 
                   'normal_sigma': CC_single[0]}                             #  normal C-C sigma bond for non-three membered ring}
        ref_pi = {'3-ring_pi':np.float64(CC_single[1] + 2 * CC_anti[1]),           #  {normal C-C pi bond for three membered ring,
                  'normal_pi':CC_single[1]}                                        #   normal C-C pi bond for non-three membered ring}
        Allkeys = [] # get all the keys
        BO_sig_tot = [] # sigma bond order
        BO_pi_tot = [] # pi bond order
        num_BO = [] # number of unique bond orders or 3-membered bond order
        BO_ref_sig = [] # reference sigma bond order 
        BO_ref_pi = [] # reference pi bond order
        for l in Atom_Positions.keys():
            Allkeys += l,
            BO_ref_sig += ref_sig[f'{l}_sigma'],
            BO_ref_pi += ref_pi[f'{l}_pi'],
            n = 0
            BO_sig_all = 0
            BO_pi_all = 0
            for Atom1, Atom2 in Atom_Positions[l]:
                BO_sig_all += 2 * self.bo_sig[Atom1-1][Atom2-1]
                BO_pi_all += 2 * self.bo_pi[Atom1-1][Atom2-1]
                n += 1
            num_BO = n, 
            BO_sig_tot += BO_sig_all,
            BO_pi_tot += BO_pi_all,
        # Create numpy arrays
        Allkeys = np.array(Allkeys)
        BO_sig_tot = np.array(BO_sig_tot)
        BO_pi_tot = np.array(BO_pi_tot)
        num_BO = np.array(num_BO)
        BO_ref_sig = np.array(BO_ref_sig)
        BO_ref_pi = np.array(BO_ref_pi)
        # create a dictionary containing the key, total number of unique bond orders (or 3-membered rings),
        # reference bond orders, and bonded atoms respectively
        Data = {key: (bo_sig, bo_pi, num_bo, bond_ref_sig, bond_ref_pi) 
                for key, bo_sig, bo_pi, num_bo, bond_ref_sig, bond_ref_pi  
                in zip(Allkeys, BO_sig_tot, BO_pi_tot, num_BO, BO_ref_sig, BO_ref_pi)}
        strain_DE = bebop_eq.strain(Data, self.parameter_folder) # get resonance energy
        return strain_DE
