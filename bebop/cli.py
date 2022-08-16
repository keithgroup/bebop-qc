#!/usr/bin/env python3

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

"""Argument parser used for computing and printing BEBOP-1 results 

parsing:: parser for computing BEBOP-1
file_properties:: class contianing properties for writing the files
print_be:: print the bond energy tables
write_json:: write the json file (if user wishes to)
print_file:: print the BEBOP output file
"""

import argparse
import json
import os
from datetime import date
from datetime import datetime
import numpy as np
from .bebop1 import BEBOP

def parsing():
    """Parsers for BEBOP-2"""
    parser = argparse.ArgumentParser(description = 'compute BEBOP atomization energies and bond energies (i.e., gross and net)')
    parser.add_argument('-f', action = 'store', help = 'name of the Gaussian Hartree-Fock output file', required = True, type = str)
    parser.add_argument('--be', action = 'store_true', help = 'compute BEBOP bond energies (net and gross bond energies)', required=False)
    parser.add_argument('--sort', action = 'store_true', help = 'sort the net BEBOP bond energies (from lowest to highest in energy)', required=False)
    parser.add_argument('--json',action = 'store_true', help = 'save the job output into JSON', required=False)
    return parser.parse_args()

class file_properties:
    """Class that contains all methods used to write the files"""
    
    def horizontal_numbers(atoms, previous):
        """Print the atomic numbers for the bond energy tables"""
        
        print('',end= 10 * ' ')
        new_atoms = atoms[previous:]
        size = new_atoms.shape[0]
        if size < 5:
            for l in range(size):
                previous += 1
                if (size - 1) != l:
                    end = ' '
                else:
                    end = '\n'
                print(f'{previous:>10}',end=end)
        elif size >= 5:
            for l in range(5):
                previous += 1
                if 4 != l:
                    end = ' '
                else:
                    end = '\n'
                print(f'{previous:>10}',end=end)
       
        return None
    
    def lower_diagonal(bond_energies, atoms, previous):
        """Print elements of the composite tables and 
           move the columns after five printed elements"""
        
        new_atoms = atoms[previous:]
        skip = 0
        vert = previous
        size = new_atoms.shape[0]
        new_be = bond_energies[previous:,previous:]
        if size > 5:
            previous += 5
        else:
            previous += size
        for l in range(size):
            vert += 1
            if skip == 5:
                skip += 0
            else:
                skip += 1
            print(f'    {vert:>2}  {new_atoms[l]:>4}',end='')
            for j in range(skip):
                if j == (skip-1):
                    end = '\n'
                else:
                    end = '  '
                print(f'{new_be[l][j]:>9.2f}',end=end)
            
        return previous
        
    def composite_tables(bond_energies, atoms, previous):
        """Print elements of the composite tables and move the columns after five are used"""
        
        new_atoms = atoms[previous:]
        skip = 0
        vert = previous
        size = new_atoms.shape[0]
        new_be = bond_energies[:,previous:]
        vert = previous
        if size > 5:
            previous += 5
            hort = 5
        else:
            hort = size
            previous += size
        for l in range(atoms.shape[0]):
            vert += 1
            print(f'    {l+1:>2}  {atoms[l]:>4}',end='')
            for j in range(hort):
                if (hort-1) == j:
                    end = '\n'
                else:
                    end = '  '
                print(f'{new_be[l][j]:>9.2f}',end=end)
                
        return previous 
    
    def print_file_title(file, energy):
        """Print the title of the file"""
        
        # Initiate the string for the current month, day,  and year this code was executed 
        today = date.today()
        day_month_year = today.strftime("%d-%B-%Y")

        # Initiate the string for the current time 
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        print(f"\n\n\n{20 * ' '}SUMMARY OF BEBOP CALCULATION\n")
        print(f"{24 * ' '}BEBOP (Version 1.0.0)")
        print(f"{23 * ' '}{day_month_year} {current_time}\n\n")
        print(f'   HARTREE-FOCK OUTPUT:  {os.getcwd()}/{file}\n\n')

        # Print the total BEBOP atomization energy
        print(f'   BEBOP ATOMIZATION ENERGY (0 K)     =     {energy} KCAL/MOL\n\n')
        
        return None
    
    def sort_be_tables(sort_bonds, sort_be):
        """Print the values of the bond and the identies of the bond, 
           from lowest to highest in energy"""
        
        print(f'  SORTED NET BONDING ENERGIES (LOWEST TO HIGHEST)\n')
        print(f'  ----------------------------')
        print(f'     BOND       BOND ENERGIES ')
        print(f'   IDENTITY       (KCAL/MOL)  ')
        print(f'  ----------------------------')
        for l in range(sort_bonds.shape[0]):
            print(f'  {sort_bonds[l]:>9}         {sort_be[l]:<5.2f}')
        print(f'  ----------------------------')
        print('') 
        
        return None
    
    def merge_dict(dictionary, name_of_table, values):
        """Merge dictionary for json files"""
    
        if name_of_table == 'bond energies':
            total_bond,sigma_pi = values
            new_values = {'bond energies':{'gross':{'sigma':sigma_pi['Egross_sigma'],
                                                   'pi':sigma_pi['Egross_pi'],
                                                   'total':total_bond['GrossBond']
                                                   },
                                          'net':{'sigma':sigma_pi['Enet_sigma'],
                                                 'pi':sigma_pi['Enet_pi'],
                                                 'total':total_bond['NetBond']
                                                   },
                                          'composite table':total_bond['CompositeTable']}}

        elif name_of_table == 'sorted net bond energies':
            sort_bonds,sort_be = values
            new_values = {'sorted net bond energies':{'sorted bonds':sort_bonds,
                                                 'bond energies':sort_be}}
        
        new_dict = {**dictionary, **new_values}

        return new_dict
    
def print_be(name_of_table, bond_energies, atoms, diagonals=0):
    """Print the bond energy tables with respect to the bond energies"""
    
    titles = {'Egross_sigma':'Gross Sigma Bond Energies including Repulsion',
              'Egross_pi':'Gross Pi Bond Energies including Repulsion',
              'GrossBond':'Gross Total Bond Energies including Repulsion',
              'Enet_sigma':'Net Sigma Bond Energies including Hybridization',
              'Enet_pi':'Net Pi Bond Energies including Hybridization',
              'NetBond':'Net Total Energies including Hybridization',
              'CompositeTable':'Composite Table:    Eii(hybrid)  Eij(gross)\n' + 23 * ' ' + 'Eji(net)     Ejj(hybrid)'
              }
    
    energy = {'Egross_sigma':'Total Gross Sigma Energy',
              'Egross_pi':'Total Gross Pi  Energy',
              'GrossBond':'Total Gross Energy',
              'Enet_sigma':'Total Net Sigma Energy',
              'Enet_pi':'Total Net Pi Bond Energy',
              'NetBond':'Total Net Energy'
              }

    # print the title of the header
    print(f"   {titles[name_of_table].upper()}\n",end='')

    # Print the values of the elements of the matrix
    if name_of_table in np.array(['Egross_sigma','Egross_pi','Enet_sigma','Enet_pi','GrossBond','NetBond']):
        previous = 0
        while previous != atoms.shape[0]:
            file_properties.horizontal_numbers(atoms,previous)
            previous = file_properties.lower_diagonal(bond_energies,atoms,previous)
        print('\n')
        print(f'   {energy[name_of_table].upper()}     =     {np.sum(bond_energies):0.2f} KCAL/MOL',end='')
        
    elif name_of_table == 'CompositeTable':
        previous = 0
        while previous != atoms.shape[0]:
            file_properties.horizontal_numbers(atoms,previous)
            previous = file_properties.composite_tables(bond_energies,atoms,previous)
        print('\n')
        print(f'   TOTAL HYBRIDIZATION ENERGY     =     {np.sum(diagonals):0.2f} KCAL/MOL',end='')
    
    print('\n\n')

    return None

def write_json(dictionary):
    """Write the json file"""
    
    class NumpyJSONEncoder(json.JSONEncoder):
        """Encode numpy.ndarrays() to a list object."""
        def default(self,obj):
            if isinstance(obj,np.ndarray):
                return [obj[i].tolist() for i in range(obj.shape[0])]
            return json.JSONEncoder.default(self, obj) 
        
    with open("bopout.json", mode='w') as writer:
        json.dump(dictionary,writer, indent = 4, cls = NumpyJSONEncoder)
            
    return None 
    
def print_file(parser_args):
    """Write the output file"""
    
    # Get information from the the ROHF output file
    data = BEBOP(parser_args.f)

    # Print the title, BEBOP version, date, time, name or file, and energy
    file_properties.print_file_title(parser_args.f,data.total_E())

    if parser_args.json == True:
        dictionary = {'method':'BEBOP',
                      'version':'1.0.0',
                      'HF output file':os.getcwd() + '/' + parser_args.f,
                      'date':date.today().strftime("%d-%B-%Y"),
                      'time':datetime.now().strftime("%H:%M:%S"),
                      'atomization energy':data.total_E()
                      }

    if parser_args.be == True:
        total_bond, sigma_pi = data.bond_E(NetBond = True, GrossBond = True, Composite = True, sig_pi = True)
        for l in np.array(['Egross_sigma','Egross_pi','GrossBond','Enet_sigma','Enet_pi','NetBond','CompositeTable']):
            if l in np.array(['CompositeTable','GrossBond','NetBond']):
                print_be(l,total_bond[l],data.mol,diagonals=total_bond[l].diagonal()) # print net, gross bond energies, and composite tables
            else:
                print_be(l,sigma_pi[l],data.mol) # print sigma and pi bond energies (for gross and net)
        if parser_args.json == True: # append values to json file 
            dictionary = file_properties.merge_dict(dictionary,'bond energies',(total_bond,sigma_pi))
    

    if parser_args.sort == True: # sort the bond energies
        if parser_args.be != True:
            total_bond = data.bond_E()
        sort_bonds, sort_be = data.sort_bondE(total_bond['NetBond'])
        file_properties.sort_be_tables(sort_bonds, sort_be)
        if parser_args.json == True:
            dictionary = file_properties.merge_dict(dictionary,'sorted net bond energies',(sort_bonds,sort_be))
            
    if parser_args.json == True:
        write_json(dictionary)

    return None 

def main():
    pars_arg = parsing()
    print_file(pars_arg)
    return None

if __name__ == "__main__":
    main()
