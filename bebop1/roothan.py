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

"""Script that computes Roothan's overlap integral equations and sigma/pi bond orders
   RoothanParameters :: zeta, tau, rho, kappa, rho_a, and rho_b parameters
   RoothanOverlapIntegrals :: overlap integrals for sigma and pi bonding between 2p orbitals
   SigmaPiBondOrders :: sigma and pi bond orders computed using parameters and overlap integrals
   
   Note: all equations and parameters came from Roothan's 1951 paper (see https://doi.org/10.1063/1.1748100)
   """
import numpy as np

def RoothanParameters(zeta1, zeta2, R):
    """Get the Roothan Parameters from https://doi.org/10.1063/1.1748100, Eq. 15 and Eq.16
    
       Parameters
       ----------
       zeta1: :obj:`float`
           The Slater parameter for Atom1
       zeta2: :obj:`float`
           The Slater parameter for Atom2
       R: :obj:`float`
           The distance between Atom1 and Atom2
       
       Returns
       -------
       zeta: :obj:`float`
           The average of the two Slater parameters
       tau: :obj:`float`
           The difference between two Slater parameters divided by the sum of the two Slater parameters
       rho: :obj:`float`
           The average of the two Slater parameters times R divided by the Bohr radius 
       kappa: :obj:`float`
           The sum of the squares of the two Slater parameters divided by the difference in squares of the two parameters
       rho_a,rho_b: :obj:`float`
           Slater parameter times R divided by the Bohr radius
       """
    
    zeta = (zeta1 + zeta2) / 2
    tau = (zeta1 - zeta2) / (zeta1 + zeta2)
    rho = (zeta1 + zeta2) * R / (2 * 0.52917706) # R is in units of Ansgtroms

    if zeta1 != zeta2:
        kappa = (zeta1**2 + zeta**2) / (zeta1**2 - zeta2**2)
    else:
        kappa = np.inf

    rho_a = zeta1 * R / 0.52917706 # R is in units of Angstroms
    rho_b = zeta2 * R / 0.52917706 # R is in units of Angstroms
    return (zeta, tau, rho, kappa, rho_a, rho_b)

def RoothanOverlapIntegrals(R, Elements):
    """Calculate the Roothan overlap integrals from https://doi.org/10.1063/1.1748100, 
       Eq.25 and Eq.25a
       Note that the equations used are: (2p \sigma_a|2p \sigma_b) and (2p \pi_a|2p \pi_b)
       
       Parameters
       ----------
       R: :obj:`np.float`
           Distance between two atoms 
       nAtoms: :obj:`np.ndarray`
           Array of atoms present in the molecule
       
       Returns
       -------
       S_sigma: :obj:`np.float`
           The overlap integral for the sigma bond
       S_pi (type: float)
           The overlap integral for the pi bond
       """
    
    Slater = {'H':1.23, # Slater 2s-2p exponents to fit STO-6G minimal atomic basis sets
              'He':1.67,
              'Li':0.80,
              'Be':1.15,
              'B':1.50,
              'C':1.72,
              'N':1.95,
              'O':2.25,
              'F':2.55
              } 
    zeta, tau, rho, kappa, rho_a, rho_b = RoothanParameters(Slater[Elements[0]],Slater[Elements[1]],R)
    if Elements[0] == Elements[1]:
        Ssigma = (-1 - rho - 1/5 * rho**2 + 2/15 * rho**3 + 1/15 * rho**4) * np.exp(-rho)
        Spi = (1 + rho + 2/5 * rho**2 + 1/15 * rho**3) * np.exp(-rho)
    else:
        u1 = 1 - kappa
        u2 = 1 + kappa
        u3 = 5 + 6 * kappa
        u4 = 5 - 6 * kappa
        term1 = -u1**2 * (48 * u2**2 * (1 + rho_a + 1/2 * rho_a**2) + 2 * u3 * rho_a**3 + 2 * rho_a**4) * np.exp(-rho_a)
        term2 = u2**2 * (48 * u1**2 * (1 + rho_b + 1/2 * rho_b**2) + 2 * u4 * rho_b**3 + 2 * rho_b**4) * np.exp(-rho_b)
        Ssigma = (1/(np.sqrt(1 - tau**2) * tau * rho**3)) * (term1 + term2)
        term3 = -u1**2 * (24 * u2 **2 * (1 + rho_a) + 12 * u2 * rho_a**2 + 2 * rho_a**3) * np.exp(-rho_a)
        term4 = u2**2 * (24 * u1 ** 2 * (1 + rho_b) + 12 * u1 * rho_b**2 + 2 * rho_b**3) * np.exp(-rho_b)
        Spi = (1/(np.sqrt(1 - tau**2) * tau * rho**3)) * (term3 + term4)
    return (Ssigma,Spi)  
    
def SigmaPiBondOrders(CiCjMatrix, PopMatrix, Mulliken, Trig, DistanceMatrix, NBFN, nAtoms):
    """Calculate the sigma and pi bond orders.
    
       Parameters
       ----------
       CiCjMatrix: :obj:`np.ndarray`
           Total CiCj (\alpha+\beta) Population MBS condensed to orbitals
       PopMatrix: :obj:`np.ndarray`
           Mulliken Population MBS matrix condensed to orbitals
       Mulliken: :obj:`np.ndarray`
           Mulliken MBS bond orders condensed to atoms
       Trig: obj:'tuple'
           Values of the trig functions in the X,Y, and Z directions
       DistanceMatrix: :obj:`np.ndarray`
           Array containing the distance matrixes 
       NBFN: :obj:`np.ndarray`
           Array containing the indexes where the orbitals are located in CiCjMatrix and PopMatrix
       nAtoms: :obj:`np.ndarray`
           Array of atoms present in the molecule 
       
       Returns
       -------
       SigmaBondOdr: :obj:`np.ndarray`
           Sigma bond order matrix
       PiBondOdr: :obj:`np.ndarray`
           Pi bond order matrix"""
    
    CosX,SinX,CosY,SinY,CosZ,SinZ = Trig
    size = nAtoms.shape[0] # set the dimension of the Sigma and Pi bond orders
    SigmaBondOdr = np.zeros((size,size),dtype=np.float64)
    PiBondOdr = np.zeros((size,size),dtype=np.float64)
    
    # Compute Bond Orders
    for i in range(1,size):
        for j in range(i):
            # Bond order (1s,1s)
            SigmaBondOdr[i][j] = PopMatrix[NBFN[i]][NBFN[j]]

            # Bond order(1s,2s-2px-2py-2pz)
            if (nAtoms[j] not in np.array(['H','He'])):
                n = 1
                while n != 5:
                    SigmaBondOdr[i][j] += PopMatrix[NBFN[i]][NBFN[j] + n]
                    n += 1

            # Bond order(2s-2px-2py-2pz,1s)      
            if (nAtoms[i] not in np.array(['H','He'])): 
                n = 1
                while n != 5:
                    SigmaBondOdr[i][j] += PopMatrix[NBFN[i] + n][NBFN[j]] 
                    n += 1
            if (nAtoms[j] in np.array(['H','He'])) or (nAtoms[i] in np.array(['H','He'])):
                continue
            else:       
                # Bond order(2s,2s) 
                SigmaBondOdr[i][j] += PopMatrix[NBFN[i] + 1][NBFN[j] + 1]
                
                # Bond order(2s,2px-2py-2pz)   
                n = 2
                while n != 5:
                    SigmaBondOdr[i][j] += PopMatrix[NBFN[i] + 1][NBFN[j] + n]
                    n += 1
                    
                # Bond order(2px-2py-2pz,2s)
                n = 2
                while n != 5:
                    SigmaBondOdr[i][j] += PopMatrix[NBFN[i] + n][NBFN[j] + 1]
                    n += 1
                            
            SumProducts1 = 0 # dummy variables used for the sum of the projection angles for sigma bond orders
            SumProducts2 = 0 # dummy variable used for the sum of the projection angles for the pi bond orders
            Ssigma,Spi = RoothanOverlapIntegrals(DistanceMatrix[i][j],np.array([nAtoms[i],nAtoms[j]]))
            SumProducts1 += CosX[i][j] * CosX[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 2]
            SumProducts1 += CosX[i][j] * CosY[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 3]
            SumProducts1 += CosX[i][j] * CosZ[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 4]
            SumProducts1 += CosY[i][j] * CosX[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 2]
            SumProducts1 += CosY[i][j] * CosY[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 3]
            SumProducts1 += CosY[i][j] * CosZ[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 4]
            SumProducts1 += CosZ[i][j] * CosX[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 2]
            SumProducts1 += CosZ[i][j] * CosY[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 3]
            SumProducts1 += CosZ[i][j] * CosZ[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 4]
            SigmaBondOdr[i][j] -= Ssigma * SumProducts1

            SumProducts2 += SinX[i][j] * SinX[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 2]
            SumProducts2 -= SinX[i][j] * SinY[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 3]
            SumProducts2 -= SinX[i][j] * SinZ[i][j] * CiCjMatrix[NBFN[i] + 2][NBFN[j] + 4]
            SumProducts2 -= SinY[i][j] * SinX[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 2]
            SumProducts2 += SinY[i][j] * SinY[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 3]
            SumProducts2 -= SinY[i][j] * SinZ[i][j] * CiCjMatrix[NBFN[i] + 3][NBFN[j] + 4]
            SumProducts2 -= SinZ[i][j] * SinX[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 2]
            SumProducts2 -= SinZ[i][j] * SinY[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 3]
            SumProducts2 += SinZ[i][j] * SinZ[i][j] * CiCjMatrix[NBFN[i] + 4][NBFN[j] + 4]
            PiBondOdr[i][j] = Spi * SumProducts2
                
            # Renormalize Sigma and Pi Bond Orders
            if np.abs(SigmaBondOdr[i][j] + PiBondOdr[i][j]) < 0.0001:
                continue
            else:
                ReNormij = Mulliken[i][j] / (SigmaBondOdr[i][j] + PiBondOdr[i][j])
                PiBondOdr[i][j] = ReNormij * PiBondOdr[i][j]
                SigmaBondOdr[i][j] = ReNormij * SigmaBondOdr[i][j]
                
    # Add bonds for the upper diagonal elements for the sigma and pi bond orders
    SigmaBondOdr += SigmaBondOdr.T
    PiBondOdr += PiBondOdr.T
       
    return (SigmaBondOdr,PiBondOdr)
