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

"""Compute the distance matrix and the trig. projection functions

MakeSureZero :: makes the cosine equal to zero if it is less than 0.00001
CalculateSin :: calculate the sine of the function
Spatial_Properties :: Compute the distance matrix and angles
"""

import numpy as np

def MakeSureZero(CosTheta):
    """If the absolute CosTheta < 0.0001, then make it zero.
    
    Parameters
    ----------
    CosTheta : obj:`np.float64`
        Value of \cos(\theta)
        
    Returns
    -------
    CosTheta : obj: 'np.float64'
        Correct value of \cos(\theta)
    """ 
    if np.abs(CosTheta) < 0.00001:
        CosTheta = np.float64(0)
    return CosTheta

def CalculateSin(CosTheta):
    """Calculate the sine of the theta using the value of \cos(\theta)

    Note this subroutine was created to resemble -sign(A,B) in FORTRAN.
    
    Parameters
    ----------
    CosTheta : obj:`np.float64`
        Value of \cos(\theta)
        
    Returns
    -------
    :obj:`np.float64`
        -SinTheta: Value of the opposite sign of \sin(\theta)
    """
    SinTheta = np.sqrt(1 - CosTheta**2)
    if CosTheta < np.float64(0):
        SinTheta *= -1
    elif CosTheta >= np.float64(0):
        SinTheta *= 1
    return -SinTheta

def Spatial_Properties(XYZ):
    """Compute the distance matrix and the trig. projection functions
    
    Parameters
    ----------
    XYZ: :obj:`np.ndarray`
        Standard orientation cartesian coordinates.
    
    Returns
    ------
    :obj:`np.ndarray`
        DistanceMatrix: Calculated distance matrix (should agree with Gaussian16 output).  
    :obj:`np.ndarray`
        Trig: Values of trigonometric functions in the X, Y, and Z.  
    """
    
    length = XYZ.shape[0] 
    DistanceMatrix = np.zeros((length,length), dtype=np.float64)

    Cos = {'X':np.zeros((length,length), dtype=np.float64),
           'Y':np.zeros((length,length), dtype=np.float64),
           'Z':np.zeros((length,length), dtype=np.float64)
          }

    Sin = {'X':np.zeros((length,length), dtype=np.float64),
           'Y':np.zeros((length,length), dtype=np.float64),
           'Z':np.zeros((length,length), dtype=np.float64)
          }

    for param1 in range(1,length):
        for param2 in range(param1):
            Delta = {'X':XYZ[param2][0] - XYZ[param1][0], # differences between X,Y, and Z
                     'Y':XYZ[param2][1] - XYZ[param1][1],
                     'Z':XYZ[param2][2] - XYZ[param1][2]
                    }

            d = np.sqrt(Delta['X']**2 + Delta['Y']**2 + Delta['Z']**2)
            DistanceMatrix[param1][param2] = d

            for l in np.array(['X','Y','Z']):
                p = MakeSureZero(Delta[l] / d)
                Cos[l][param1][param2] = p
                Sin[l][param1][param2] = CalculateSin(p) 
        
    Trig = (Cos['X'], Sin['X'], Cos['Y'], Sin['Y'], Cos['Z'], Sin['Z'])
    return (DistanceMatrix, Trig)
