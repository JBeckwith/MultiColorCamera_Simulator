import numpy as np
from scipy.special import erf

class GaussFuncs():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def SkewGaussian_wavelength(sigma, alpha, x0, x):
        # SkewGaussian function
        # outputs normalised Skew Gaussian Function, in wavelength
        # ================INPUTS============= 
        # sigma is width (nm)
        # alpha is skewness parameter
        # x0 is central wavelength in nm
        # x is wavelength in nm
        # ================OUTPUT============= 
        # S_ln is lineshape
        phi = np.multiply(np.divide(2., sigma), np.multiply(np.divide(1., np.sqrt(np.multiply(2., np.pi))), np.exp(-0.5*np.square(np.divide(x - x0, sigma)))))
        xparam = np.multiply(alpha, np.divide(x - x0, sigma))
        Phi = 0.5*(1 + erf(np.divide(xparam, np.sqrt(2))))
        S_ln = np.multiply(phi, Phi)
        S_ln = S_ln/np.max(S_ln)
        return S_ln

    @staticmethod
    def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        # gaussian_2d function
        # generates normalised 2d gaussian function
        # ================INPUTS=============
        # x is x matrix
        # y is y matrix
        # mx is x mean
        # my is y mean
        # sx is x sigma
        # sy is y sigma
        # ================OUTPUT============= 
        # 2d gaussian pdf
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))