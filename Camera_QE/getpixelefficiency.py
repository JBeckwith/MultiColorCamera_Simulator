import numpy as np
import pandas as pd

class GPE():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def getpixelefficiency(camera_file='Camera_QE/CS505CU_QE.csv'):
	    # getpixelefficiency function
	    # gets pixel quantum efficiencies from a .csv file, makes R G B efficiency and wavelength scale
	    # ================INPUTS============= 
	    # camera_file is file to read (csv)
	    # ================OUTPUT============= 
	    # R is red pixel efficiency as function of wavelength
	    # G is green pixel efficiency as function of wavelength
	    # B is blue pixel efficiency as function of wavelength
	    # wavelength in nm
	    data = pd.read_csv(camera_file) # read data
	    wavelength_coarse = data.wavelength # read wavelength
	    R_coarse = data.R # read red
	    G_coarse = data.G # read green
	    B_coarse = data.B # read blue
	    wavelength = np.arange(np.min(wavelength_coarse), np.max(wavelength_coarse))
	    R = np.interp(x=wavelength, xp=wavelength_coarse, fp=R_coarse)
	    G = np.interp(x=wavelength, xp=wavelength_coarse, fp=G_coarse)
	    B = np.interp(x=wavelength, xp=wavelength_coarse, fp=B_coarse)
	    return R, G, B, wavelength
