import numpy as np
import pandas as pd

class GainFunc():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def getgain(gainval, image_h=1280, image_w=1024, Mean_Gain='Gain/Mean_Gain.csv', Sigma_Gain='Gain/Sigma_Gain.csv'):
	    # getgain function
	    # gets gain map at specified gain value
	    # ================INPUTS============= 
	    # gainval is gain amount to compute
        # image_h is image height in pixels
        # image_w is image width in pixels
        # Mean_Gain is file with mean gain for camera
        # Sigma_Gain is file with sigma gain for camera
	    # ================OUTPUT============= 
	    # gain_map is gain map at specified gain value
        if gainval < 0:
            gainval = 0
        elif gainval > 10:
            gainval = 10
            
        gain_values = pd.read_csv(Mean_Gain) # read gain
        sigma_gain = pd.read_csv(Sigma_Gain) # read gain sigma
        gm_val = np.interp(x=gainval, xp=gain_values.values[:, 0], fp=gain_values.values[:, 1])
        gs_val = gm_val*0.1
        gain_map = np.random.normal(loc=gm_val, scale=gs_val, size=(image_w, image_h))
        gain_map[gain_map < 0] = 0
        return gain_map
