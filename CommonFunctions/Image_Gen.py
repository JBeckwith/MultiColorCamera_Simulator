import numpy as np
from scipy.special import erf
from CommonFunctions import Gaussians, Mask_Gen
import cv2
import pandas as pd
import xarray as xr
Gau_gen = Gaussians.GaussFuncs()
MG = Mask_Gen.MaskFuncs()

class ImageGenFuncs():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def gen_camera_images(gain_value, wavelength, R, G, B, dyes, n_photons, laser_int, object_locs, object_sigma=30, background_photons=100, image_h=1280, image_w=1024, pixel_shift=0):
        # gen_camera_images function
        # creates however may images is appropriate given the laser sequence
        # ================INPUTS=============
        # gain_value is gain value
        # wavelength is wavelength space
        # R is red pixel wavelength response
        # G is green pixel wavelength response
        # B is blue pixel wavelength response
        # dyes is tensor of N dyes * absorption * emission
        # n_photons is how many photons each dye outputs per object (must be same length as dyes tensor)
        # laser_intensities is n_wavelengths*(n_intensities) matrix saying which wavelength lasers were on at what intensitites
        # object_locs is object locations per dye
        # background_photons is average number of background photons per pixel
        # image_h is height of image
        # image_w is width of image
        # pixel_shift is shift between laser shots
        # ================OUTPUTS============= 
        # raw_images is stack of raw R G B channels camera would read out
        # colour_image_demosaic is colour images
        # gain map is simulated gain map
        h = image_h; w = image_w;
        from Gain import getgain; gain_cal = getgain.GainFunc(); gain_map = gain_cal.getgain(gain_value, h, w) # first, get gain map

        laser_wavelengths = laser_int[0, :]
        laser_intensities = laser_int[1:, :]
        n_shots = laser_intensities.shape[0]
        
        dims=['n_shots', 'im_width', 'im_height', 'n_colours']
        coords = [np.arange(n_shots), np.arange(w), np.arange(h), ['R', 'G', 'B']]
        raw_images = xr.DataArray(data=np.zeros([n_shots, w, h, 3]), dims=dims, coords=coords)        
        colour_image_demosaic = xr.DataArray(data=np.zeros([n_shots, w, h, 3]), dims=dims, coords=coords)        
       
        for shot in np.arange(n_shots):
            object_locs[:, 1, :] = object_locs[:, 1, :] + pixel_shift*shot # move objects along in X
            image_masks = MG.mask_generation(object_locs, dyes.shape[-1], len(object_locs), object_sigma, h, w); # make image mask for first image
            R_image = np.zeros([w, h]); G_image = np.zeros([w, h]); B_image = np.zeros([w, h]);
            for dye in np.arange(dyes.shape[-1]):
                dye_intensity = 0;
                for wvl in enumerate(laser_wavelengths):
                    wloc = np.argmin(np.abs(wavelength - wvl[1]))
                    dye_intensity += laser_intensities[shot, wvl[0]]*dyes[0, wloc, dye]*n_photons[dye]
                R_image += image_masks[:, :, dye]*np.sum(R*(dye_intensity*dyes[1, :, dye]))
                G_image += image_masks[:, :, dye]*np.sum(G*(dye_intensity*dyes[1, :, dye]))
                B_image += image_masks[:, :, dye]*np.sum(B*(dye_intensity*dyes[1, :, dye]))
            R_image = gain_map*R_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(w, h))
            G_image = gain_map*G_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(w, h))
            B_image = gain_map*B_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(w, h))
            red_mask, green_mask, blue_mask = MG.make_Bayer(image_h=h, image_w=w)
            R_image = R_image*red_mask; G_image = G_image*green_mask; B_image = B_image*blue_mask; 
            raw_images.values[shot, :, :, :] = np.uint16(np.dstack([R_image, G_image, B_image]))
            R_image_demosaic = cv2.cvtColor(np.uint16(R_image), cv2.COLOR_BayerRG2RGB)
            G_image_demosaic = cv2.cvtColor(np.uint16(G_image), cv2.COLOR_BayerRG2RGB)
            B_image_demosaic = cv2.cvtColor(np.uint16(B_image), cv2.COLOR_BayerRG2RGB)
            colour_image_demosaic.values[shot, :, :, :] = np.dstack([R_image_demosaic[:,:,0], G_image_demosaic[:,:,1], B_image_demosaic[:,:,2]])
        return raw_images, colour_image_demosaic, gain_map