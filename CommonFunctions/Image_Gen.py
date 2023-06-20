import numpy as np
from scipy.special import erf
from CommonFunctions import Gaussians, Mask_Gen
import cv2
Gau_gen = Gaussians.GaussFuncs()
MG = Mask_Gen.MaskFuncs()

class ImageGenFuncs():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def gen_camera_images(gain_value, wavelength, R, G, B, dyes, n_photons, laser_int, object_locs, object_sigma=30, background_photons=100, image_h=1280, image_w=1024):
        # gen_camera_images function
        # creates however may images is appropriate given the laser sequence
        # ================INPUTS=============
        # wavelength is wavelength space
        # R is red pixel wavelength response
        # G is green pixel wavelength response
        # B is blue pixel wavelength response
        # dyes is tensor of N dyes * absorption * emission
        # n_photons is how many photons each dye outputs per object (must be same length as dyes tensor)
        # laser_intensities is n_wavelengths*(n_intensities) matrix saying which wavelength lasers were on at what intensitites
        # object_locs is object locations per dye
        # background_photons is average number of background photons per pixel
        # ================OUTPUTS============= 
        # raw_images is tensor of raw R G B channels camera would read out
        # R_image_demosaic is red image
        # B_image_demosaic is blue image
        # G_image_demosaic is green image
        # gain map is simulated gain map
        h = image_h; w = image_w;
        from Gain import getgain; gain_cal = getgain.GainFunc(); gain_map = gain_cal.getgain(gain_value, h, w) # first, get gain map
        R_image = np.zeros([h, w]); G_image = np.zeros([h, w]); B_image = np.zeros([h, w]);

        laser_wavelengths = laser_int[0, :]
        laser_intensities = laser_int[1, :]
                
        image_masks = MG.mask_generation(object_locs, dyes.shape[-1], len(object_locs), object_sigma, h, w); # make image mask for first image
        for dye in np.arange(dyes.shape[-1]):
            dye_intensity = 0;
            for wvl in enumerate(laser_wavelengths):
                wloc = np.argmin(np.abs(wavelength - wvl[1]))
                dye_intensity += laser_intensities[wvl[0]]*dyes[0, wloc, dye]*n_photons[dye]
            R_image += image_masks[:, :, dye]*np.sum(R*(dye_intensity*dyes[1, :, dye]))
            G_image += image_masks[:, :, dye]*np.sum(G*(dye_intensity*dyes[1, :, dye]))
            B_image += image_masks[:, :, dye]*np.sum(B*(dye_intensity*dyes[1, :, dye]))
        R_image = gain_map*R_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(h, w))
        G_image = gain_map*G_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(h, w))
        B_image = gain_map*B_image + np.random.normal(loc=background_photons, scale=0.1*background_photons, size=(h, w))
        red_mask, green_mask, blue_mask = MG.make_Bayer(image_h=h, image_w=w)
        R_image = R_image*red_mask; G_image = G_image*green_mask; B_image = B_image*blue_mask; 
        raw_images = np.uint16(np.dstack([R_image, G_image, B_image]))
        R_image_demosaic = cv2.cvtColor(raw_images[:,:,0], cv2.COLOR_BayerRG2RGB)
        G_image_demosaic = cv2.cvtColor(raw_images[:,:,1], cv2.COLOR_BayerRG2RGB)
        B_image_demosaic = cv2.cvtColor(raw_images[:,:,2], cv2.COLOR_BayerRG2RGB)
        colour_image_demosaic = np.dstack([R_image_demosaic[:,:,0], G_image_demosaic[:,:,1], B_image_demosaic[:,:,2]])
        return raw_images, colour_image_demosaic, gain_map