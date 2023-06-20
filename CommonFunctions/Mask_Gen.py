import numpy as np
from scipy.special import erf
from CommonFunctions import Gaussians
Gau_gen = Gaussians.GaussFuncs()

class MaskFuncs():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def object_loc_generation(n_dyes=3, n_objects=10, correlation=0, image_h=1280, image_w=1024):
        # mask_generation function
        # for n dyes, create a mask for each of the dye stains
        # ================INPUTS============= 
        # n_dyes is number of dyes
        # n_objects is how many objects per dye
        # correlation is how correlated object (i.e. dye) locations will be; 1 means high correlation 0 no correlation (two options)
        # image_h is image height in pixels
        # image_w is image width in pixels
        # ================OUTPUTS============= 
        # object_locs contains object centroid locations
        sigma = np.square(np.min([image_h, image_w])/5.) # empirical
        cov = np.identity(2)
        test = np.where(~np.eye(cov.shape[0],dtype=bool))
        cov[test] = correlation
        cov = cov*sigma

        object_locs = np.zeros([n_objects, 2, n_dyes])

        object_locs[:, :, 0] = np.abs(np.random.multivariate_normal([image_h/2., image_w/2.], cov, size=(n_objects)))
        object_locs[object_locs[:, 0, 0] > image_h] = image_h    
        object_locs[object_locs[:, 1, 0] > image_w] = image_w

        for i in np.arange(n_dyes-1):
            if correlation == 0:
                object_locs[:, :, i+1] = np.abs(np.random.multivariate_normal([image_h/2., image_w/2.], cov, size=(n_objects)))
                object_locs[object_locs[:, 0, i+1] > image_h] = image_h    
                object_locs[object_locs[:, 1, i+1] > image_w] = image_w
            else:
                object_locs[:, :, i:] = object_locs[:, :, 0]
        return object_locs
    
    @staticmethod
    def mask_generation(object_locs, n_dyes=3, n_objects=10, object_sigma=30, image_h=1280, image_w=1024):
        # create image_mask tensor
        # mask_generation function
        # for n dyes, create a mask for each of the dye stains
        # ================INPUTS============= 
        # object_locs is where objects are
        # n_dyes is number of dyes
        # n_objects is how many objects per dye
        # image_h is image height in pixels
        # image_w is image width in pixels
        # ================OUTPUTS============= 
        # image_masks is image mask       
        image_masks = np.zeros([image_h, image_w, n_dyes])
        X, Y = np.meshgrid(np.arange(image_h), np.arange(image_w))
        
        for i in np.arange(n_dyes):
            for j in np.arange(n_objects):
                mx = object_locs[j, 0, i]
                my = object_locs[j, 1, i]
                image_masks[:, :, i] += Gau_gen.gaussian_2d(x=X, y=Y, mx=mx, my=my, sx=object_sigma, sy=object_sigma).T
        return image_masks
    
    @staticmethod
    def make_Bayer(image_h=1280, image_w=1024):
        # make_Bayer function
        # creates Bayer filter for 3 colour channels
        # ================INPUTS============= 
        # image_h is image height in pixels
        # image_w is image width in pixels
        # ================OUTPUTS============= 
        # image_masks is tensor containing n masks for n dyes
        # object_locs contains object centroid locations

        #create simple mask units
        blue_mask = np.array([[1, 0], [0, 0]])
        green_mask = np.identity(2)[::-1]
        red_mask = np.array([[0, 0], [0, 1]])

        #create overall image masks
        red_mask_image = np.tile(red_mask, (int(image_h/2), int(image_w/2)))
        green_mask_image = np.tile(green_mask, (int(image_h/2), int(image_w/2)))
        blue_mask_image = np.tile(blue_mask, (int(image_h/2), int(image_w/2)))
        return red_mask_image, green_mask_image, blue_mask_image