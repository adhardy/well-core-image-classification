import os
from PIL import Image 
import numpy as np

class CoreImages():

    def __init__(self, core_photos, core_x, core_y, core_x_mm, slice_window, slice_step=None):
        """
        core_photos: a list of core image paths
        core_x: Tuple (x1,x2) of the x coordinates of the core trays
        core_x_cm: length of the core in cm
        core_y: Tuple of tuples ((y1,y2),(y1,y2),(y1,y2),...) with y coordinates for each of the cores within the image
        slice: width to slice each core image into. If not integer divisible by the length of the core, the end of the core will be discarded
        """

        self.slice_window = slice_window
        assert self.slice_window > 0, "Slice window should be greater than 0"

        #default slice step size to window size
        if slice_step == None:
            self.slice_step = slice_window
        else:
            self.slice_step = slice_step
        assert self.slice_step > 0, "Slice step should be greater than 0"

        #assign arguments to self
        self.core_photos = core_photos
        self.core_x = core_x
        self.core_y = core_y
        self.core_x_mm = core_x_mm
        
        self.cores_per_image = len(self.core_y) #infer core per images from number of entries in core_y

        self.core_length = core_x[1] - core_x[0]

        self.paths = [] #stores paths of all photos, cores, slices
        self.px_cm_conversion = core_x_mm / self.core_length #get conversion factor between pixels and cm

    def px_to_mm(self, px):
        """ Converts a number of pixels to a measurement in cm"""
        return px * self.px_cm_conversion
    
    def mm_to_px(self, cm):
        """ Converts a measurment of cm to a number in pixels"""
        return cm / self.px_cm_conversion

    def __getitem__(self, idx):
        #still a bit broken, needs work to retrieve these problem
        return self.paths[idx]

    def slice_cores(self, core_dir = os.getcwd(), slice_dir = os.getcwd(), verbose = 0):
        """extract the cores from each image, and slice the cores into slices of <slice> width"""
        self.core_slices = []
        core_left, core_right = self.core_x

        for n_photo, core_photo in enumerate(self.core_photos):
            if verbose > 0:
                print(f"Processing core photo: {core_photo}")
            core_paths = []
            for n_core in range(self.cores_per_image):
                #extract the core from the image
                core_photo_img = Image.open(core_photo)
                core_img = core_photo_img.crop((core_left,self.core_y[n_core][0],core_right,self.core_y[n_core][1]))
                core_path = f"{core_dir}/{os.path.splitext(os.path.basename(core_photo))[0]}_{n_core}.jpg"
                
                core_img.save(core_path)
                

                #slice the core up
                core_width, core_height = core_img.size
                slice_left = 0
                slice_right = self.slice_window
                n_slice = 0
                slice_paths = []
                while slice_right < core_width:
                    slice_img = core_img.crop((slice_left,0,slice_right,core_height))
                    slice_path = f"{slice_dir}/{os.path.splitext(os.path.basename(core_photo))[0]}_{n_core}_{n_slice}_{{0:05.0f}}_{{1:05.0f}}.jpg".format(self.px_to_mm(slice_left)*10,self.px_to_mm(slice_right)*10)
                    slice_paths.append(slice_path)
                    slice_img.save(slice_path)
                    slice_img.close()

                    n_slice += 1
                    slice_left += self.slice_step
                    slice_right += self.slice_step
                
                core_paths.append([core_path, slice_paths])

                core_img.close()
                core_photo_img.close()
        
            self.paths.append([core_photo, core_paths])