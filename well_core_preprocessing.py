import os
from PIL import Image
import pandas as pd

def ft_to_mm(ft):
    return ft * 304.8

def m_to_mm(m):
    return m * 1000

class CoreImages():

    def __init__(self, core_photos:list, core_x:list, core_y:list, core_x_mm:int, slice_window:int, slice_step:int=None) -> None:
        """
        core_photos: a list of core image paths
        core_x: Tuple (x1,x2) of the x coordinates of the core trays
        core_x_mm: length of the core in cm
        core_y: Tuple of tuples ((y1,y2),(y1,y2),(y1,y2),...) with y coordinates for each of the cores within the image
        slice_window: width (px) to slice each core image into. If not integer divisible by the length of the core, the end of the core will be discarded
        slice_step: the interval (px) to move when taking a new slice. Minimum of 1px, defaults to value of slice_window. 
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

    def px_to_mm(self, px:int)->int:
        """ Converts a number of pixels to a measurement in cm"""
        return px * self.px_cm_conversion
    
    def mm_to_px(self, cm:int)->int:
        """ Converts a measurment of cm to a number in pixels"""
        return cm / self.px_cm_conversion

    def __getitem__(self, idx:int)-> list:
        """return the relative paths of the core images"""
        #still a bit broken, needs work to retrieve these properly
        return self.paths[idx]

    def slice_cores(self, well_core_s3=None, core_dir:str=os.getcwd(), slice_dir:str=os.getcwd(), labels:str=None, slice_metadata_path:str="slice_metadata.csv", core_metadata_path:str="core_metadata.csv",verbose:int=0, save_local:bool=True) -> None:
        """extract the cores from each image, and slice the cores into slices of <slice> width"""
        self.core_slices = []
        core_left, core_right = self.core_x

        if labels:
            df_labels = pd.read_csv(labels)
            if verbose == 2:
                print("Core labels:")
                print(df_labels)

        df_core_metadata = pd.read_csv(core_metadata_path)

        with open(slice_metadata_path, "w") as f:
            f.write("well_name,box_ID,n_core,n_slice,label,depth\n")
            for core_photo in self.core_photos:
                
                box_ID = os.path.splitext(os.path.basename(core_photo))[0]

                top_depth = float(df_core_metadata[df_core_metadata["box_ID"] == box_ID]["top_depth"])
                depth_units = df_core_metadata[df_core_metadata["box_ID"] == box_ID]["depth_units"].values[0]
                well_name  = df_core_metadata[df_core_metadata["box_ID"] == box_ID]["well_name"].values[0]

                if depth_units == "m":
                    top_depth = m_to_mm(top_depth)
                elif depth_units == "ft":
                    top_depth = ft_to_mm(top_depth)
                else:
                    raise ValueError(f"Unrecognised depth unit {depth_units} in {box_ID}")

                if verbose > 0:
                    print(f"Processing core photo: {core_photo}")

                core_paths = []
                for n_core in range(self.cores_per_image):

                    #extract the core from the image
                    
                    core_photo_img = Image.open(core_photo)
                    core_img = core_photo_img.crop((core_left,self.core_y[n_core][0],core_right,self.core_y[n_core][1]))
                    core_path = f"{core_dir}/{box_ID}_{n_core}.jpg"
                    if save_local:
                        core_img.save(core_path)
                    
                    #slice the core up
                    core_width, core_height = core_img.size
                    slice_left = 0
                    slice_right = self.slice_window
                    n_slice = 0
                    slice_paths = []
                    while slice_right < core_width:

                        # find the label for this slice
                        if labels:
                            depth_mm_relative = self.px_to_mm(slice_left+self.slice_window/2) #depth relative to top of core, add half the window size to find the label at the midpoint of the window
                            depth_mm = depth_mm_relative + (n_core+1) * self.core_x_mm + top_depth #get the depth relative to the first core in the image
                            label = (df_labels[(df_labels["box_ID"] == box_ID) & (df_labels["n_core"] == n_core+1) & (df_labels["length"] <= depth_mm_relative)].tail(1)["type"].values)
                            if len(label)>0:
                                label = label[0]
                            else:
                                label = "unlabelled"

                        #crop and save the slice
                        slice_img = core_img.crop((slice_left,0,slice_right,core_height))

                        slice_path = f"{slice_dir}/{box_ID}_{n_core}_{n_slice}.jpg"
                        slice_img.save(slice_path)
                        slice_img.close()
                        
                        if well_core_s3:
                            metadata= {
                                "box_ID":"S00066842",
                                "n_core":f"{n_core + 1}",
                                "n_slice":f"{n_slice}",
                                "label":f"{label}"
                            }

                            file_name = slice_path
                            object_name = f"{box_ID}_{n_core}_{n_slice}"

                            well_core_s3.upload_file(file_name, metadata, object_name)

                        if not save_local:
                            os.remove(slice_path)

                        f.write(f"{well_name},{box_ID},{n_core},{n_slice},{label},{depth_mm}\n")

                        # increment for next loop
                        n_slice += 1
                        slice_left += self.slice_step
                        slice_right += self.slice_step
                    
                    core_img.close()
                    core_photo_img.close()
            