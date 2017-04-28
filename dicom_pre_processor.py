import os
import numpy as np
import pandas as pd
import scipy.ndimage
from skimage import measure, morphology
import dicom


def load_complete_scan(path):
    print(path)
    slices = [dicom.read_file(os.path.join(path,s)) for s in os.listdir(path)]
    slices.sort(key= lambda slice_x: float(slice_x.ImagePositionPatient[2]) ) # Sort by position on z-axis
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    print('spacing is '+str(spacing))

    resize_factor = spacing / new_spacing
    print('resize factor is '+str(resize_factor))
    print('image shape is '+str(image.shape))
    new_real_shape = image.shape * resize_factor
    print('new_real_shape is '+str(new_real_shape))
    new_shape = np.round(new_real_shape)
    print('new_shape is '+str(new_shape))
    real_resize_factor = new_shape / image.shape
    print('real_resize_factor is '+str(real_resize_factor))
    new_spacing = spacing / real_resize_factor
    print('new_spacing is '+str(new_spacing))
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


# This function returns the label value
# that occupies the most pixels in the segmented image
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # measure.label() considers 0 to correspond to the background;
    # since we aren't sure about this, we force the method to determine the background algorithmically
    # by adding 1 to each pixel, instead of relying on 0 as a default
    binary_image = np.array(image > -320, dtype=np.int8)+1

    #Segmenting the image
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #The air around the lungs is not of interest to us
    #so we set it to the background valueFill the air around the person
    #The air outside the lungs is assumed to be connected; hence this should definitely be identified as 
    #background pixel by our segmentation algorithm.
    binary_image[labels == background_label] = 2 # Setting background pixels in the original image to 2 

    # Now the background is set to 2 and so one would assume
    # that lung tissue would be represented by 1
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        # At this point the air around the lungs will have the value 1
        for i, axial_slice in enumerate(binary_image):
            #Setting the voxels that include lung tissue to zero
            axial_slice = axial_slice - 1 
            #When the image is segmented (measure.label()), the lung tissue will be
            #considered as background
            labeling = measure.label(axial_slice)
            
            # Since the lung tissue is considered as background by measure.label(), it is obvious that 
            # applying largest_label_volume to find the largest connected structure besides the background
            # will give us the voxels that represent the air around the body.
            
            l_max = largest_label_volume(labeling, bg=0) # We find the largest segment besides the 
                                                        # background(air). This should represent lung tissue
            
            if l_max is not None: #This slice contains some lung
                # Setting all regions that are not lung tissue to background
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


def dicom_numpy_converter(path):
    for f in os.listdir(path):
        filepath = os.path.join(path,f)
        first_patient = load_complete_scan(filepath)
        first_patient_pixels = get_pixels_hu(first_patient)
        pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
        #Use the line below if filling is not required
        #segmented_lungs = segment_lung_mask(pix_resampled, False)
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
        bin_name = os.path.basename(os.path.normpath(f))
        np.save(bin_name+'.npy',segmented_lungs_fill)

    return 

dicom_numpy_converter('/home/ajsrk1207/sample_images/')
#x = dicom_numpy_converter('/home/ajsrk1207/Downloads/sample_images/0a0c32c9e08cc2ea76a71649de56be6d')
#np.save(x,)
#print(x.shape)
#print(x.nbytes)



