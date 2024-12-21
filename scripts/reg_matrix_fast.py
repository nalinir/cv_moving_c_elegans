import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress tracking
from numba import njit, prange
# numpy.set_printoptions(threshold=sys.maxsize)

# worm = 'sub-20190928-03' # Neel
worm = 'sub-20190928-11' # Maren
# worm = 'sub-20190929-06' # Nalini
# worm = 'sub-20191030-07' # Ryan

with open(f"/scratch/mie8014/BAN/computer_vision/BrainAlignNet/EY/fold1_lrlarge_2/registered_outputs_{worm}.pkl", 'rb') as f:
    warped_data = pickle.load(f)

mappings = list(warped_data.keys())

# with open('/scratch/nar8991/computer_vision/BrainAlignNet/demo_notebook/registered_outputs.pkl', 'rb') as f:
#     warped_data = pickle.load(f)

filepath = '/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test/'

fixed_images = h5py.File(f"{filepath}/{worm}/fixed_images.h5",'r')
fixed_labels = h5py.File(f"{filepath}/{worm}/fixed_labels.h5",'r')
fixed_rois = h5py.File(f"{filepath}/{worm}/fixed_rois.h5",'r')

@njit(parallel=True)
def IoU_3D(seg1, seg2):
    ROIs1=np.unique(seg1)
    ROIs2=np.unique(seg2)

    # assert(len(ROIs1)==len(ROIs2))

    submatrix = np.zeros((400,400))

    for i, label1 in enumerate(ROIs1):
        if label1>0 and i < 400:
            for j,label2 in enumerate(ROIs2):
                if label2 > 0 and j < 400:
                    intersection = np.where(seg1==label1, np.where(seg2==label2,1,0),0)
                    union=np.where((seg1==label1) | (seg2==label2),1,0)#np.where(seg1==label1, np.where(seg2==label2,1,0),0)
                    submatrix[i,j]=np.sum(intersection)/np.sum(union)

    return submatrix[1:,1:] #exclude background

@njit(parallel=True)
def IoU_3D_numba(seg1, seg2):
    ROIs1 = np.unique(seg1)
    ROIs2 = np.unique(seg2)

    # Filter only positive ROI labels
    ROIs1 = ROIs1[ROIs1 > 0]
    ROIs2 = ROIs2[ROIs2 > 0]

    # Limit to 400 ROIs
    ROIs1 = ROIs1[:400]
    ROIs2 = ROIs2[:400]

    # Preallocate matrix
    submatrix = np.zeros((len(ROIs1), len(ROIs2)))

    for i in prange(len(ROIs1)):  # Use prange for parallel loops
        label1 = ROIs1[i]
        mask1 = seg1 == label1  # Boolean mask for label1
        for j in range(len(ROIs2)):
            label2 = ROIs2[j]
            mask2 = seg2 == label2  # Boolean mask for label2
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            if union > 0:
                submatrix[i, j] = intersection / union

    return submatrix

# Define the file path to save the matrix_data in HDF5 format
matrix_data_file = f'{worm}_fast_similarity_matrix.h5'
print(matrix_data_file)
# Check if the file already exists
if os.path.exists(matrix_data_file):
    with h5py.File(matrix_data_file, 'r') as f:
        # Load existing matrix_data from the HDF5 file
        matrix_data = {key: f[key][...] for key in f.keys()}
else:
    matrix_data = {}

print("Total mappings to find matrices for: ", len(mappings), matrix_data)

for time_map in tqdm(mappings, desc="Processing Mappings", unit="mapping"):
    if time_map in matrix_data:
        print(f"Skipping {time_map}, already computed.")
        continue
    
    print(f"Started {time_map}")
    
    # Assuming the data is correctly indexed here
    array1_3d = warped_data[time_map]['warped_moving_roi']
    array2_3d = fixed_rois[time_map][:,:,:]
    
    # Compute the IoU matrix (or any relevant function)
    roi_overlap_matrix = IoU_3D_numba(array1_3d, array2_3d)
    
    # Save the computed matrix in the dictionary
    matrix_data[time_map] = roi_overlap_matrix
    
    print(f"Completed {time_map}")
    print("\n\n")
    
    # Save the matrix_data to the HDF5 file after processing this time_map
    with h5py.File(matrix_data_file, 'a') as f:  # Use 'a' to append
        if time_map not in f:  # Check if the dataset exists
            f.create_dataset(time_map, data=roi_overlap_matrix)  # Create dataset if it doesn't exist
        else:
            print(f"Dataset {time_map} already exists, skipping creation.")

print(f"Matrix data saved to {matrix_data_file}")