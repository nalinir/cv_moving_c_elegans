# %%
print('starting file')

import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
import numba

'''available worms:
 - "/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test/sub-20190928-03"
 - "/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test/sub-20190928-11"
 - "/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test/sub-20190929-06"
- "/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test/sub-20191030-07"'''
# %%
with open('/scratch/mie8014/BAN/computer_vision/BrainAlignNet/registration_problems.json', 'r') as file:
    reg_problems= json.load(file)


#change these 2 variables/paths!
worm = 'sub-20191030-07'
mappings = reg_problems['test']['EY_Out/sub-20191030-07_ses-20191030']


#everything below stays the same
with open(f"/scratch/mie8014/BAN/computer_vision/BrainAlignNet/EY/fold1_lrlarge_2/registered_outputs_{worm}.pkl", 'rb') as f:
    warped_data = pickle.load(f)

path='/scratch/mie8014/BAN/computer_vision/data/EY/fold1/test'
fixed_images = h5py.File(f"{path}/{worm}/fixed_images.h5",'r')
fixed_labels = h5py.File(f"{path}/{worm}/fixed_labels.h5",'r')
fixed_rois = h5py.File(f"{path}/{worm}/fixed_rois.h5",'r')

print(f'worm {worm} loaded')

@numba.jit(nopython=True, parallel=True,fastmath=True)
def IoU_3D(seg1, seg2):
    ROIs1=np.unique(seg1)
    ROIs2=np.unique(seg2)

    submatrix = np.zeros((400,400))

    for i, label1 in enumerate(ROIs1):
        if label1>0 and i<400:
            for j,label2 in enumerate(ROIs2):
                if label2 > 0 and j<400:
                    intersection = np.where(seg1==label1, np.where(seg2==label2,1,0),0)
                    union=np.where((seg1==label1) | (seg2==label2),1,0)#np.where(seg1==label1, np.where(seg2==label2,1,0),0)
                    submatrix[i,j]=np.sum(intersection)/np.sum(union)
    print(submatrix)

    return submatrix[1:,1:] #exclude background

def IoU_3D_fast(seg1, seg2):
    # skip background (0)
    ROIs1 = np.unique(seg1)[1:]
    ROIs2 = np.unique(seg2)[1:]  

    # submatrix to store IoU values
    submatrix = np.zeros((len(ROIs1), len(ROIs2)))

    # mask for all label combinations
    for i, label1 in enumerate(ROIs1):
        if i<400:
            for j, label2 in enumerate(ROIs2):
                if j<400:
                    intersection = np.logical_and(seg1 == label1, seg2 == label2)
                    union = np.logical_or(seg1 == label1, seg2 == label2)

                    intersection_count = np.sum(intersection)
                    union_count = np.sum(union)
                    if union_count > 0:
                        submatrix[i, j] = intersection_count / union_count
    

    return submatrix



all_moving_arrays=[]
all_fixed_arrays=[]


# input to jit function cannot be h5, only list of numpy arrays
for time_map in mappings:
    all_moving_arrays.append(np.array(warped_data[time_map]['warped_moving_roi']))
    all_fixed_arrays.append(np.array(fixed_rois[time_map]))


@numba.jit(nopython=True, parallel=True,fastmath=True)
def loop_timesteps(all_moving_arrays, all_fixed_arrays, mappings):
    matrix_data={}
    for i in numba.prange(len(all_moving_arrays)):
        #for i in np.arange(len(all_moving_arrays)):
        print(f'timestep: {mappings[i]}')
        roi_overlap_matrix = IoU_3D(all_moving_arrays[i], all_fixed_arrays[i])
        matrix_data[mappings[i]] = roi_overlap_matrix
    return matrix_data

# %%
matrix_data=loop_timesteps(all_moving_arrays, all_fixed_arrays, mappings)


# save
matrix_data_file = f'{worm}_similarity_matrix.h5'
for i in np.arange(len(all_moving_arrays)):
    # Save the matrix_data to the HDF5 file after processing this time_map
    with h5py.File(matrix_data_file, 'a') as f:  # Use 'a' to append
        if time_map not in f:  # Check if the dataset exists
            f.create_dataset(time_map, data=roi_overlap_matrix)  # Create dataset if it doesn't exist
        else:
            print(f"Dataset {time_map} already exists, skipping creation.")
