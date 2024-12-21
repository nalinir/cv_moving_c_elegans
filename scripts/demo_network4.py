# %% [markdown]
# ### training the BrainAlignNet

# %%
import os
os.chdir('/scratch/mie8014/BAN/computer_vision/BrainAlignNet/scripts')
from train import set_GPU, fit_deepreg

# %%

# %%
# base = "/scratch/mie8014/BAN/computer_vision/BrainAlignNet"
# config_path = f"{base}/demo_network_config.yaml"
# log_dir = f"{base}/demo_notebook"
# experiment_name = "Atanas_processed" # This is also the name of the dataset
# max_epochs = 2
# initial_epoch = 1


# %%
fold='fold4'
base = "/scratch/mie8014/BAN/computer_vision/BrainAlignNet"
config_path = f"{base}/demo_network_config_EY_{fold}_lrlarge.yaml"
log_dir = f"{base}/EY"
experiment_name = f"{fold}_lrlarge" # This is also the name of the dataset
max_epochs = 50
initial_epoch = 1
print(f'running {fold}')

# %%
import tensorflow as tf
#tf.debugging.enable_check_numerics()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
gpus = tf.config.list_physical_devices('GPU')
gpus[0]
print('running registration for all data')
# %%
print(tf.__version__)

# %%
# Centroid Labeler does not determine this
import h5py
import numpy as np
# Path to your .h5 file
'''
file_path = '/scratch/mie8014/BAN/computer_vision/data/EY/train/sub-20190924-01/moving_images.h5'

# Open the file in read-only mode
with h5py.File(file_path, 'r') as f:
    # Iterate through the datasets to check their shapes
    for dataset_name in f:
        dataset = f[dataset_name]
        print(f"Dataset: {dataset_name}, Shape: {dataset.shape}")
        print(np.array(dataset))
'''

# %%

# %%
from register import set_GPU, register

# %%
set_GPU(0)
fit_deepreg(
    config_path,
    log_dir,
    experiment_name,
    max_epochs,
    initial_epoch)

# %% [markdown]
# ### register test (unseen) images with checkpoint (weights of the trained network)

# %%

model_config_path = config_path
model_ckpt_path = f"{log_dir}/{experiment_name}/save/ckpt-50"
print(model_ckpt_path)

# %%
set_GPU(0)
registered_outputs = register(model_config_path, model_ckpt_path, 'test', f"{log_dir}/{experiment_name}", True)

# %%
import matplotlib.pyplot as plt
#plt.imshow(registered_outputs['20190925-04']['107to437']['warped_moving_roi'][:,:,4])
#plt.scatter(registered_outputs['20190925-04']['107to437']['warped_moving_centroids'][:,1],registered_outputs['20190925-04']['107to437']['warped_moving_centroids'][:,0])

# %%
#32307/32768


# %% [markdown]
# #### plot registered outputs

# %%
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

# %%
final_path = os.path.join(f'{log_dir}/{experiment_name}', 'registered_outputs.pkl')

# %%
print(final_path)

# %%
with open(final_path, 'wb') as f:
    pickle.dump(registered_outputs, f)
'''
# %%
# Check to make sure it saved properly
with open(final_path, 'rb') as f:
    loaded_data = pickle.load(f)

# %%
loaded_data['20190925-04']['193to31']

# %%
registered_outputs.keys()

# %%
registered_outputs['2022-04-14-04'].keys()

# %%
# These keys are slightly different because of how I got the test data this time around/various versions vs. what was reported in Atanas (NR)
output_dict = registered_outputs['2022-04-14-04']['1003to1469']
warped_moving_image = output_dict['warped_moving_image']
warped_moving_roi = output_dict['warped_moving_roi']
warped_moving_centroids = output_dict['warped_moving_centroids']

# %%
warped_moving_image.shape, warped_moving_roi.shape, warped_moving_centroids.shape

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 12))
axes[0].imshow(warped_moving_image.max(2));
axes[0].set_title("warped moving image", fontsize=20);
axes[1].imshow(warped_moving_roi.max(2));
axes[1].set_title("warped moving roi", fontsize=20);

axes[2].imshow(warped_moving_roi.max(2));
axes[2].set_title("warped moving roi", fontsize=20);

xs = [x for (x, _, _) in warped_moving_centroids]
ys = [y for (_, y, _) in warped_moving_centroids]


axes[2].scatter(ys, xs, s=5, c='w');
axes[2].set_title("warped moving centroids", fontsize=20);
axes[2].set_xlim(0, 120);
axes[2].set_ylim(0, 284);
axes[2].invert_yaxis()

# %%
# These keys are slightly different because of how I got the test data this time around/various versions vs. what was reported in Atanas (NR)
output_dict = loaded_data['2022-04-14-04']['1003to1469']
warped_moving_image = output_dict['warped_moving_image']
warped_moving_roi = output_dict['warped_moving_roi']
warped_moving_centroids = output_dict['warped_moving_centroids']

# %%
warped_moving_image.shape, warped_moving_roi.shape, warped_moving_centroids.shape

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 12))
axes[0].imshow(warped_moving_image.max(2));
axes[0].set_title("warped moving image", fontsize=20);
axes[1].imshow(warped_moving_roi.max(2));
axes[1].set_title("warped moving roi", fontsize=20);

axes[2].imshow(warped_moving_roi.max(2));
axes[2].set_title("warped moving roi", fontsize=20);

xs = [x for (x, _, _) in warped_moving_centroids]
ys = [y for (_, y, _) in warped_moving_centroids]


axes[2].scatter(ys, xs, s=5, c='w');
axes[2].set_title("warped moving centroids", fontsize=20);
axes[2].set_xlim(0, 120);
axes[2].set_ylim(0, 284);
axes[2].invert_yaxis()
'''

