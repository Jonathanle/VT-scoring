import matplotlib.pyplot as plt
import numpy as np
import argparse

import os

# this neeeds to be ArgumentParser not Parser
parser = argparse.ArgumentParser(description="file for visualizing files")

# hypehsn are not good use underscore for precicesness
# Additionaly I forgot to consider using "type" parameters as well as the "help parameters" for descrbiging the names
parser.add_argument("--input_dir", "-i", default = "./data/processed_mri_data/", required = False)
parser.add_argument("--patient", '-p', type=int, default = 529, required = False, help = 'patient number')
parser.add_argument('-s', '--slice', type=int, default = 0, required = False, help='slice number')



args = parser.parse_args()

images = []
names = ['cine'] #'cine_whole', 'lge', 'raw']

name = 'cine'


for index in range(5): 
    images.append(np.load(os.path.join(args.input_dir, f'{args.patient}_PSIR_{name}_{index}.npy'))) 

fig, axes = plt.subplots(1, 5, figsize=(10, 5))

for ax, image, in zip(axes, images):
    ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 1)
import pdb

plt.show()


# Notes: why is there a discrepancy between the cropped and made segmentation? 
