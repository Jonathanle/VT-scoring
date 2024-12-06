import os
import pdb # TODO: once done remove this import for debugging

from attention_model import AttentionUNet
from torchmetrics.classification import Dice
import argparse

import numpy as np
import torch

import matplotlib.pyplot as plt
# this neeeds to be ArgumentParser not Parser
parser = argparse.ArgumentParser(description="basic file for processing models")

parser.add_argument("--input_dir", "-i", default = "./data/processed_mri_data/", required = False)
parser.add_argument("--patient", '-p', type=int, default = 529, required = False)
parser.add_argument('-s', '--slice', type=int, default = 0, required = False, help='slice number')
parser.add_argument("--device", required = False, default = 'cuda:0')
parser.add_argument("--use_example", action = 'store_true', required = False, help = 'use nivetha original exapm9le')


args = parser.parse_args()

DEVICE = args.device

images = []
names = ['raw']

images.append(np.load(os.path.join(args.input_dir, f'{args.patient}_PSIR_cine_whole_{args.slice}.npy'))) 
for name in names: 
    images.append(np.load(os.path.join(args.input_dir, f'{args.patient}_PSIR_{name}_{args.slice}.npy'))) 

raw_image = torch.tensor(images[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if args.use_example:
    import json
    inf_data = json.load(open("./data/inference_sample.json"))

    whole_te = torch.tensor(inf_data['lge_whole']).unsqueeze(0).unsqueeze(0)
    cine_te = torch.tensor(inf_data['lge_cropped']).unsqueeze(0).unsqueeze(0)
    x = torch.tensor(inf_data['masked_input']).unsqueeze(0).unsqueeze(0)
    y = torch.tensor(inf_data['lge_seg']).unsqueeze(0).unsqueeze(0)
    raw_image = x

    images[0] = inf_data['masked_input']



model_save_path= './models/unet_att_focal_dice350.pt'

model = AttentionUNet(drop_out_prob=0.3).to(DEVICE)
model.load_state_dict(torch.load('./' + model_save_path))
model.eval()


y_pred = model(raw_image.to(DEVICE))


# Process the image logits with thresholded binary values
binary_thresholded_y = y_pred.detach().cpu().numpy().squeeze(0).squeeze(0)

binary_thresholded_y[binary_thresholded_y < 0.5] = 0 
binary_thresholded_y[binary_thresholded_y >= 0.5] = 255 # the images are scaled from 0 255
# deatch().cpu().numpy() === get rid of gradietn computation (needed for getting using as numpy on cpu)
images.append(binary_thresholded_y)

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for ax, image, in zip(axes, images):
    ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)


plt.show()


