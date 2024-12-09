"""

New and improved segmenter to test the dataset object for visualizing.
"""

import matplotlib.pyplot as plt
from attention_model import AttentionUNet
from dataset import LGEDataset, Preprocessor
import torch

DEVICE = "cuda:0"

pp = Preprocessor()


X, y, ids = pp.transform() # this thing takes the image directories and works 
dataset = LGEDataset(X, y, ids)


model_save_path= './models/unet_att_focal_dice350.pt'

model = AttentionUNet(drop_out_prob=0.3).to(DEVICE)
model.load_state_dict(torch.load('./' + model_save_path))
model.eval()


images, label, patient_id = dataset[2]

raw_image = images[2].unsqueeze(0).unsqueeze(0)

y_pred = model(raw_image.to(DEVICE))

# Process the image logits with thresholded binary values
binary_thresholded_y = y_pred.detach().cpu().numpy().squeeze(0).squeeze(0)

threshold = 0.5 
binary_thresholded_y[binary_thresholded_y < threshold] = 0 
binary_thresholded_y[binary_thresholded_y >= threshold] = 255 # the images are scaled from 0 255
# deatch().cpu().numpy() === get rid of gradietn computation (needed for getting using as numpy on cpu)


fig, axes = plt.subplots(1, 4, figsize=(10, 5))

plt.imshow(binary_thresholded_y)


plt.show()




