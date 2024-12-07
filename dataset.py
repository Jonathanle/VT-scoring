


import torch
import os
# how to import datasets? where to retrieve
# syntax error - i need to import torch.utils.data 
# ai is really fast as long as you have a rigorous understanding and fast verification process =
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
# for errors ensure that I am online and fully present in the decisions i make

# Writing a test is useful - only when I want to complete it once ---> I




# vim macros repeat common actions with macros.


# How to do I work with tabular data? with pandas? specifically what? 
# How do I work with image data? using PIL vs npy? 

# basic heursitics load data, know about the objects in data form. 
# distract - nuasnces - 


# with every index -> i have a individual slice --> later I want to gropu them all as a tensor. 
# for every slice / tensor I want to have a label --> where I can access via dicdtionaryu
#TODO: Create a pytorch dataset object that jd


class Preprocessor(): 
    def __init__(self, images_file_path = './data/processed_mri_data2/', labels_filepath =  './data/CAD CMRs for UVA.xlsx'):

        self.df = pd.read_excel(labels_filepath)

        # create a new feature / composite outcome?
        self.df['composite'] = self.df['sustainedvt'] | self.df['cardiacarrest'] | self.df['scd']
        self.images_file_path = images_file_path


        # what do I want? for the images?  ---> goal create a dataset 
        """
        The dataset contains a patient --> in each patient we have all the masked raw imagess.  
        # for now the dataset will contain the first 3 slices
        """ 


    def transform(self, n_slices = 3):
        """
        Function that transforms the image data + the xlsx files into a pytorch dataset 


        Requirements: 
            - processed_mri_data - directory containing the npy files of images, here the
            directory contains patient slices of the format (patient_id)_raw_(slice_no).npy
            
    
        Returns: 
            Pytorch Dataset - dataset class acting as interface for tensors. In particuilar the tensors will X going by image, label
            where indexing by image returns tensor of shape (n_slices, 128, 128) and the y tensor is a single value 

            X = (n_patients, n_slices, 128, 128) - note that X values 
        """

        # heuristic + easy create a dictionary --> np array from patient id
        # df is this good? 
        image_dict = {} # Create a dictionary of tensors for torch array

        for patient_id in self.df['studyid']:
            # use of a flag is heuristically good
            valid_case = True 


            slices = np.zeros((n_slices, 128, 128)) # hardcoded 128 values
            # Error - Here I need to consider np array as taking in a TUPLE for the shape not the actual parameters next time online monitoring see that


            for slice_no in range(n_slices):
                filename = f'raw_{slice_no}.npy' # forgoot to aadd numpy (not a prioritized error
                filepath = os.path.join(self.images_file_path, str(patient_id),  filename)

                # validate that a certain file exists using guard statemnt
                if not os.path.exists(filepath):
                    print(f"Warning: filename {filename} slice {slice_no} does not exist skipping case (patient {patient_id})")
                    valid_case = False
                    break 

                slice = np.load(filepath)
                slices[slice_no] = slice

            if not valid_case: 
                continue

                
            image_dict[patient_id] = torch.Tensor(slices)

        label_dict = {id: torch.tensor(label) for id, label in zip(self.df['studyid'], self.df['composite'])} # need tensor values in the dictionary to create stack


        keys = self.df['studyid'].tolist()
        keys = [id for id in keys if id in image_dict.keys()]


        X = torch.stack([image_dict[key] for key in keys])
        y = torch.stack([label_dict[key] for key in keys])

        return X, y, keys


                


import torch
from torch.utils.data import Dataset

class LGESliceDataset(Dataset):
    def __init__(self, X, y, keys):
        """
        Args:
            X (torch.Tensor): Image tensor of shape (n_patients, n_slices, height, width)
            y (torch.Tensor): Labels tensor of shape (n_patients,)
        """
        self.X = X
        self.y = y
        self.keys = keys 
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y must have same length. Got {len(self.X)} and {len(self.y)}")
            
        self.n_patients = len(self.X)
    
    def __len__(self):
        return self.n_patients
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label) where image has shape (n_slices, height, width)
        """
        return self.X[idx], self.y[idx], self.keys[idx]
    
    def get_dimensions(self):
        """Returns dimensions of data"""
        return {
            'n_patients': self.n_patients,
            'n_slices': self.X.shape[1],
            'height': self.X.shape[2],
            'width': self.X.shape[3]
        }



def main(): 
    # this functions as the integration test while using pdb is there anything bettr?
    # system / integration test very good helps with equivalent cases

    pp = Preprocessor()

    X, y, keys = pp.transform()

    dataset = LGESliceDataset(X, y, keys)


    import pdb
    pdb.set_trace()
    
if __name__ == '__main__':
   main() 