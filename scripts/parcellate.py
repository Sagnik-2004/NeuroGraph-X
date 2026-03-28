'''
    Script to parcellate the brain according to the AAL Atlas
    @author: Silajeet
'''

from nilearn import image
import nibabel as nib
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import re

# paths to all the data and needed files

ATLAS_PATH = "./data/atlas/aal_3v2/AAL3v1.nii.gz" 
INPUT_DIR = Path("./data/PostRegistration/MCI/")
XML_PATH = "./data/atlas/aal_3v2/AAL3v1.xml"
OUTPUT_BASE_DIR = Path("./data/CNN_Tensors/")

aal = nib.load(ATLAS_PATH) #load the AAL Atlas

tree = ET.parse(XML_PATH) # generate the xml tree
root = tree.getroot() # points to the xml tree's root

labels = {} # dictionary for storing the index : region pairs corresponding to AAL Atlas
for label in root.findall(".//label"):
    index = int(label.find("index").text)
    name = label.find("name").text
    labels[index] = name

print("Starting batch processing...")

for patient_file in INPUT_DIR.rglob("*.nii.gz"): # find all files with the ending extension .nii.gz
    match = re.search(r'\d{3}_S_\d{4}_I\d+', patient_file.name) # regex to get patient id
    if not match:
        continue
    patient_id = match.group(0) # get the patient id

    patient_out_folder = OUTPUT_BASE_DIR / patient_id
    os.makedirs(patient_out_folder, exist_ok=True) # create a folder to store tensors for brain regions

    print(f"\nProcessing Patient: {patient_id}")
    try:
        img = nib.load(patient_file) # load the patient scan

        resampled_img = image.resample_to_img(aal, img, interpolation = 'nearest') #resample the AAL Atlas to match patient's scan
        aal_data = resampled_img.get_fdata() # convert the AAL nii object to a 3d numpy array
        patient_data = img.get_fdata() # convert the patient nii object to 3d numpy array

        for region_id, region_name in labels.items():
            mask = (aal_data == region_id) # a 3d boolean mask to filter out regions based on region id
            if not np.any(mask): # if due to some human or machine fault the index doesn't matches
                continue
            x, y, z = np.where(mask) # each of x, y, z are lists corresponding to the coordinates where the condition is satisfied i.e. mask = True
            cropped_region = patient_data[x.min() : x.max() + 1, y.min() : y.max() + 1, z.min() : z.max() + 1] # extract the region
            tnsr = torch.tensor(cropped_region, dtype = torch.float32) # create a tensor object for the region
            save_path = patient_out_folder / f"{region_name}.pt"
            torch.save(tnsr, save_path) # store them

        print(f"Finished processing patient: {patient_id}")
    except Exception as e:
        print(f"CORRUPTED DATA: Skipping {patient_id}. Error: {e}")
        continue

print("Process completed successfully")
