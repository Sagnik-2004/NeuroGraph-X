'''
    A script utilizing the 3d ResNet of PyTorch with pre-trained weights
    for extracting useful features from each region of each patient

    @author: Silajeet
'''

import torch.nn as nn
import torchvision
import torch
import re
from pathlib import Path

TENSOR_DIR = Path("./data/CNN_Tensors/") #path to the parcellated region tensors
OUTPUT_DIR = Path("./data/GNN_Features/") #path to the storage location of the feature vector
OUTPUT_DIR.mkdir(exist_ok=True)

patient_scans = {} #to extract and store unique day 1 scans of each patient

for folder in TENSOR_DIR.iterdir():
    if not folder.is_dir():
        continue
    match = re.search(r'(\d{3}_S_\d{4})_I(\d+)', folder.name) #find the patient folder names

    if match:
        ptid = match.group(1) #extract patient id
        image_id = int(match.group(2)) #extract image id and convert to int

        if ptid not in patient_scans: #if a new entry create a new list
            patient_scans[ptid] = []
        patient_scans[ptid].append((image_id, folder.name)) #append a tuple containing the image id and patient id+image id, grouped under same patient id

baseline_folders = [] #list storing the Path object of each patient's first day scans
for _, scans in patient_scans.items():
    scans.sort(key = lambda x : x[0]) #sort based on image id, lowest one is the first scan
    earliest_scan_folder = scans[0][1] #get the earliest scan folder name
    baseline_folders.append(TENSOR_DIR / earliest_scan_folder) #append in the form of directory structure

print(f"Filtered down to {len(baseline_folders)} unique Day 1 baseline scans of the patients...")

model = torchvision.models.video.r3d_18(weights = 'DEFAULT') #load the Video ResNet model with default weights

#modify the first layer, since default expects in_channels = 3 (RGB) but we have grayscale the rest of the parameters are in convention with Facebook's actual code for the model
model.stem[0] = nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = (3, 7, 7), stride = (1, 2, 2), padding = (1, 3, 3), bias = False)
model.fc = nn.Identity() #remove the fully connected classification layer since we are extracting features and not labels

model.eval() #put the model in evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #push the model to the GPU (if available)
model.to(device)

for files in baseline_folders:
    patient_out_folder = OUTPUT_DIR / files.name #folders to store the feature vectors for each individual patients
    patient_out_folder.mkdir(exist_ok = True)

    print(f"Extracting features for patient : {files.name}...")
    for regions in files.rglob("*.pt"): #fetch each of the tensors for a particular patient
        try:
            tnsr = torch.load(regions) #load the tensors
            tnsr = torch.nan_to_num(tnsr, nan = 0.0, posinf = 0.0, neginf = 0.0) #handling NaN
            #r3d_18 expects tensors of the form: (B, C, T, H, W), we set B(batch) = 1 and C(channel) = 1
            tnsr = tnsr.unsqueeze(0) #sets batch = 1
            tnsr = tnsr.unsqueeze(0) #sets channel = 1
            tnsr = tnsr.to(device, dtype = torch.float32) #sends the tensor to the device CPU/GPU

            #since we are using pre trained model, we have no need of gradient calculation so we disable it saving compute time
            with torch.no_grad():
                features = model(tnsr) #get the feature vector for the particular region

            save_path = patient_out_folder / regions.name #path to save the feature vector
            torch.save(features.cpu(), save_path) #bring the tensor on the cpu and then save it
        except Exception as e:
            print(f"Failed on region: {regions.name} for patient: {files.name}. Error: {e}")
            continue

    print(f"Extracted features successfully for patient : {files.name}...")

print(f"\nFeature Extraction completed!")