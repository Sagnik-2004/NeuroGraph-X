import torch
import numpy as np
import torch.nn.functional as F
import json
import os

with open("./data/region_mapping.json", "r") as f:
    REGION_TO_IDX = json.load(f)

def get_Patient_Matrix(patient_folder_path):
    patient_matrix = torch.zeros((166, 512))
    for filenames in os.listdir(patient_folder_path):
        if filenames.endswith(".pt"):
            region_name = filenames.replace(".pt", "")
            if region_name in REGION_TO_IDX:
                row_idx = REGION_TO_IDX[region_name]
                feature_vector = torch.load(os.path.join(patient_folder_path, filenames))
                patient_matrix[row_idx] = feature_vector.squeeze()
    return patient_matrix

def get_Morphological_Features(patient_matrix, threshold=0.75):

    #divide every 512 dimensional vector by its L2 norm, each becomes an unit vector
    normalized_mtx = F.normalize(patient_matrix, dim = 1)

    #compute cosine similarity between each brain region using the extracted featues
    cosine_mtx = torch.mm(normalized_mtx, normalized_mtx.t())

    #thresholding
    mask = cosine_mtx >= threshold
    mask.fill_diagonal_(False)

    #PyG expects data in COO format, thus extract only the nonzero edges and their attributes
    edge_index_morphological = torch.nonzero(mask, as_tuple=False).t().contiguous()
    edge_attr_morphological = cosine_mtx[mask].clone().detach().to(torch.float32)  
    
    return edge_index_morphological, edge_attr_morphological