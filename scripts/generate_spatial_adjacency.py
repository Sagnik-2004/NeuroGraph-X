import torch
import numpy as np
import nibabel as nib
from nilearn import image
import json
from scipy.spatial.distance import pdist, squareform

with open("./data/region_mapping.json", "r") as f:
    REGION_TO_IDX = json.load(f)

def get_Spatial_Features():
    K_NEIGHBORS = 5
    RBF_SIGMA = 50.0
    NUM_NODES = 166
    ATLAS_PATH = "./data/atlas/aal_3v2/AAL3v1.nii.gz" 
    MASTER_TEMPLATE = "./data/PostRegistration/MCI/002_S_1155_I843510.nii.gz"

    nifti_to_name = {}
    with open("./data/atlas/aal_3v2/AAL3v1.nii.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nifti_id = int(parts[0])
                region_name = parts[1]
                nifti_to_name[nifti_id] = region_name
    
    atlas = nib.load(ATLAS_PATH)
    ref_img = nib.load(MASTER_TEMPLATE)

    print("Resampling AAL Atlas")
    resampled_img_mask = image.resample_to_img(atlas, ref_img, interpolation = 'nearest')

    mask_data = resampled_img_mask.get_fdata()
    affine = resampled_img_mask.affine

    print("Extracting physical centroids...")
    centroids_mm = np.zeros((NUM_NODES,3))

    for nifti_id, region_name in nifti_to_name.items():
        if region_name in REGION_TO_IDX:
            target_row = REGION_TO_IDX[region_name]

            coords_voxel = np.argwhere(mask_data == nifti_id)

            if len(coords_voxel) == 0:
                print(f"WARNING: Region{region_name} has no voxels!")
                continue
                
            centroid_voxel = coords_voxel.mean(axis = 0)

            centroid_voxel_4d = np.append(centroid_voxel, 1.0)
            centroid_mm = affine.dot(centroid_voxel_4d)[:3]

            centroids_mm[target_row] = centroid_mm

    print("Centroid Extracted! Calculating Adjacency Matrix...")

    dist_mtx = squareform(pdist(centroids_mm, metric = 'euclidean'))

    knn_mask = np.zeros_like(dist_mtx)

    for i in range(NUM_NODES):
        nearest_idx = np.argsort(dist_mtx[i])[1:K_NEIGHBORS+1]
        knn_mask[i, nearest_idx] = 1
        knn_mask[nearest_idx, i] = 1

    rbf_weights = np.exp(-(dist_mtx ** 2) / (2 * (RBF_SIGMA ** 2)))
    A_spatial_dense = rbf_weights * knn_mask

    mask = A_spatial_dense > 0
    edge_idx_spatial = torch.nonzero(torch.tensor(mask), as_tuple=False).t().contiguous()
    edge_attr_spatial = torch.tensor(A_spatial_dense[mask], dtype = torch.float32)

    torch.save({'edge_index' : edge_idx_spatial,
               'edge_attr': edge_attr_spatial}, 
               "./data/A_spatial_master.pt")

FEATURE_PATH = "./data/GNN_Features/"

print("=== Generating Master Spatial Matrix ===")
get_Spatial_Features()