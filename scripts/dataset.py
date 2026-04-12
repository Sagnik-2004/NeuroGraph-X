import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from utils import get_Patient_Matrix, get_Morphological_Features

class NeuroGraphDataset(InMemoryDataset):
    def __init__(self, root, csv_path, spatial_matrix_path, transform=None, pre_transform=None):
        """
        root: Base directory. PyG will automatically look for /raw and /processed inside here.
        csv_path: Path to your ground truth ADNI CSV.
        spatial_matrix_path: Path to your A_spatial_master.pt.
        """
        self.csv_path = csv_path
        self.spatial_matrix_path = spatial_matrix_path
        super().__init__(root, transform, pre_transform)
        
        # Instantly loads the processed dataset into RAM if it already exists
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Returning an empty list tells PyG we are handling raw files manually
        return []

    @property
    def processed_file_names(self):
        # The name of the final optimized tensor PyG will generate
        return ['neurograph_processed.pt']

    def process(self):
        """
        Executes exactly ONCE to build the dataset and save it to disk.
        """
        data_list = []
        
        print("Loading ADNI labels...")
        df = pd.read_csv(self.csv_path)
        
        # Create a dictionary mapping Subject_ID to an integer label
        label_dict = {}
        for _, row in df.iterrows():
            subject_id = str(row['PTID'])   
            label_dict[subject_id] = int(row['Label'])

        print("Loading global Spatial Matrix...")
        spatial_data = torch.load(self.spatial_matrix_path)
        edge_index_spatial = spatial_data['edge_index']
        edge_attr_spatial = spatial_data['edge_attr']

        # PyG automatically appends '/raw' to your root directory
        raw_features_dir = os.path.join(self.root, 'raw', 'GNN_Features')

        print("Building PyG Data objects for all patients...")
        # Loop through every patient folder with a progress bar!
        patient_folders = [f for f in os.listdir(raw_features_dir) if os.path.isdir(os.path.join(raw_features_dir, f))]
        
        for subject_id in tqdm(patient_folders):
            patient_path = os.path.join(raw_features_dir, subject_id)
            
            if subject_id not in label_dict:
                print(f"\nSkipping {subject_id}: No label found in CSV.")
                continue
                
            # A. Get the Y Label (Target)
            y_label = torch.tensor([label_dict[subject_id]], dtype=torch.float32)

            # B. Build Node Matrix X [166, 512]
            x_matrix = get_Patient_Matrix(patient_path)

            # C. Build Morphological Edges dynamically
            edge_index_morph, edge_attr_morph = get_Morphological_Features(
                patient_matrix=x_matrix, 
                threshold=0.75
            )

            # D. Construct the multiplex PyG Data Object
            patient_data = Data(
                x=x_matrix,
                edge_index_spatial=edge_index_spatial,
                edge_attr_spatial=edge_attr_spatial,
                edge_index_morph=edge_index_morph,
                edge_attr_morph=edge_attr_morph,
                y=y_label
            )
            
            data_list.append(patient_data)

        print(f"Compilation complete! Saving {len(data_list)} graphs to disk...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# --- QUICK TEST EXECUTION ---
# Running this script directly will trigger the build process!
if __name__ == "__main__":
    print("Initializing NeuroGraphDataset...")
    dataset = NeuroGraphDataset(
        root='./data', 
        csv_path='./data/GNN_Target_Labels.csv', # <--- Verify this path!
        spatial_matrix_path='./data/A_spatial_master.pt'
    )
    
    print("\n=== Dataset Successfully Loaded! ===")
    print(f"Total patients ready for training: {len(dataset)}")
    print(f"Sample Graph Architecture:\n{dataset[0]}")