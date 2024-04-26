import os
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/Data_Aug_GA/patch_label_median_verified2.csv'

#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################

def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
    y_size = width * height
    patches, patch_numbers = [], []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, 'rb') as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
            patch_numbers.append(len(patches) - 1)
    return np.array(patches), np.array(patch_numbers)

#########################################################################################################################################################################################################################################################################

def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)
    all_patches = []
    all_scores = []
    for _, row in df.iterrows():
        original_path = os.path.join(original_dir, f"original_{row['original_image_name']}.raw")
        denoised_path = os.path.join(denoised_dir, f"denoised_{row['original_image_name']}.raw")
        original_patches, _ = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, _ = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])
        all_patches.extend(np.array(denoised_patches) - np.array(original_patches))
        all_scores.extend([1 if float(score) > 0 else 0 for score in row['patch_score'].split(',')])

    return torch.tensor(all_patches, dtype=torch.float32), torch.tensor(all_scores, dtype=torch.long)

#########################################################################################################################################################################################################################################################################

def patches_to_graph(patches, labels):
    num_nodes = patches.shape[0]
    nodes = patches.view(num_nodes, -1)  
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  
            if labels[i] == labels[j]:
                edges.append((i, j))
                edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=nodes, edge_index=edge_index, y=labels)

#########################################################################################################################################################################################################################################################################

def load_and_create_graph(csv_path, original_dir, denoised_dir):
    patches, labels = load_data_from_csv(csv_path, original_dir, denoised_dir)
    
    # if not isinstance(patches, torch.Tensor):
    #     patches = torch.tensor(patches, dtype=torch.float32)  
    # if not isinstance(labels, torch.Tensor):
    #     labels = torch.tensor(labels, dtype=torch.long) 
    
    print(len(patches))
    print(len(labels))
    print(patches.shape)
    print(labels.shape)
    print(type(patches))
    print(type(labels))

    graph_data = patches_to_graph(patches, labels)
    return graph_data

#########################################################################################################################################################################################################################################################################

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = torch.device('cuda')
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device('cpu')

#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(50176, 1024) 
        self.conv2 = GCNConv(1024, 512)
        self.conv2 = GCNConv(512, 256)
        self.fc = torch.nn.Linear(256, 2)  

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=int).to(x.device))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#########################################################################################################################################################################################################################################################################

model = GCN().to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model)
# model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train_model(csv_path, original_dir, denoised_dir):
  
  # graph_data = load_and_create_graph(csv_path, original_dir, denoised_dir).to(device)
  
  graph_data = load_and_create_graph(csv_path, original_dir, denoised_dir)
  graph_data = graph_data.to(device)
  for epoch in range(10):
    optimizer.zero_grad()
    output = model(graph_data)
    loss = criterion(output, graph_data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss {loss.item()}')
      
train_model(csv_path, original_dir, denoised_dir)


