import os
import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv, GlobalAvgPool
from spektral.data import Graph, Dataset
from spektral.data.loaders import DisjointLoader
from sklearn.utils import shuffle as sklearn_shuffle
from tensorflow.keras import layers


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
    return patches, patch_numbers

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

    return np.array(all_patches), np.array(all_scores)

#########################################################################################################################################################################################################################################################################

# def image_to_graph(image, grid_size=16):
#     H, W = image.shape
#     num_nodes = (H // grid_size) * (W // grid_size)
#     nodes = np.zeros((num_nodes, grid_size*grid_size))
#     node_positions = []
#     edges = []
#     senders = []
#     receivers = []

#     index = 0
#     for y in range(0, H, grid_size):
#         for x in range(0, W, grid_size):
#             nodes[index] = image[y:y+grid_size, x:x+grid_size].flatten()
#             node_positions.append((y // grid_size, x // grid_size))
#             index += 1

#     for i, pos1 in enumerate(node_positions):
#         for j, pos2 in enumerate(node_positions):
#             if i != j and ((abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]) or (abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0])):
#                 edges.append(1)
#                 senders.append(i)
#                 receivers.append(j)

#     return {'nodes': nodes, 'edges': np.array(edges), 'senders': np.array(senders), 'receivers': np.array(receivers)}

# def image_to_graph(image, grid_size=16):
#     H, W = image.shape
#     num_nodes = (H // grid_size) * (W // grid_size)
#     nodes = np.zeros((num_nodes, grid_size*grid_size))
#     node_positions = []
    
#     index = 0
#     for y in range(0, H, grid_size):
#         for x in range(0, W, grid_size):
#             nodes[index] = image[y:y+grid_size, x:x+grid_size].flatten()
#             node_positions.append((y // grid_size, x // grid_size))
#             index += 1

#     # Initialize the adjacency matrix with zeros
#     a = np.zeros((num_nodes, num_nodes), dtype=int)

#     # Fill the adjacency matrix
#     for i, pos1 in enumerate(node_positions):
#         for j, pos2 in enumerate(node_positions):
#             if i != j and ((abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]) or (abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0])):
#                 a[i, j] = 1  # Assume undirected graph for simplicity

#     return {'nodes': nodes, 'a': a}
def image_to_graph(image, grid_size=16):
    H, W = image.shape
    num_nodes = (H // grid_size) * (W // grid_size)
    nodes = np.zeros((num_nodes, grid_size * grid_size))
    a = np.zeros((num_nodes, num_nodes), dtype=int)  # Adjacency matrix

    node_positions = [(y//grid_size, x//grid_size) for y in range(0, H, grid_size) for x in range(0, W, grid_size)]
    for i, pos1 in enumerate(node_positions):
        for j, pos2 in enumerate(node_positions):
            if i != j and ((abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]) or (abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0])):
                a[i, j] = 1

    nodes = [image[y:y+grid_size, x:x+grid_size].flatten() for y in range(0, H, grid_size) for x in range(0, W, grid_size)]
    return {'nodes': np.array(nodes), 'a': a}

#########################################################################################################################################################################################################################################################################


def split_data(combined, train_size=0.8, val_size=0.1):
    np.random.shuffle(combined)
    n_total = len(combined)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)
    train_data = combined[:n_train]
    val_data = combined[n_train:n_train + n_val]
    test_data = combined[n_train + n_val:]
    return train_data, val_data, test_data
  

#########################################################################################################################################################################################################################################################################


class SimpleGNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(32, activation='relu')
        self.conv2 = GCNConv(64, activation='relu')
        self.pool = GlobalAvgPool()
        self.dense1 = layers.Dense(128, activation='relu')
        self.classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.pool(x)
        x = self.dense1(x)
        return self.classifier(x)


#########################################################################################################################################################################################################################################################################


# class ImageGraphDataset(Dataset):
#     def __init__(self, data_list, **kwargs):
#         self.data_list = data_list
#         super().__init__(**kwargs)

#     def read(self):
#         return [Graph(x=data['nodes'], a=data['edges'], y=label) for data, label in self.data_list]

class ImageGraphDataset(Dataset):
    def __init__(self, data_list, **kwargs):
        self.data_list = data_list
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        for graph_data, label in self.data_list:  # Ensure correct unpacking here
            # graph_data should be a dictionary with 'nodes' and 'a'
            x = graph_data['nodes']
            a = graph_data['a']
            y = np.array([label], dtype=np.float32)  # Ensure labels are correctly shaped
            graphs.append(Graph(x=x, a=a, y=y))
        return graphs


#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################

patches, labels = load_data_from_csv(csv_path, original_dir, denoised_dir)

print(len(patches))
print(len(labels))

graph_data = [image_to_graph(patch.reshape(224, 224)) for patch in patches]
combined = list(zip(graph_data, labels))

train_data, val_data, test_data = split_data(combined)

print(f"Train Dataset Size: {len(train_data)}")
print(f"Validation Dataset Size: {len(val_data )}")
print(f"Test Dataset Size: {len(test_data )}")
print(train_data[0])  # Check the first item's structure

train_dataset = ImageGraphDataset(train_data)
val_dataset = ImageGraphDataset(val_data)
test_dataset = ImageGraphDataset(test_data)

train_loader = DisjointLoader(train_dataset, batch_size=32, epochs=10)
val_loader = DisjointLoader(val_dataset, batch_size=32)
test_loader = DisjointLoader(test_dataset, batch_size=32)


#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################


model = SimpleGNN()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_loader, validation_data=val_loader, steps_per_epoch=train_loader.steps_per_epoch, validation_steps=val_loader.steps_per_epoch)


#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################


test_labels = np.array([data.y for data in test_dataset.read()])
predictions = model.predict(test_loader)
predicted_labels = (predictions > 0.5).astype(int)


#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################


cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))


#########################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_labels, predictions)
plt.figure()
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('/Dataset/precision_recall_curve.png')
