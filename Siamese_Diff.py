import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import cv2
import textwrap
import pandas as pd
import resource
from tensorflow.keras.regularizers import l1
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, Lambda
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam


models = []
class_1_accuracies = []

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/Data_Aug_GA/patch_label_median_verified2.csv'
result_file_path = "/Dataset/Results/New_Overall_results.csv"


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
    patch_number = 0

    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers


def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)
    
    all_original_patches = []
    all_denoised_patches = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        
        original_file_name = f"original_{row['original_image_name']}.raw"
        denoised_file_name = f"denoised_{row['original_image_name']}.raw"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        original_patches, original_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['original_image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers) 


        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['original_image_name']}")
            continue
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers


def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl


def save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):
    
    if path.exists(result_file_path):
    
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
    
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],            
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    
    df_metrics.to_csv(result_file_path, index=False)


def augmented_images_paired(data, num_augmented_images_per_original):
    
    augmented_pairs = []
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for original, denoised in data:  # 
        original_expanded = np.expand_dims(original, axis=0)
        denoised_expanded = np.expand_dims(denoised, axis=0)
        
        
        seed = np.random.randint(0, 10000)

        temp_generator_original = data_augmentation.flow(original_expanded, batch_size=1, seed=seed)
        temp_generator_denoised = data_augmentation.flow(denoised_expanded, batch_size=1, seed=seed)
        
        for _ in range(num_augmented_images_per_original):
            aug_original = next(temp_generator_original)[0]
            aug_denoised = next(temp_generator_denoised)[0]

            augmented_pairs.append((aug_original, aug_denoised))

    return augmented_pairs
    



def create_siamese_model(input_shape=(224, 224, 1)):
    def base_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        return Model(inputs, x)

    base_network = base_model(input_shape)
        
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
        
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    diff = Lambda(lambda tensors:(tensors[0] - tensors[1]))([processed_a, processed_b])
    predictions = Dense(2, activation='softmax')(diff)
        
    return Model(inputs=[input_a, input_b], outputs=predictions)


original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)


paired_patches = []
for original_patch, denoised_patch in zip(original_patches, denoised_patches):
    paired_patches.append([original_patch, denoised_patch])

print(len(paired_patches))


diff_patches_np, labels_np = prepare_data(paired_patches, labels)
print(f"Patch Shape {diff_patches_np.shape}")
print(f"label Shape {labels_np.shape}")



combined = list(zip(diff_patches_np, labels_np, denoised_image_names, all_patch_numbers))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

print(f"Ghosting Artifacts: {len(ghosting_artifacts)}")
print(f"Non Ghosting Artifacts: {len(non_ghosting_artifacts)}")

num_ghosting_artifacts = len(ghosting_artifacts)
num_non_ghosting_artifacts = len(non_ghosting_artifacts)
num_test_ghosting = 1500
num_test_non_ghosting = 1500


num_train_ghosting = num_ghosting_artifacts - num_test_ghosting
num_train_non_ghosting = num_non_ghosting_artifacts - num_test_non_ghosting


train_ghosting = ghosting_artifacts[num_test_ghosting:]
test_ghosting = ghosting_artifacts[:num_test_ghosting]

train_non_ghosting = non_ghosting_artifacts[num_test_non_ghosting:]
test_non_ghosting = non_ghosting_artifacts[:num_test_non_ghosting]


train_dataset = train_ghosting + train_non_ghosting
test_dataset = test_ghosting + test_non_ghosting

train_patches, train_labels, train_image_names, train_patch_numbers = zip(*train_dataset)
test_patches, test_labels, test_image_names, test_patch_numbers = zip(*test_dataset)


train_patches = np.array(train_patches)
train_labels = np.array(train_labels)

test_patches = np.array(test_patches)
test_labels = np.array(test_labels)


ghosting_patches = train_patches[train_labels == 1]

ghosting_patches_expanded = np.expand_dims(ghosting_patches, axis=-1)
augmented_images_result = augmented_images_paired(ghosting_patches_expanded, num_augmented_images_per_original=12)
augmented_images_np = np.stack(augmented_images_result)
augmented_labels = np.ones(len(augmented_images_np))

train_patches_expanded = np.expand_dims(train_patches, axis=-1)

print(f"Shape of Train_patches: {train_patches_expanded.shape}")
print(f"Shape of Augmented_patches: {augmented_images_np.shape}")

train_patches_combined = np.concatenate((train_patches_expanded, augmented_images_np), axis=0)

print(f"Shape of Augmented_labels: {augmented_labels.shape}")
print(f"Shape of Train_labels: {train_labels.shape}")

train_labels_combined = np.concatenate((train_labels, augmented_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(train_patches_combined, train_labels_combined, test_size=0.15, random_state=42)
X_train = [X_train[:, 0], X_train[:, 1]]
X_test = [X_test[:, 0], X_test[:, 1]]


y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print(f"Shape of test_patches[0]: {X_train[0].shape}")
print(f"Shape of test_patches[1]: {X_train[1].shape}")
print(f"Shape of test_labels: {y_test.shape}")



## Without Class Weight

opt = Adam(learning_rate=0.0001)
siam_wcw_model = create_siamese_model()
siam_wcw_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

wcw_model_checkpoint = ModelCheckpoint(filepath='/Dataset/new_Model/Siamese_Diff_wCW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
wcw_history = siam_wcw_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[wcw_model_checkpoint])
memMb_vgg19 =resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print("%5.1f MByte" % memMb_vgg19)

# With Class Weight

ng = len(train_patches[train_labels == 0])
ga =  len(train_patches[train_labels == 1])
total = ng + ga

imbalance_ratio = ng / ga  
weight_for_0 = (1 / ng) * (total / 2.0)
weight_for_1 = (1 / ga) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))


opt = Adam(learning_rate=0.0001)
siam_cw_model = create_siamese_model()
siam_cw_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


cw_model_checkpoint = ModelCheckpoint(filepath='/Dataset/new_Model/Siamese_Diff_CW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cw_history = siam_cw_model.fit(X_train, y_train, epochs=20, class_weight=class_weight, validation_data=(X_test, y_test), callbacks=[cw_model_checkpoint])
memMb_vgg19 =resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print("%5.1f MByte" % memMb_vgg19)

# With Class Balance
 

combined = list(zip(train_patches_combined, train_labels_combined))
combined = sklearn_shuffle(combined)

ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

print(f"Ghosting Artifacts: {len(ghosting_artifacts)}")
print(f"Non Ghosting Artifacts: {len(non_ghosting_artifacts)}")

num_ghosting_artifacts = len(ghosting_artifacts)


train_val_ghosting = ghosting_artifacts[:num_ghosting_artifacts]
train_val_non_ghosting = non_ghosting_artifacts[:num_ghosting_artifacts]

cb_train_dataset = train_val_ghosting + train_val_non_ghosting

print(f"Class balance train size {len(cb_train_dataset)}")

cb_train_patches, cb_train_labels = zip(*cb_train_dataset)

cb_train_patches, cb_test_patches, cb_train_labels, cb_test_labels = train_test_split(cb_train_patches, cb_train_labels, test_size=0.20, random_state=42)

cb_train_patches = np.array(cb_train_patches)
cb_train_labels = np.array(cb_train_labels)
cb_test_patches = np.array(cb_test_patches)
cb_test_labels = np.array(cb_test_labels)

cb_train_patches = [cb_train_patches[:, 0], cb_train_patches[:, 1]]
cb_train_labels = keras.utils.to_categorical(cb_train_labels, 2)

cb_test_patches = [cb_test_patches[:, 0], cb_test_patches[:, 1]]
cb_test_labels = keras.utils.to_categorical(cb_test_labels, 2)

print(len(cb_train_patches))
print(len(cb_test_patches))
print(f"Shape of test_patches[0]: {cb_test_patches[0].shape}")
print(f"Shape of test_patches[1]: {cb_test_patches[1].shape}")
print(f"Shape of test_labels: {cb_test_labels.shape}")

opt = Adam(learning_rate=0.0001)
siam_cb_model = create_siamese_model()
siam_cb_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

cb_model_checkpoint = ModelCheckpoint(filepath='/Dataset/new_Model/Siamese_Diff_CB.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cb_history = siam_cb_model.fit(cb_train_patches, cb_train_labels, epochs=20, class_weight=class_weight, validation_data=(cb_test_patches, cb_test_labels), callbacks=[cb_model_checkpoint])
memMb_vgg19 =resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print("%5.1f MByte" % memMb_vgg19)



## Testing

test_patches = np.array(test_patches)
test_patches = test_patches.reshape((3000, 2, 224, 224, 1))
test_patches = [test_patches[:, 0], test_patches[:, 1]] 

test_labels = np.array(test_labels)
test_labels = keras.utils.to_categorical(test_labels, 2)


## Without Class Weight

# test_loss, test_acc = siam_wcw_model.evaluate(test_patches, test_labels)
test_loss, test_acc = siam_wcw_model.evaluate([test_patches[0][:3000], test_patches[1][:3000]], test_labels[:3000])

test_acc  = test_acc *100

predictions = siam_wcw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)


report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_wCW_csv_path = '/Dataset/New_CSV/Siamese_Diff_wCW_misclassified_patches.csv'
misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []

for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_wCW_csv_path, index=False)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "Siamese"
feature_name = "Difference Map"
technique = "Without Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")


class_1_precision = report['Ghosting Artifact']['precision']
models.append(siam_wcw_model)
class_1_accuracies.append(class_1_precision)


## With Class Weight

test_loss, test_acc = siam_cw_model.evaluate([test_patches[0][:3000], test_patches[1][:3000]], test_labels[:3000])
test_acc  = test_acc *100

predictions = siam_cw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_CW_csv_path  = '/Dataset/New_CSV/Siamese_Diff_CW_misclassified_patches.csv'    

misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_CW_csv_path, index=False)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "Siamese"
feature_name = "Difference Map"
technique = "Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")


class_1_precision = report['Ghosting Artifact']['precision']
models.append(siam_cw_model)
class_1_accuracies.append(class_1_precision)


## With Class Balance

test_loss, test_acc = siam_cb_model.evaluate([test_patches[0][:3000], test_patches[1][:3000]], test_labels[:3000])
test_acc  = test_acc *100

predictions = siam_cb_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_CB_csv_path  = '/Dataset/New_CSV/Siamese_Diff_CB_misclassified_patches.csv'    
misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []

for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_CB_csv_path, index=False)


conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]


total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "Siamese"
feature_name = "Difference Map"
technique = "Class Balance"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")


class_1_precision = report['Ghosting Artifact']['precision']
models.append(siam_cb_model)
class_1_accuracies.append(class_1_precision)


## ENSEMBLE 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss

weights = np.array(class_1_accuracies) / np.sum(class_1_accuracies)
predictions = np.array([model.predict(test_patches)[:, 1] for model in models])
weighted_predictions = np.tensordot(weights, predictions, axes=([0], [0]))
predicted_classes = (weighted_predictions > 0.5).astype(int)
true_labels = np.argmax(test_labels, axis=-1)
test_acc = accuracy_score(true_labels, predicted_classes)

weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='weighted')
test_loss = log_loss(true_labels, weighted_predictions)

conf_matrix = confusion_matrix(true_labels, predicted_classes)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FN  
total_class_1 = TP + FP  
correctly_predicted_0 = TN  
correctly_predicted_1 = TP

test_acc = test_acc *100
weighted_precision = weighted_precision * 100
weighted_recall   = weighted_recall * 100
weighted_f1_score = weighted_f1_score * 100

accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

model_name = "Siamese"
feature_name = "Difference Map"
technique = "Ensemble"

save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)
print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")


misclass_En_csv_path = '/Dataset/New_CSV/Ensemble_Siamese_Diff_misclassified_patches.csv'

misclassified_indexes = np.where(predicted_classes != true_labels)[0]
misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_classes[index]

    probability_non_ghosting = 1 - weighted_predictions[index]
    probability_ghosting = weighted_predictions[index]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])
misclassified_df.to_csv(misclass_En_csv_path, index=False)
