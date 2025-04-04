# To extract gld23k, see all of gldv2_import_parallel.py, exasperated_code.py, 
# gld23_make.py in that order. 

# Collect image data into clients. The paths file saves to needs to be used 
# in dataset.py when loading the data, specifically save_dir_train, save_dir_test. 

import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import shutil

start_time = time.time()

# Read the CSV file
csv_path = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore/datasets/mini_gld_train_split.csv'
df = pd.read_csv(csv_path)

csv_path_test = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore/datasets/mini_gld_test.csv'
df_test = pd.read_csv(csv_path_test)

# Base directory containing the images
base_dir = '/net/scratch/sulee/landmark_backup'

# Directory to save the processed .npz files
save_dir_train = '/net/scratch/sulee/gld23k_build/train_np'
save_dir_test = '/net/scratch/sulee/gld23k_build/test_np'

# If the directory already exists, remove it
if os.path.exists(save_dir_train):
    shutil.rmtree(save_dir_train)
os.makedirs(save_dir_train)
if os.path.exists(save_dir_test):
    shutil.rmtree(save_dir_test)
os.makedirs(save_dir_test)

# Mean and standard deviation for normalization (ImageNet values)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Dictionary to hold the data grouped by user_id
data_dict = {}

def load_data(df,data_dict,save_dir_train, train = True):
    count = 0

    # Ask what might make sense (this is like 10 pictures per client for validation, too small.)
    # Iterate over the rows of the CSV file
    for index, row in df.iterrows():
        if train:
            user_id = row['user_id']
        else: 
            # for validation, give some images to test on. 
            user_id = count % 233
            count += 1
        image_id = row['image_id']
        label = row['class']

        # Construct the file path
        a, b, c = image_id[0], image_id[1], image_id[2]
        file_path = os.path.join(base_dir, a, b, c, f"{image_id}.jpg")

        # Load the image
        image = Image.open(file_path)

        # Apply the resize transformation
        image = image.resize((224, 224))

        # Convert the image to a NumPy array
        # image_array = np.array(image)

        # Convert the image to a NumPy array and scale to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0

        # Normalize the image
        image_array = (image_array - mean) / std

        # Append the image and label to the data dictionary
        if user_id not in data_dict:
            data_dict[user_id] = {'x': [], 'y': []}
        data_dict[user_id]['x'].append(image_array)
        data_dict[user_id]['y'].append(label)

    # Save the data for each user_id
    for user_id, data in data_dict.items():
        x = np.array(data['x'])
        y = np.array(data['y']).reshape(-1, 1)
        np.savez(os.path.join(save_dir_train, f"user_{user_id}.npz"), x=x, y=y)

load_data(df,{},save_dir_train)
print("Data processing and saving completed successfully for train data.")
print(f"Total time: {-start_time + time.time()} seconds elapsed.")
print(("Data processing and saving starting for test data."))
start_time = time.time()
load_data(df_test,{},save_dir_test, train = False)
print("Data processing and saving completed successfully for test data.")
print(f"Total time: {-start_time + time.time()} seconds elapsed.")
