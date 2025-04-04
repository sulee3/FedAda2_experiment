# Use tensorflow-federated 0.80.0, other versions throw different errors and 
# download different files. Will raise error due to md5 hash not matching.
# Go to tensorflow core code and comment out if statement raising the error and 
# rerun. Most likely, it will download all the data in .tar and extract. 
# Should see it download 000.tar to 499.tar. Ignore that, 
# and just locate .csv files containing the train/test split. Based on cache_dir
# cache_dir = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore'
# it is in
# csv_path = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore/datasets/mini_gld_train_split.csv'
# csv_path_test = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore/datasets/mini_gld_test.csv'
# Ignore the error tensorflow federated throws and go to exasperated_code.py. 

# To extract gld23k, see all of gldv2_import_parallel.py, exasperated_code.py, 
# gld23_make in that order. 

import tensorflow_federated as tff
import os
import time

t = time.time()

cache_dir = '/net/scratch/sulee/cache_parallel/landmark_subset_onemore'
os.makedirs(cache_dir, exist_ok=True)

gldv2_train, gldv2_test = tff.simulation.datasets.gldv2.load_data(
    gld23k=True,
    cache_dir=cache_dir
)

print(f"Time elapsed: {time.time()-t} seconds.")