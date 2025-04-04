import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import os
import shutil
import time

def preprocess_stackoverflow(num_files, natural=1):
    """
    Preprocesses the Stackoverflow dataset from TensorFlow Federated (TFF) and saves the data into specified number of files.

    Parameters:
    - num_files (int): Number of output files.
    - natural (int): If 1, maintains non-IID structure by saving each client's data separately. 
      If 0, shuffles and splits data into specified number of files.

    Process:
    1. Loads the Stackoverflow dataset.
    2. Deletes existing preprocessed files and creates necessary directories.
    3. For natural=1:
       - Randomly selects num_files clients, saves each client's data in separate files.
    4. For natural=0:
       - Aggregates all data, shuffles, and splits into num_files files.

    Example usage:
    preprocess_stackoverflow(num_files=10, natural=1)
    preprocess_stackoverflow(num_files=10, natural=0)
    """
    start_time = time.time()
    
    print("Loading the Stackoverflow dataset...")
    stackoverflow_train, stackoverflow_test = tff.simulation.datasets.stackoverflow.load_data()
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    print("Deleting existing preprocessed files...")
    train_dir = 'data/stackoverflow/train_np/'
    test_dir = 'data/stackoverflow/test_np/'
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Directories prepared in {time.time() - start_time:.2f} seconds")

    np.random.seed(123)
    
    if natural == 1:
        print("Preprocessing data to maintain non-IID structure...")
        selected_train_clients = np.random.choice(stackoverflow_train.client_ids, num_files, replace=False)
        for client_id in selected_train_clients:
            print(f"Processing client {client_id}...")
            client_data = stackoverflow_train.create_tf_dataset_for_client(client_id)
            client_x, client_y = [], []
            for example in client_data:
                client_x.append(example['tokens'].numpy().astype(np.float32))
                client_y.append(example['tags'].numpy())
            client_x = np.array(client_x)
            client_y = np.array(client_y)
            np.savez(f'{train_dir}client_{client_id}.npz', x=client_x, y=client_y)
            print(f"Client {client_id} processed in {time.time() - start_time:.2f} seconds")
        
        selected_test_clients = np.random.choice(stackoverflow_test.client_ids, num_files, replace=False)
        for client_id in selected_test_clients:
            print(f"Processing test client {client_id}...")
            client_data = stackoverflow_test.create_tf_dataset_for_client(client_id)
            client_x, client_y = [], []
            for example in client_data:
                client_x.append(example['tokens'].numpy().astype(np.float32))
                client_y.append(example['tags'].numpy())
            client_x = np.array(client_x)
            client_y = np.array(client_y)
            np.savez(f'{test_dir}client_{client_id}.npz', x=client_x, y=client_y)
            print(f"Test client {client_id} processed in {time.time() - start_time:.2f} seconds")
    
    elif natural == 0:
        print("Preprocessing data to break non-IID structure...")
        all_train_x, all_train_y = [], []
        all_test_x, all_test_y = [], []
        
        for client_id in stackoverflow_train.client_ids:
            client_data = stackoverflow_train.create_tf_dataset_for_client(client_id)
            for example in client_data:
                all_train_x.append(example['tokens'].numpy().astype(np.float32))
                all_train_y.append(example['tags'].numpy())
        
        for client_id in stackoverflow_test.client_ids:
            client_data = stackoverflow_test.create_tf_dataset_for_client(client_id)
            for example in client_data:
                all_test_x.append(example['tokens'].numpy().astype(np.float32))
                all_test_y.append(example['tags'].numpy())
        
        all_train_x = np.array(all_train_x)
        all_train_y = np.array(all_train_y)
        all_test_x = np.array(all_test_x)
        all_test_y = np.array(all_test_y)
        
        np.random.seed(123)
        perm_train = np.random.permutation(len(all_train_y))
        perm_test = np.random.permutation(len(all_test_y))
        all_train_x, all_train_y = all_train_x[perm_train], all_train_y[perm_train]
        all_test_x, all_test_y = all_test_x[perm_test], all_test_y[perm_test]
        
        train_partition_size = len(all_train_x) // num_files
        test_partition_size = len(all_test_x) // num_files
        
        for i in range(num_files):
            start_train = i * train_partition_size
            end_train = start_train + train_partition_size
            np.savez(f'{train_dir}partition_{i}.npz', x=all_train_x[start_train:end_train], y=all_train_y[start_train:end_train])
            
            start_test = i * test_partition_size
            end_test = start_test + test_partition_size
            np.savez(f'{test_dir}partition_{i}.npz', x=all_test_x[start_test:end_test], y=all_test_y[start_test:end_test])
            print(f"Partition {i} processed in {time.time() - start_time:.2f} seconds")
    
    print(f"Preprocessing complete in {time.time() - start_time:.2f} seconds")

# Example usage
preprocess_stackoverflow(num_files=100, natural=1)
# preprocess_stackoverflow(num_files=10, natural=0)
