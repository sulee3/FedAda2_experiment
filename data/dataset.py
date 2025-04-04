import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
from  .lda import Preprocess 
import numpy as np 
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import os
import gc
from transformers import MobileViTImageProcessor
import json
from PIL import Image

class Femnist_data():
    def __init__(self, NUM_CLIENTS, IID, BATCH_SIZE, data_dir='/home/sulee/leaf/data/femnist/data/all_data/'):
        """
        Initializes the Femnist_data class.

        Args:
            NUM_CLIENTS (int): Number of federated clients.
            IID (bool): Flag indicating whether the data distribution is IID.
            BATCH_SIZE (int): Batch size for DataLoaders.
            data_dir (str): Directory containing the FEMNIST .json files.
        """
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
        self.data_dir = data_dir  # Directory containing the .json files

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])   # ImageNet std
        ])

    class TransformedCustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            """
            Custom Dataset that applies transformations to the images.

            Args:
                images (np.ndarray): Array of image data.
                labels (np.ndarray): Array of labels.
                transform (callable, optional): Transformation to apply to images.
            """
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # Get the image and label
            image = self.images[idx]
            label = self.labels[idx]

            # Reshape the image to (28, 28)
            image = image.reshape(28, 28).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image, mode='L')  # 'L' mode for grayscale

            # Apply transformations
            if self.transform:
                pil_image = self.transform(pil_image)

            return pil_image, label

    def load_datasets(self):
        """
        Loads and partitions the FEMNIST dataset.

        Returns:
            trainloaders (list): List of DataLoaders for training data per client.
            valloaders (list): List of DataLoaders for validation data per client.
            testloader (DataLoader): DataLoader for the global test set.
        """
        # List all JSON files in the data directory
        data_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.json')]

        # Initialize a dictionary to hold all data
        all_data = {'users': [], 'num_samples': [], 'user_data': {}}

        # Load data from each JSON file
        for f in data_files:
            with open(f, 'r') as infile:
                data = json.load(infile)
                all_data['users'].extend(data['users'])
                all_data['num_samples'].extend(data['num_samples'])
                all_data['user_data'].update(data['user_data'])

        if self.IID:
            print("Loading IID data...")
            # Merge all data and shuffle
            all_images = []
            all_labels = []

            for user in all_data['users']:
                user_images = all_data['user_data'][user]['x']
                user_labels = all_data['user_data'][user]['y']
                all_images.extend(user_images)
                all_labels.extend(user_labels)

            # Convert lists to numpy arrays
            all_images = np.array(all_images, dtype=np.float32)
            all_labels = np.array(all_labels, dtype=np.int64)

            # Normalize images (pixel values from 0-255 to 0-1)
            all_images /= 255.0

            # Since transformations are handled in the Dataset class, no need to reshape here

            # Create a TransformedCustomDataset
            dataset = self.TransformedCustomDataset(all_images, all_labels, transform=self.transform)

            # Shuffle and partition the dataset among clients
            generator = torch.Generator().manual_seed(42)
            # Shuffle dataset indices
            total_size = len(dataset)
            indices = torch.randperm(total_size, generator=generator)
            shuffled_dataset = torch.utils.data.Subset(dataset, indices)

            # Determine partition sizes
            partition_size = total_size // self.NUM_CLIENTS
            lengths = [partition_size] * self.NUM_CLIENTS
            remainder = total_size % self.NUM_CLIENTS
            if remainder:
                lengths[-1] += remainder  # Add the remainder to the last client

            # Split the dataset
            client_datasets = random_split(shuffled_dataset, lengths, generator=generator)

            trainloaders = []
            valloaders = []
            test_datasets = []

            for client_ds in client_datasets:
                # Split each client's data into 80% train and 20% test
                len_test = int(0.2 * len(client_ds))
                len_train = len(client_ds) - len_test
                lengths = [len_train, len_test]
                client_train_ds, client_test_ds = random_split(client_ds, lengths, generator=generator)

                # Create DataLoaders
                trainloader = DataLoader(client_train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
                valloader = DataLoader(client_test_ds, batch_size=self.BATCH_SIZE)

                trainloaders.append(trainloader)
                valloaders.append(valloader)
                test_datasets.append(client_test_ds)

            # Combine all test datasets to create a global test set
            test_dataset = ConcatDataset(test_datasets)
            testloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE)

            print("IID data loading completed.")
            return trainloaders, valloaders, testloader

        else:
            print("Loading non-IID data...")
            # Non-IID: Use data per client
            trainloaders = []
            valloaders = []
            test_datasets = []

            users = all_data['users']
            # Limit the number of clients if necessary
            if self.NUM_CLIENTS < len(users):
                users = users[:self.NUM_CLIENTS]

            for idx, user in enumerate(users):
                user_images = all_data['user_data'][user]['x']
                user_labels = all_data['user_data'][user]['y']

                # Convert lists to numpy arrays
                user_images = np.array(user_images, dtype=np.float32)
                user_labels = np.array(user_labels, dtype=np.int64)

                # Normalize images
                user_images /= 255.0

                # Create a TransformedCustomDataset
                dataset = self.TransformedCustomDataset(user_images, user_labels, transform=self.transform)

                # Split into train/test (80% train, 20% test)
                len_test = int(0.2 * len(dataset))
                len_train = len(dataset) - len_test
                lengths = [len_train, len_test]
                generator = torch.Generator().manual_seed(42)
                ds_train, ds_test = random_split(dataset, lengths, generator=generator)

                # Create DataLoaders
                trainloader = DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True)
                valloader = DataLoader(ds_test, batch_size=self.BATCH_SIZE)

                trainloaders.append(trainloader)
                valloaders.append(valloader)
                test_datasets.append(ds_test)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(users):
                    print(f"Processed {idx + 1}/{len(users)} clients.")

            # Combine all test datasets to create a global test set
            test_dataset = ConcatDataset(test_datasets)
            testloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE)

            print("Non-IID data loading completed.")
            return trainloaders, valloaders, testloader

class CustomDataset(Dataset):
    def __init__(self, feature_data, target_labels):
        self.feature_data = feature_data
        self.target_labels = target_labels

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, idx):
        feature = self.feature_data[idx]
        target = self.target_labels[idx]
        return feature, target


class Mnist_data():
    def __init__(self, NUM_CLIENTS, IID, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
    
    def load_datasets(self):
        preprocess = Preprocess()
        iid = self.IID
        transform = transforms.Compose(
            [transforms.ToTensor()]#, transforms.Normalize((0.5), (0.5))]
        )
        trainset = MNIST("/net/scratch/sulee/fedada2_all/dataset/", train=True, download=True, transform=transform)
        testset = MNIST("/net/scratch/sulee/fedada2_all/dataset/", train=False, download=True, transform=transform)

        data_new = torch.zeros(60000,28, 28)
        count = 0
        for image, _ in trainset:
            data_new[count] = image
            count +=1
            
        partition_size = len(trainset) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if iid:
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds) // 10  # 10 % validation set
                len_train = len(ds) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        else:
            # Till here the same in maintained but then it loses its shape when it comes out of the create_lda_paritions
            flwr_trainset = (data_new, np.array(trainset.targets, dtype=np.int32))
            datasets,_ =  preprocess.create_lda_partitions(
                dataset=flwr_trainset,
                dirichlet_dist= None,
                num_partitions= self.NUM_CLIENTS,
                concentration=0.5,
                accept_imbalanced=True,
                seed=2,
            )
        # Split each partition into train/val and create DataLoader
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds[0]) // 10  # 10 % validation set
                len_train = len(ds[0]) - len_val
                lengths = [len_train, len_val]
                cd = CustomDataset(ds[0].astype(np.float32),ds[1])
                ds_train, ds_val = random_split(cd, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        return trainloaders, valloaders, testloader


class RGBToBGR:
    def __call__(self, img):
        # Convert PIL Image to numpy array
        np_img = np.array(img)
        # Swap channels: RGB to BGR
        np_img = np_img[:, :, ::-1]
        # Convert numpy array back to PIL Image
        return transforms.functional.to_pil_image(np_img)


class Cifar_data():
    def __init__(self, NUM_CLIENTS, IID, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
    
    def load_datasets(self):
        preprocess = Preprocess()
        iid = self.IID
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        trainset = CIFAR10("/net/scratch/sulee/fedada2_all/dataset/", train=True, download=True, transform=transform)
        testset = CIFAR10("/net/scratch/sulee/fedada2_all/dataset/", train=False, download=True, transform=transform)

        data_new = torch.zeros(50000, 3, 224, 224)
        count = 0
        for image, _ in trainset:
            data_new[count] = image
            count +=1   
        partition_size = len(trainset) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if iid:
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds) // 10  # 10 % validation set
                len_train = len(ds) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        else:
            # Till here the same in maintained but then it loses its shape when it comes out of the create_lda_paritions
            flwr_trainset = (data_new, np.array(trainset.targets, dtype=np.int32))
            del data_new
            gc.collect()
            datasets,_ =  preprocess.create_lda_partitions(
                dataset=flwr_trainset,
                dirichlet_dist= None,
                num_partitions= self.NUM_CLIENTS,
                concentration=0.001,#0.5,
                accept_imbalanced= False,
                seed= 12,
            )
            #raise ValueError("print4") <- I don't see this. OOM kill happens in create_lda_partitions.
            del flwr_trainset
            gc.collect()
        # Split each partition into train/val and create DataLoader
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds[0]) // 10  # 10 % validation set
                len_train = len(ds[0]) - len_val
                lengths = [len_train, len_val]
                cd = CustomDataset(ds[0].astype(np.float32),ds[1])
                ds_train, ds_val = random_split(cd, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        del datasets
        gc.collect()
        return trainloaders, valloaders, testloader


class Cifar100_data():
    def __init__(self, NUM_CLIENTS, IID, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
    
    def load_datasets(self):
        preprocess = Preprocess()
        iid = self.IID
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        trainset = CIFAR100("/net/scratch/sulee/fedada2_all/dataset/", train=True, download=True, transform=transform)
        testset = CIFAR100("/net/scratch/sulee/fedada2_all/dataset/", train=False, download=True, transform=transform)

        data_new = torch.zeros(50000, 3, 224, 224)
        count = 0
        for image, _ in trainset:
            data_new[count] = image
            count +=1   
        partition_size = len(trainset) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if iid:
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds) // 10  # 10 % validation set
                len_train = len(ds) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        else:
            # Till here the same in maintained but then it loses its shape when it comes out of the create_lda_paritions
            flwr_trainset = (data_new, np.array(trainset.targets, dtype=np.int32))
            del data_new
            gc.collect()
            datasets,_ =  preprocess.create_lda_partitions(
                dataset=flwr_trainset,
                dirichlet_dist= None,
                num_partitions= self.NUM_CLIENTS,
                concentration=0.001,#0.5,
                accept_imbalanced= False,
                seed= 12,
            )
            del flwr_trainset
            gc.collect()
        # Split each partition into train/val and create DataLoader
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds[0]) // 10  # 10 % validation set
                len_train = len(ds[0]) - len_val
                lengths = [len_train, len_val]
                cd = CustomDataset(ds[0].astype(np.float32),ds[1])
                ds_train, ds_val = random_split(cd, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        del datasets
        gc.collect()
        return trainloaders, valloaders, testloader


class WMT16Data:
    def __init__(self, NUM_CLIENTS,IID, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
    
    def load_datasets(self):
        dataset = load_dataset('wmt16', 'de-en')

        # Split training set into partitions to simulate the individual dataset
        partition_size = len(dataset['train']) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        datasets = random_split(dataset['train'], lengths)

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths)
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(dataset['test'], batch_size=self.BATCH_SIZE)
        return trainloaders, valloaders, testloader

class Stackoverflow():
    def __init__(self, NUM_CLIENTS, IID,BATCH_SIZE):
        self.NUM_CLIENTS= NUM_CLIENTS
        self.BATCH_SIZE= BATCH_SIZE
        self.IID= IID
        
    def non_iid_create_val_loaders(self):
        valloaders = []
        for i,f in enumerate(os.listdir("data/stackoverflow/test_np/")):
            #print(f)
            feature_data = np.load('data/stackoverflow/test_np/'+f)['x'].astype(np.float32)
            target_labels = np.load('data/stackoverflow/test_np/'+f)['y']
            ds_val = CustomDataset(feature_data,target_labels)
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return valloaders

    def non_iid_create_train_loaders(self):
        trainloaders = []
        for i,f in enumerate(os.listdir("data/stackoverflow/train_np/")):
            #print(f)
            feature_data = np.load('data/stackoverflow/train_np/'+f)['x'].astype(np.float32)
            target_labels = np.load('data/stackoverflow/train_np/'+f)['y']
            ds_train = CustomDataset(feature_data,target_labels)
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
        return trainloaders

    def create_iid_loaders(self):
        # Merge all the clients (train and test) data to create the trainset 
        trainset = {'x': [], 'y': []}
        
        for i,f in enumerate(os.listdir("data/stackoverflow/train_np/")):
            trainset['x'].append(np.load('data/stackoverflow/train_np/'+f)['x'].astype(np.float32))
            trainset['y'].append(np.load('data/stackoverflow/train_np/'+f)['y'])

        for i,f in enumerate(os.listdir("data/stackoverflow/test_np/")):
            trainset['x'].append(np.load('data/stackoverflow/test_np/'+f)['x'].astype(np.float32))
            trainset['y'].append(np.load('data/stackoverflow/test_np/'+f)['y'])

        trainset['x'] = np.concatenate(trainset['x'], axis=0)
        trainset['y'] = np.concatenate(trainset['y'])
        
        # Shuffle the data 
        trainset['x']= trainset['x'][:-1]
        trainset['y']= trainset['y'][:-1]
        np.random.seed(123)
        perm = np.random.permutation(len(trainset['y']))
        trainset['x'], trainset['y'] = trainset['x'][perm], trainset['y'][perm]
        trainset = CustomDataset(trainset['x'],trainset['y'])
        # create trainloader, valloaders and testloader from this shuffled data 
        print(len(trainset))
        partition_size = len(trainset) // self.NUM_CLIENTS
        remainder = len(trainset) % self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if remainder:
            lengths.append(remainder)
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        if remainder:
            datasets = datasets[:-1]
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return trainloaders,valloaders, valloaders[0]

    def load_datasets(self):
        if self.IID:
            trainloaders, valloaders, testloader = self.create_iid_loaders()
        else:
            trainloaders = self.non_iid_create_train_loaders()
            valloaders = self.non_iid_create_val_loaders()
            testloader = valloaders[0]
        return trainloaders, valloaders, testloader
    
class GLDV2():
    def __init__(self, NUM_CLIENTS, IID,BATCH_SIZE):
        self.NUM_CLIENTS= NUM_CLIENTS
        self.BATCH_SIZE= BATCH_SIZE
        self.IID= IID
        self.test_directory = "/net/scratch/sulee/gld23k_build/test_np/"
        self.train_directory = "/net/scratch/sulee/gld23k_build/train_np/" 
        
    def non_iid_create_val_loaders(self):
        valloaders = []
        location = self.test_directory
        for i,f in enumerate(os.listdir(location)): 
            feature_data = np.load(location+f)['x'].astype(np.float32).transpose(0,3,1,2)
            target_labels = np.load(location+f)['y'].squeeze(1)
            #raise ValueError(f"Feature data has shape {np.load(location+f)['x'].astype(np.float32).shape}")
            # Feature data has shape (8, 224, 224, 3)
            print(f"Validation feature data has shape {np.load(location+f)['x'].astype(np.float32).shape}, client {i}, label length {len(target_labels)}")
            ds_val = CustomDataset(feature_data,target_labels)
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return valloaders

    def non_iid_create_train_loaders(self):
        trainloaders = []
        location = self.train_directory
        for i,f in enumerate(os.listdir(location)):
            feature_data = np.load(location+f)['x'].astype(np.float32).transpose(0,3,1,2)
            target_labels = np.load(location+f)['y'].squeeze(1)
            print(f"Train feature data has shape {np.load(location+f)['x'].astype(np.float32).shape}, client {i}, label length {len(target_labels)}")
            ds_train = CustomDataset(feature_data,target_labels)
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
        return trainloaders

    def create_iid_loaders(self):
        # Merge all the clients (train and test) data to create the trainset 
        trainset = {'x': [], 'y': []}
        
        for i,f in enumerate(os.listdir(self.train_directory)):
            trainset['x'].append(np.load(self.train_directory+f)['x'].astype(np.float32).transpose(0,3,1,2))
            trainset['y'].append(np.load(self.train_directory+f)['y'].squeeze(1))

        for i,f in enumerate(os.listdir(self.test_directory)):
            trainset['x'].append(np.load(self.test_directory+f)['x'].astype(np.float32).transpose(0,3,1,2))
            trainset['y'].append(np.load(self.test_directory+f)['y'].squeeze(1))

        trainset['x'] = np.concatenate(trainset['x'], axis=0)
        trainset['y'] = np.concatenate(trainset['y'])
        
        # Shuffle the data 
        trainset['x']= trainset['x'][:-1]
        trainset['y']= trainset['y'][:-1]
        np.random.seed(123)
        perm = np.random.permutation(len(trainset['y']))
        trainset['x'], trainset['y'] = trainset['x'][perm], trainset['y'][perm]
        trainset = CustomDataset(trainset['x'],trainset['y'])
        # create trainloader, valloaders and testloader from this shuffled data 
        partition_size = len(trainset) // self.NUM_CLIENTS
        remainder = len(trainset) % self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if remainder:
            lengths.append(remainder)
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        if remainder:
            datasets = datasets[:-1]
        trainloaders = []
        valloaders = []
        
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return trainloaders,valloaders, valloaders[0]

    def load_datasets(self):
        if self.IID:
            trainloaders, valloaders, testloader = self.create_iid_loaders()
        else:
            trainloaders = self.non_iid_create_train_loaders()
            valloaders = self.non_iid_create_val_loaders()
            testloader = valloaders[0]
        return trainloaders, valloaders, testloader