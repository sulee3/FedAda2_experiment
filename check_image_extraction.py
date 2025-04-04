# import tensorflow as tf
import pathlib
import os
from PIL import Image
import numpy as np

test_directory = "/net/scratch/sulee/gld23k_build/test_np/"

location = test_directory
image_dict = {}
count = 0
for i,f in enumerate(os.listdir(location)): 
    feature_data = np.load(location+f)['x'].astype(np.float32)#.transpose(0,3,1,2)
    target_labels = np.load(location+f)['y'].squeeze(1)
    #raise ValueError(f"Feature data has shape {np.load(location+f)['x'].astype(np.float32).shape}")
    # Feature data has shape (8, 224, 224, 3)
    print(f"Validation feature data has shape {np.load(location+f)['x'].astype(np.float32).shape}, client {i}, label length {len(target_labels)}")
    image_dict[count] = feature_data
    count += 1
    if count == 5:
        break

print(image_dict[0].shape)

output_dir='/net/scratch/sulee/output_images_1/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count1 = 0
for key in image_dict.keys():
    for img_num in range(len(image_dict[key])):
        img = image_dict[key][img_num,:,:,:].astype('uint8')
        image = Image.fromarray(img)
        image.save(os.path.join(output_dir, f'image_{count1}.png'))
        print(f'Saved image_{count1}.png')
        count1 += 1