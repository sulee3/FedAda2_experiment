import tensorflow as tf
import pathlib
import os
from PIL import Image
import time

t = time.time()

# Define the path to the dataset
data_dir = pathlib.Path('/net/scratch/sulee/cache_new/landmark_subset/gld23k')

# Load and preprocess the dataset
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalize the dataset
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Create a function to save images
def save_images(dataset, num_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    for images, labels in dataset.take(1):
        for i in range(num_images):
            img = images[i].numpy()
            img = (img * 255).astype('uint8')
            image = Image.fromarray(img)
            image.save(os.path.join(output_dir, f'image_{count}.png'))
            print(f'Saved image_{count}.png with label: {int(labels[i])}')
            count += 1

# Save a few images from the training dataset
save_images(train_ds, num_images=9, output_dir='/net/scratch/sulee/output_images')

print(f"{time.time()-t} seconds elapsed")