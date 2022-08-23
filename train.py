import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model import unet_model
from model import DiceLoss

# CONSTANTS
IMAGE_SHAPE = (768, 768)
path = ''
IMAGE_PATH = os.path.join(path, 'data/Images/')
IMAGES_LIST = os.listdir(IMAGE_PATH)
IMAGES_LIST = sorted([IMAGE_PATH + i for i in IMAGES_LIST])


def rle_to_mask(rle):
    """
    Converts Encoded Pixels to image
    Args:
        rle: string of Encoded Pixels

    Returns:
        NumPy array representation of an image
    """
    shape_x = IMAGE_SHAPE[0]
    shape_y = IMAGE_SHAPE[1]
    if rle == '':
        return np.zeros((shape_x, shape_y, 1), dtype=np.float32)
    else:
        mask = np.zeros(shape_x * shape_y, dtype=np.float32)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = 1
            current_position += lengths[index]
        return np.flipud(np.rot90(mask.reshape(shape_y, shape_x, 1), k=1))


def image_mask_gen():
    """
    Generator for a (image, mask) set item
    Reads image from images folder and converts it to a normalized tensor
    Gets Encoded Pixels for the corresponding image from csv file and converts it to a mask
    Warning: Make sure IMAGES_LIST is sorted the same way as the csv file
    """
    for i in images_range:
        img = tf.io.read_file(IMAGES_LIST[i])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (256, 256))
        img = tf.cast(img, tf.float32) / 255.   # Normalizing input
        mask = rle_to_mask(df['EncodedPixels'].iloc[i])
        mask = tf.image.resize(mask, (256, 256))
        yield (img, mask)


if __name__ == "__main__":
    df = pd.read_csv('data/test_ship_segmentations_v3.csv')
    df['EncodedPixels'] += ' '
    df = df.groupby(['ImageId']).sum()
    df['EncodedPixels'] = df['EncodedPixels'].replace(0, '')

    # DOWNSAMPLE IMAGES WITH NO SHIPS
    images_range = [index for index in range(len(df)) if not (
                index % 3 and df['EncodedPixels'].iloc[index] == '')]  # Leaving approx 33% of images without ship

    training = tf.data.Dataset.from_generator(image_mask_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((256, 256, 3), (256, 256, 1)))

    unet = unet_model()
    print(unet.summary())
    unet.compile(optimizer='adam',
                 loss=DiceLoss(),
                 metrics=tf.keras.metrics.BinaryIoU())

    EPOCHS = 9
    BATCH_SIZE = 16
    train_dataset = training.batch(BATCH_SIZE)
    print(train_dataset.element_spec)
    model_history = unet.fit(train_dataset, epochs=4)
    plt.plot(model_history.history['loss'])
    plt.savefig("logs/model_loss15000x4.png")
    unet.save("trained_model15000x4")
