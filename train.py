import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model import unet_model
from model import DiceLoss
import argparse


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
        img = tf.cast(img, tf.float32) / 255.  # Normalizing input
        mask = rle_to_mask(df['EncodedPixels'].iloc[i])
        mask = tf.image.resize(mask, (256, 256))
        yield (img, mask)


if __name__ == "__main__":
    # Inout image shape constant
    IMAGE_SHAPE = (768, 768)

    # Get paths for images folder and masks csv file
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='train images folder')
    parser.add_argument('csv', type=str, help='csv file path')
    args = parser.parse_args()

    # Create and preprocess dataframe from csv file
    df = pd.read_csv(args.csv)
    df['EncodedPixels'] += ' '
    df = df.groupby(['ImageId']).sum()
    df['EncodedPixels'] = df['EncodedPixels'].replace(0, '')
    print(len(df))

    # Create images list from an img folder
    path = ''
    IMAGE_PATH = os.path.join(path, args.img)
    IMAGES_LIST = os.listdir(IMAGE_PATH)
    IMAGES_LIST = sorted([IMAGE_PATH + i for i in IMAGES_LIST])

    # DOWNSAMPLE IMAGES WITH NO SHIPS
    images_range = [index for index in range(len(df)) if not (
            index % 3 and df['EncodedPixels'].iloc[index] == '')]  # Leaving approx 33% of images without ship

    ds = tf.data.Dataset.from_generator(image_mask_gen,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=((256, 256, 3), (256, 256, 1)))

    # Splitting data 70%/15%/15% training/validation/test sets
    train_size = int(len(images_range) * 0.7)
    val_size = int(len(images_range) * 0.15)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    unet = unet_model()
    print(unet.summary())
    optimizer = tf.keras.optimizers.Adam(0.0007)
    unet.compile(optimizer=optimizer,
                 loss=DiceLoss(),
                 metrics=[tf.keras.metrics.BinaryIoU()])

    EPOCHS = 10
    BATCH_SIZE = 32

    train_dataset = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_ds.batch(BATCH_SIZE)
    test_dataset = test_ds.batch(BATCH_SIZE)

    print(train_ds.element_spec)
    model_history = unet.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
    test = unet.evaluate(test_dataset)
    print(test)

    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("logs/model_loss.png")
    plt.close()
    plt.plot(model_history.history['binary_io_u'], label='iou')
    plt.plot(model_history.history['val_binary_io_u'], label='val_iou')
    plt.legend()
    plt.savefig("logs/model_iou.png")

    unet.save("trained_model")