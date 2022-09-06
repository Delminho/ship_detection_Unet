import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

trained_model = tf.keras.models.load_model("trained_model", compile=False)


def predict_from_image(img_path, compare):
    """
    Using trained model to predict ship locations
    Args:
        img_path: string path to an image
        compare: True if you want to save plot image and prediction and False if you want to save just mask

    Returns:
        prediction: NumPy (96x96) array representation of predicted image
    """
    filename = img_path.split(sep='/')[-1]
    # Process image
    pred_img = tf.io.read_file(img_path)
    pred_img = tf.image.decode_jpeg(pred_img, channels=3)
    pred_img = tf.image.resize(pred_img, (96, 96))
    pred_img = tf.cast(pred_img, tf.float32) / 255.
    pred_img = tf.reshape(pred_img, (1, 96, 96, 3))
    # Make a prediction
    prediction = trained_model.predict(pred_img)
    prediction = prediction.reshape(96, 96)
    # Create plot with image and prediction
    if compare:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(pred_img.numpy().reshape(96, 96, 3))
        plt.axis('off')
        plt.title(f"Image")
        fig.add_subplot(1, 2, 2)
        plt.imshow(prediction)
        plt.axis('off')
        plt.title(f"Prediction")
        # Save figure
        plt.savefig("predictions/predicted_" + filename, bbox_inches='tight')
    else:
        plt.imsave("predictions/predicted_mask_" + filename, prediction)
    return prediction


parser = argparse.ArgumentParser()
parser.add_argument('img', type=str, help='image path')
parser.add_argument('-compare_flag', type=str, default='True')
args = parser.parse_args()
if args.compare_flag.lower() in ['f', 'false', 'n', 'no']:
    predict_from_image(args.img, compare=False)
else:
    predict_from_image(args.img, compare=args.compare_flag)

