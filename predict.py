import tensorflow as tf
import matplotlib.pyplot as plt
from train import IMAGES_LIST
trained_model = tf.keras.models.load_model("trained_model10000x9", compile=False)


def predict_from_image(img_path, visualize=True):
    """
    Using trained model to predict ship locations
    Args:
        img_path: string path to an image
        visualize: True if you want to save plot with image and prediction to a predictions folder

    Returns:
        prediction: NumPy (256x256) array representation of predicted image
    """
    filename=img_path.split(sep='/')[-1]
    # Process image
    pred_img = tf.io.read_file(img_path)
    pred_img = tf.image.decode_jpeg(pred_img, channels=3)
    pred_img = tf.image.resize(pred_img, (256, 256))
    pred_img = tf.cast(pred_img, tf.float32) / 255.
    pred_img = tf.reshape(pred_img, (1, 256, 256, 3))
    # Make a prediction
    prediction = trained_model.predict(pred_img)
    prediction = prediction.reshape(256, 256)
    # Create plot with image and prediction
    if visualize:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(pred_img.numpy().reshape(256, 256, 3))
        plt.axis('off')
        plt.title(f"Image")
        fig.add_subplot(1, 2, 2)
        plt.imshow(prediction)
        plt.axis('off')
        plt.title(f"Prediction")
        # Save figure
        plt.savefig("predictions/predicted_" + filename, bbox_inches='tight')
    return prediction


# Predictions for last 100 images
for image_path in IMAGES_LIST[-100:]:
    predict_from_image(image_path, visualize=True)
