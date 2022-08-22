import tensorflow as tf
import matplotlib.pyplot as plt
trained_model = tf.keras.models.load_model("trained_model", compile=False)


def predict_from_image(img):
    pred_img = tf.io.read_file(img)
    pred_img = tf.image.decode_jpeg(pred_img, channels=3)
    pred_img = tf.image.resize(pred_img, (256, 256))
    pred_img = tf.cast(pred_img, tf.float32) / 255.
    pred_img = tf.reshape(pred_img, (1, 256, 256, 3))
    prediction = trained_model.predict(pred_img)
    prediction = prediction.reshape(256, 256)
    plt.imsave("predicted_image.png", prediction)


predict_from_image('data/Images/0a0df8299.jpg')
