import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


def conv_block(inputs=None, n_filters=4, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = tf.keras.layers.Conv2D(n_filters,  # Number of filters
                                  kernel_size=(3, 3),  # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='he_normal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters,  # Number of filters
                                  kernel_size=(3, 3),  # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='he_normal')(conv)
    # If dropout_porb is not 0 using a Dropout layer
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = tf.keras.layers.MaxPool2D()(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(prev_input, skipped_input, n_filters=4):
    """
    Convolutional upsampling block

    Arguments:
        prev_input -- Input tensor from previous layer
        skipped_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """

    up = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size=(3, 3),
                                         strides=(2, 2),
                                         padding='same')(prev_input)

    # Merge the previous output and the skipped_input
    merge = tf.keras.layers.concatenate([up, skipped_input], axis=3)

    conv = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3),
                                  activation='relu', padding='same',
                                  kernel_initializer='he_normal')(merge)
    conv = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3),
                                  activation='relu', padding='same',
                                  kernel_initializer='he_normal')(conv)

    return conv


def unet_model(input_size=(96, 96, 3), n_filters=4, n_classes=1):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns:
        model -- tf.keras.Model
    """
    inputs = tf.keras.layers.Input(input_size)
    # Encoding
    cblock1 = conv_block(inputs, n_filters, dropout_prob=0.2)
    cblock2 = conv_block(cblock1[0], n_filters * 2, dropout_prob=0.2)
    cblock3 = conv_block(cblock2[0], n_filters * 4, max_pooling=False, dropout_prob=0.3)
    #cblock4 = conv_block(cblock3[0], n_filters * 8, max_pooling=False)

    # Decoding
    #ublock6 = upsampling_block(cblock4[0], cblock3[1], n_filters * 4)
    ublock7 = upsampling_block(cblock3[0], cblock2[1], n_filters * 2)
    ublock8 = upsampling_block(ublock7, cblock1[1], n_filters)

    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   (3, 3),
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(ublock8)

    # Output Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv10 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model