import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


def create_cnn_model(input_dim, output_dim, filters=64, dropout_rate=0.3, learning_rate=0.001):
    """Create CNN model for speech recognition"""
    # Inputs
    input_feat = layers.Input(shape=(None, input_dim), name="input")
    y_true = layers.Input(shape=(None,), name="y_true")
    input_length = layers.Input(shape=(1,), name="input_length")
    label_length = layers.Input(shape=(1,), name="label_length")

    # CNN layers
    x = layers.Conv1D(filters, 3, activation="relu", padding="same")(input_feat)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters * 2, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters * 2, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters * 4, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters * 4, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Output layer
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(output_dim, activation="softmax", name="output")(x)

    # Add CTC layer
    ctc_output = CTCLayer(name="ctc_loss")(y_true, output)

    model = keras.models.Model(
        inputs=[input_feat, y_true, input_length, label_length],
        outputs=ctc_output
    )

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    return model


def create_lstm_model(input_dim, output_dim, lstm_units=128, dropout_rate=0.3, learning_rate=0.001):
    """Create LSTM model for speech recognition"""
    # Inputs
    input_feat = layers.Input(shape=(None, input_dim), name="input")
    y_true = layers.Input(shape=(None,), name="y_true")
    input_length = layers.Input(shape=(1,), name="input_length")
    label_length = layers.Input(shape=(1,), name="label_length")

    # LSTM layers
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(input_feat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(output_dim, activation="softmax", name="output")(x)

    # Add CTC layer
    ctc_output = CTCLayer(name="ctc_loss")(y_true, output)

    model = keras.models.Model(
        inputs=[input_feat, y_true, input_length, label_length],
        outputs=ctc_output
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    return model


def create_transformer_model(input_dim, output_dim, d_model=128, num_heads=8, ff_dim=512, learning_rate=0.001):
    """Create Transformer model for speech recognition"""

    class PositionalEncoding(layers.Layer):
        def __init__(self, position, d_model):
            super(PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_angles(self, position, i, d_model):
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model)

            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])

            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]
            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    # Inputs
    inputs = layers.Input(shape=(None, input_dim), name="input")
    y_true = layers.Input(shape=(None,), name="y_true")
    input_length = layers.Input(shape=(1,), name="input_length")
    label_length = layers.Input(shape=(1,), name="label_length")

    # Embedding
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(1000, d_model)(x)

    # Transformer blocks
    for _ in range(4):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(x, x)
        x = layers.LayerNormalization()(x + attn_output)

        # Feed forward
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        x = layers.LayerNormalization()(x + ffn_output)

    # Output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(output_dim, activation="softmax", name="output")(x)

    # Add CTC layer
    ctc_output = CTCLayer(name="ctc_loss")(y_true, output)

    model = keras.models.Model(
        inputs=[inputs, y_true, input_length, label_length],
        outputs=ctc_output
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    return model


def decode_predictions(predictions, num_to_char):
    """Decode model predictions to text"""
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    results = keras.backend.ctc_decode(predictions,
                                       input_length=input_len,
                                       greedy=True)[0][0]

    texts = []
    for result in results:
        text = ""
        for idx in result:
            if idx != -1:
                text += num_to_char.get(idx.numpy(), '')
        texts.append(text)

    return texts