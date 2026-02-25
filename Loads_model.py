import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet

#provide all necessery information of the model
IMAGE_SIZE = (299, 299)
EMBED_DIM = 512
FF_DIM = 512
SEQ_LENGTH = 25
VOCAB_SIZE = 15000

def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None)
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input
#yo config method chai serialize garna ko lagi rakheko....save hunxa but load garda serialize garna parxa tensorflow/keras ma
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
#serialize Positional embeding layer
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.out = layers.Dense(VOCAB_SIZE)
        self.droupout_1 = layers.Dropout(0.1)
        self.droupout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.causal_attention_mask(inputs)
        inputs = self.droupout_1(inputs, training=training)
        attention_mask_1 = causal_mask
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            attention_mask_1 = tf.minimum(combined_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=attention_mask_1
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs,
            attention_mask=padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.droupout_2(proj_out, training=training)
        preds = self.out(proj_out)
        return preds

    def causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)
#serialize decoder block
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.num_heads,
        })
        return config

class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def call(self, inputs, training=False):
        try:
            if isinstance(inputs, tuple) and len(inputs) == 2:
                img, caption = inputs
            else:
                img = inputs
                caption = tf.zeros((tf.shape(img)[0], SEQ_LENGTH), dtype=tf.int32)
            
            if self.image_aug and training:
                img = self.image_aug(img)
                
            img_embed = self.cnn_model(img)
            encoder_out = self.encoder(img_embed, training=training)
            decoder_out = self.decoder(caption, encoder_out, training=training)
            return decoder_out
        except Exception as e:
            print(f"Error in model call: {str(e)}")
            raise

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:, i, :], training=True)
                batch_loss += loss
                batch_acc += acc

            train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars))

        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:, i, :], training=False)
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
    
    def get_config(self):
            config = super().get_config()
            config.update({
                "cnn_model": keras.saving.serialize_keras_object(self.cnn_model),
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "decoder": keras.saving.serialize_keras_object(self.decoder),
                "num_caps_per_image": self.num_caps_per_image,
            })
            return config

    @classmethod
    def from_config(cls, config):
        cnn_model_config = config.pop("cnn_model")
        encoder_config = config.pop("encoder")
        decoder_config = config.pop("decoder")
        cnn_model = keras.saving.deserialize_keras_object(cnn_model_config)
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        decoder = keras.saving.deserialize_keras_object(decoder_config)
        return cls(cnn_model=cnn_model, encoder=encoder, decoder=decoder, **config)
##############

#yo vanda agadi chai hamro lrs schedule theo but aaile chai hamle tesle complexity badaune vayera teslai hatako...aaile chai aauta constant rate xa i.e 0.001

###############
# class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, post_warmup_learning_rate, warmup_steps, **kwargs):
#         super().__init__(**kwargs)
#         self.post_warmup_learning_rate = post_warmup_learning_rate
#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         global_step = tf.cast(step, tf.float32)
#         warmup_steps = tf.cast(self.warmup_steps, tf.float32)
#         warmup_progress = global_step / warmup_steps
#         warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
#         return tf.cond(
#             global_step < warmup_steps,
#             lambda: warmup_learning_rate,
#             lambda: self.post_warmup_learning_rate,
#         )
        
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "post_warmup_learning_rate": float(self.post_warmup_learning_rate),  # Ensure serializable
#             "warmup_steps": int(self.warmup_steps),  # Ensure serializable
#         })
#         return config

###########
#load garda problem aako vayera pahila dummy data ma model build garne ani load garne..

##############



def load_trained_model(weights_path):
    """Load a trained image captioning model from weights file"""
    cnn_model = get_cnn_model()
    encoder = TransformerEncoder(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=2)
    decoder = TransformerDecoder(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    
    try:
        dummy_img = tf.random.normal((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        dummy_seq = tf.random.uniform((1, SEQ_LENGTH), maxval=VOCAB_SIZE, dtype=tf.int32)
        _ = caption_model((dummy_img, dummy_seq))
        caption_model.load_weights(weights_path)
        print(f"Model weights loaded successfully from {weights_path}")
    except Exception as e:
        print(f"Error building/loading model: {str(e)}")
        raise
    
    return caption_model

# Example usage:
# model = load_trained_model('Saved_model/betters.weights.h5')