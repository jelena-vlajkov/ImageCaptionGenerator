import re
import numpy as np
import nltk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from transformers import caption_model

from preprocess import load_captions_data, train_val_split

IMAGES_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE

captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")
train_data, valid_data = train_val_split(captions_mapping)

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

vectorization = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=25,
    standardize=custom_standardization,
)
vectorization.adapt(text_data)

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset

train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)


caption_model.compile(loss=cross_entropy)

caption_model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)

caption_model.cnn_model.save('./ann/cnn_model.mod')
w = caption_model.encoder.get_weights()
np.save('./ann/encoder.mod', w)
w = caption_model.decoder.get_weights()
np.save('./ann/decoder.mod', w)

# caption_model.cnn = keras.models.load_model('./ann/cnn_model.mod')
# caption_model.encoder.set_weights(np.load('./ann/encoder.mod.npy', allow_pickle=True))
# caption_model.decoder.set_weights(np.load('./ann/decoder.mod.npy', allow_pickle=True))

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = 24
valid_images = list(valid_data.keys())

def get_caption(sample_img):
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    encoded_img = caption_model.encoder(img, training=False)

    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace(" <end>", "")
    decoded_caption = decoded_caption + ' <end>'
    return decoded_caption

scores = []
def get_bleu_scores():
  for valid_img in valid_images:
    true_sentances = valid_data[valid_img]
    references = []
    for sent in true_sentances:
      references.append(sent.split())
    hypothesis = get_caption(valid_img).split()
    bleu = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights = [1])
    scores.append(bleu)
get_bleu_scores()
bleu_score = sum(scores) / len(scores)
print(bleu_score)
