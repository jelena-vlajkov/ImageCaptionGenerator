from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEQ_LENGTH = 25

def preprocess_captions(captions):
    tv = TextVectorization(output_sequence_length=SEQ_LENGTH)
    tv.adapt(captions)
