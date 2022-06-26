from os import listdir

from keras.models import Model
from efficientnet.tfkeras import *
from keras.preprocessing.image import load_img, img_to_array

IMG_DIR = '../dataset/Images/'


def extract_features_effnet(version, images):
    configurations_for_effnet_version = {
        "effnetb0": {
            "img_size": 224,
            "model_function": EfficientNetB0()
        },
        "effnetb1": {
            "img_size": 240,
            "model_function": EfficientNetB1()
        },
        "effnetb2": {
            "img_size": 260,
            "model_function": EfficientNetB2()
        },
        "effnetb3": {
            "img_size": 300,
            "model_function": EfficientNetB3()
        },
        "effnetb4": {
            "img_size": 380,
            "model_function": EfficientNetB4()
        },
        "effnetb5": {
            "img_size": 456,
            "model_function": EfficientNetB5()
        },
        "effnetb6": {
            "img_size": 528,
            "model_function": EfficientNetB6()
        },
        "effnetb7": {
            "img_size": 600,
            "model_function": EfficientNetB7()
        }
    }

    model = configurations_for_effnet_version[version]['model_function']
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in images:
    # for name in listdir(directory)[:10]:    # proof of concept on the first 10 images
        filename = IMG_DIR + name
        img_size = configurations_for_effnet_version[version]['img_size']
        image = load_img(filename, target_size=(img_size, img_size))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)

    return features


if __name__ == '__main__':
    features = extract_features_effnet('effnetb0', IMG_DIR)
    print(features)