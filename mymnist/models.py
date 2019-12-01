import base64
import datetime
import os
import random
import re

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.datasets import mnist
from keras.models import load_model
from keras.utils.np_utils import to_categorical

img_save_path = 'image_input'
mnist_save_path = 'image_mnist'
model_path = 'model\\mnist_self.h5'

model = load_model(model_path)
graph = tf.get_default_graph()


def conver_to_mnist(convas_jpg):
    rdint = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    img_str = re.search(r'base64,(.*)', convas_jpg).group(1)
    img_binary = base64.b64decode(img_str)
    img_np = np.frombuffer(img_binary, dtype=np.uint8)
    # raw image <- jpg
    input_image_original = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    # 画像を保存する場合
    cv2.imwrite(os.path.join(img_save_path, '{}_input_original.jpg'.format(rdint)), input_image_original)

    size = (28, 28)
    input_image_mnist = cv2.resize(input_image_original, size)
    cv2.imwrite(os.path.join(img_save_path, '{}_input_mnist_color.jpg'.format(rdint)), input_image_mnist)

    jpg = Image.fromarray(input_image_mnist)

    # 画像が RGB ならグレースケールに変換する。(28, 28, 3) -> (28, 28)
    if jpg.mode == 'RGB':
        img_gray = jpg.convert("L")

    img_gray.save(os.path.join(img_save_path, '{}_input_minist_gray.png'.format(rdint)))

    # 行列を1次元に変換する(28, 28) -> (1, 784) にする。
    data = np.array(img_gray).reshape(1, -1)
    mnist_data = 255 - data

    img_mnist = Image.fromarray(mnist_data.reshape(28, -1))
    img_mnist.save(os.path.join(img_save_path, '{}_input_minist_gray_2.png'.format(rdint)))

    if model_path == 'model\\mnist_net.h5':
        mnist_data = data.reshape(1, 28, 28, 1)
    else:
        mnist_data = mnist_data / 255

    return mnist_data


def predict(img):
    global graph
    with graph.as_default():
        score = model.predict(img)
        result = score.argmax()

    score = np.around(score, 2)

    return score, result


def predict_demo():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(10000, 784)

    x_test = x_test.astype('float32') / 255

    y_test = to_categorical(y_test, 10)

    score = (model.predict(x_test[0].reshape(1, 784))).argmax()

    return score


def save_all_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # y_test = to_categorical(y_test, 10)

    for idx in range(x_train.shape[0]):
        save_mnist_train(x_train[idx], y_train[idx])

    for idx in range(x_test.shape[0]):
        save_mnist_test(x_test[idx], y_test[idx])


def save_mnist_train(mnist_data, label):
    rdint = 100000000 + random.randint(0, 10000000)
    img_mnist = Image.fromarray(mnist_data.reshape(28, -1))
    img_mnist.save(os.path.join(mnist_save_path, '{}_minist_{}.png'.format(label, rdint)))


def save_mnist_test(mnist_data, label):
    rdint = 200000000 + random.randint(0, 10000000)
    img_mnist = Image.fromarray(mnist_data.reshape(28, -1))
    img_mnist.save(os.path.join(mnist_save_path, '{}_minist_{}.png'.format(label, rdint)))
