# -*- coding: utf-8 -*-
# http://www.robots.ox.ac.uk/~vgg/data/pets/
# 개 25종, 고양이 12종

import tensorflow as tf
import numpy as np
import glob

from PIL import Image

import utils
import skimage
import matplotlib.pyplot as plt
import os,datetime,time
from sys import getsizeof
import gzip, pickle, struct,sys
import matplotlib.pyplot as plt
import vgg16_trainable as vgg16

img_size = 224
root_dir = "F:\SoccerPlayer_detect_2"
category = os.listdir(root_dir)

########################################################
# category 별로 image를 가져와 npz 파일 만들어주는 역할
def convert_imgs_to_npz():
    category_num = 0

    for idx, cat in enumerate(category):
        image_dir = root_dir + "/" + cat
        files = glob.glob(image_dir + "/*.jpg")
        print('----', cat, '처리 중')
        y_train = np.array([], dtype=np.int)
        x_train = None
        for i, f in enumerate(files):
            img = utils.load_image(f, img_size=img_size, float_flag=False)
            if img.shape == (img_size, img_size, 3):
                img = img.reshape((1, img_size, img_size, 3))
                if x_train is None:
                    x_train = img
                else:
                    x_train = np.concatenate((x_train, img), 0)
                y_train = np.append(y_train, category_num)

        if x_train is not None:
            output_filename = './dataSet/dataSet_' + str(category_num) + '.npz'
            np.savez_compressed(output_filename, x_train=x_train, y_train=y_train)
            category_num += 1
            print(x_train.shape)
            print(len(y_train), getsizeof(x_train))
            print(y_train.shape)


########################################################
# image를 배열화 해놓은 npz 파일을 섞습니다.
def concat_shuffle_npz():
    n_split = 3

    category.sort()
    index_to_category = {i:ca for i, ca in enumerate(category)}
    index_to_category_list = [[int(a), b] for a, b in index_to_category.items()]

    npz_filenames = []
    for i in range(len(os.listdir('./dataSet/'))):
        npz_filenames.append('./dataSet/dataSet_' + str(i) + '.npz')

    x_train = None
    y_train = None
    for i in range(len(npz_filenames)):
        print(i)
        data = np.load(npz_filenames[i])
        x_train_temp = data['x_train']
        y_train_temp = data['y_train']
        if x_train_temp.shape == ():
            continue
        if x_train is None:
            x_train = x_train_temp
            y_train = y_train_temp
        else:
            x_train = np.concatenate((x_train, x_train_temp), 0)
            y_train = np.concatenate((y_train, y_train_temp), 0)

    np.savez_compressed("./dataSet/all_dataSet.npz", x_train = x_train, y_train = y_train)
    print(x_train.shape)
    print(getsizeof(x_train))

    # 합쳐진 image data가 너무 커서 메모리가 충분하지 못할 경우를 대비하여 image의 순서를 섞은
    # shuffle 파일 생성
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]

    imgs_per_shffle = int(len(x_train)/n_split)

    for i in range(n_split):
        print(i)
        np.savez_compressed("./dataSet/dataSet_shuffle_" + str(i),
                            x_train = x_train[i*imgs_per_shffle:(i+1)*imgs_per_shffle],
                            y_train = y_train[i*imgs_per_shffle:(i+1)*imgs_per_shffle],ind = index_to_category_list)

########################################################
# shuffle된 npz 파일 테스트 합니다
def test_npz():
    data = np.load('./dataSet 보관중/dataSet_shuffle_0.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    print(y_train.shape)
    print(y_train)

    ind = data['ind']
    index_to_category = {int(c[0]):c[1] for i,c in enumerate(ind)}

    choice = 150

    plt.figure()
    plt.title('correct answer: ' + index_to_category[y_train[choice]])
    plt.imshow(x_train[choice] / 255.0)
    plt.show()
    plt.close()

########################################################
#
def train_from_pretrained():
    n_class = len(os.listdir(root_dir))
    data = np.load('./dataSet/dataSet_shuffle_0.npz')

    x_train = data['x_train']
    y_train = data['y_train']

    y_train_one_hot = np.zeros((y_train.size, n_class), dtype=np.int)
    y_train_one_hot[np.arange(y_train.size), y_train] = 1

    batch_size =10
    cost_all = []

    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, n_class])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy', n_class = n_class)
    vgg.build(images, train_mode, int_image=True)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    for step in range(100000):
        batch_mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train_one_hot[batch_mask]
        _, c = sess.run([train, cost], feed_dict={images: x_batch, true_out: y_batch, train_mode: True})
        if step % 1000 == 0:
            cost_all.append([step, c])
            print(step,c)

    np.savetxt("cost.txt", cost_all)
    # test classification again, should have a higher probability about tiger

    # test accuracy 계산
    n_sample = 50
    n = int(x_train.shape[0] / n_sample)
    accuracy = 0
    for i in range(n):
        prob = sess.run(vgg.prob, feed_dict={images: x_train[i*50:(i+1)*50], train_mode: False})
        prob = np.argmax(prob, axis=1)
        y_batch = y_train[i*50:(i+1)*50]
        accuracy += np.sum(prob == y_batch) / float(n_sample)
    print("accuracy: ", accuracy/n)

    #plt.figure()
    #plt.title(category[y_batch[0]])
    #plt.imshow(x_batch[0]/255.0)
    #plt.show()
    #plt.close()
    #utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './test-save.npy')

########################################################
#
def predict_from_pretrained():
    n_class = len(os.listdir(root_dir))

    test_source = 1
    if test_source == 1:
        filenames = []
        for i in range(len(os.listdir('./test_picture/'))):
            filenames.append('./test_picture/' + str(i+1) + '.jpg')
        label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        test_batch=None
        for i in range(len(filenames)):
            img = utils.load_image(filenames[i],img_size=img_size,float_flag=False).reshape((1,img_size,img_size,3))
            if test_batch is None:
                test_batch = img
            else:
                test_batch = np.concatenate((test_batch,img), 0)

    elif test_source == 2:
        data = np.load('./dataSet/dataSet_shuffle_0.npz')

        test_batch = data['x_train']
        label = data['y_train']

        ndata = min(50, test_batch.shape[0])
        batch_mask = np.random.choice(test_batch.shape[0], ndata)

        test_batch = test_batch[batch_mask]
        label = label[batch_mask]

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg16.Vgg16('./test-save.npy', trainable = False, n_class = n_class)
        with tf.name_scope("content_vgg"):
            vgg.build(images,train_mode,int_image=True)

        prob = sess.run(vgg.prob, feed_dict = {train_mode: False, images: test_batch})
        #print(prob)
        prob_argmax = np.argmax(prob,axis=1)
        for i in range(min(100, test_batch.shape[0])):
            print(i, "who: ", category[label[i]], "\t\t | predict: ", category[prob_argmax[i]],
                  "\t\t | result: ", label[i] == prob_argmax[i])

        print("acc: ", np.sum(label==prob_argmax)/float(test_batch.shape[0]))

if __name__ == "__main__":
    s = time.time()
    # convert_imgs_to_npz()
    # concat_shuffle_npz()
    # test_npz();

    # train_from_pretrained()
    predict_from_pretrained()

    e=time.time()

    print("경과시간: ", e-s, "sec")