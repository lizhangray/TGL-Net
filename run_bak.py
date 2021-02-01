# Reference Paper: 
#   Z. Li, J. Zhang, R. Zhong, B. Bhanu, Y. Chen, Q. Zhang, H. Tang
#   Lightweight and Efficient Image Dehazing Network Guided by Transmission Estimation from Real-world Hazy Scenes
#    Sensors 2021, 21(3), 960; https://doi.org/10.3390/s21030960
# Any questions please contact：lizhangray@qq.com


import tensorflow as tf
import cv2
import numpy as np
import sys,getopt
import os
import time




checkpoint_dir=''
test_path =''
IMAGE_HEIGHT=0
IMAGE_WIDTH=0
batch_size=1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def getRadiance(atmLight=None, im=None, transmission=None):

    J = atmLight + (im - atmLight) / tf.clip_by_value(transmission, 0.01, 1)
    return J / tf.reduce_max(J)




def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.05, dtype=tf.float32, name='name')
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)


def inference(inputs):
    with tf.variable_scope("inference"):
        w1 = weight_variable([3, 3, 3, 3], name='w1')
        h1 = tf.nn.conv2d(inputs, w1, strides=[1, 1, 1, 1], padding='SAME')
        h2= tf.nn.max_pool(h1, [1, 5, 5, 1], [1, 5, 5, 1], padding='VALID', name='max')
        h, skip = encoder(h2, name='encoder')
        h = decoder(h, skip, name='decoder')
        w4 = weight_variable([3, 3, 3, 1], name='w4')
        h4 = tf.nn.conv2d(h, w4, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.sigmoid(h4) 
        logits = tf.image.resize_images(h, (IMAGE_HEIGHT, IMAGE_WIDTH), method=1)
        w6 = weight_variable([5, 5, 1, 1], name='w6')
        logits = tf.nn.conv2d(logits, w6, strides=[1, 1, 1, 1], padding='SAME')

    return logits


def n_enc_block(inputs, W, b, name):
    h = inputs
    with tf.variable_scope(name):
        h = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        h = tf.nn.bias_add(h, b, name='bias')
    skip = h
    return h, skip


def encoder(inputs, name='encoder'):
    with tf.variable_scope(name):
        we1 = weight_variable([3, 3, 3, 5], name='we1')
        bias_c1 = bias_variable([5])
        h, skip_1 = n_enc_block(inputs, we1, bias_c1, name='block_1')
        h = tf.nn.elu(h, name='elu')
		
        we2 = weight_variable([5, 5, 5, 5], name='we2')
        bias_c2 = bias_variable([5])
        h, skip_2 = n_enc_block(h, we2, bias_c2, name='block_2')
        h = tf.nn.elu(h, name='elu')

        we3 = weight_variable([5, 5, 5, 8], name='we3')
        bias_c3 = bias_variable([8])
        h, skip_3 = n_enc_block(h, we3, bias_c3, name='block_3')
        h = tf.nn.elu(h, name='elu')

    return h, [skip_3, skip_2,skip_1]


def n_dec_block(inputs, skip, w, output_shape, name):
    with tf.variable_scope(name):
        h = inputs + skip
        h = tf.nn.conv2d_transpose(h, w, output_shape, strides=[1, 1, 1, 1], padding="SAME")
    return h


def decoder(inputs, skip, name='decoder'):
    with tf.variable_scope(name):

        wd2 = weight_variable([5, 5, 5, 8], name='wd2')
        out2 = [batch_size, IMAGE_HEIGHT//5 , IMAGE_WIDTH//5 , 5]
        h = n_dec_block(inputs, skip[0], wd2, out2, name='block_4')
        h = tf.nn.elu(h, name='elu')

        wd3 = weight_variable([5, 5, 5, 5], name='wd3')
        out3 = [batch_size, IMAGE_HEIGHT//5 , IMAGE_WIDTH//5 , 5]
        h = n_dec_block(h, skip[1], wd3, out3, name='block_3')
        h = tf.nn.elu(h, name='elu')
		
        wd4 = weight_variable([3, 3, 3, 5], name='wd4')
        out4 = [batch_size, IMAGE_HEIGHT//5 , IMAGE_WIDTH//5 , 3]
        h = n_dec_block(h, skip[2], wd4, out4, name='block_41')
        h = tf.nn.elu(h, name='elu')

    logits = h
    return logits



def read_one_image(path):
    img = cv2.imread(path)/255
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return img


def main(argv):
    global checkpoint_dir
    global test_path
    global IMAGE_HEIGHT
    global IMAGE_WIDTH

    try:
        opts, args = getopt.getopt(argv,"hd:t:o:e:",["ckpt_d=","path_t=","out_path=","epoch="])
    except getopt.GetoptError:
        print('test.py -d <checkpoint_dir> -t <test_path> -o <output_path> -e <epoch>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -d <checkpoint_dir> -t <test_path> -o <output_path> -e <epoch>')
            sys.exit()
        elif opt in ("-d", "--ckpt_d"):
            checkpoint_dir = arg
        elif opt in ("-t", "--path_t"):
            test_path = arg
        elif opt in ("-o", "--out_path"):
            output_path = arg
        elif opt in ("-e", "--epoch"):
            epoch = arg
    print(checkpoint_dir)
    print(test_path)
    print(output_path)
    img = cv2.imread(test_path)/255
    IMAGE_HEIGHT = img.shape[0]
    IMAGE_WIDTH = img.shape[1]
    with tf.name_scope('input_data') as scope:
        X = tf.placeholder(tf.float32, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='X')

    graph = tf.Graph()
    sess = tf.Session(config=config)
    data = []
    data1 = read_one_image(test_path)
    data.append(data1)
    output1 = inference(X)
    output2 = getRadiance(1.0, X, output1)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('fail to load model')
    result = sess.run(output2, feed_dict = {X: data})
    result = result[0, :, :, :]
    result = result*255

    if os.path.isdir(output_path):
        print('dir exists')
    else:
        os.mkdir(output_path)
    
    if os.path.isdir(output_path+'/'+ 'result'):
        print('dir exists')
    else:
        os.mkdir(output_path+'/'+ 'result')


    print(output_path)


    output_filename=output_path+'/'+ 'result/'+ test_path.split("/")[-1]

    print(output_filename)


    cv2.imwrite(output_filename[:-4]+'.jpg', result)



if __name__=="__main__":
    main(sys.argv[1:])
