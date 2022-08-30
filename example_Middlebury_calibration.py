import os
import cv2
import json
import sys
import math
import time
import numpy as np
import tensorflow as tf
import libs.QcImage as QcImage
from libs.TrainingSet import TrainingSet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


true_intensities = np.array([81.0, 228.0, 116.0, 69.0, 153.0, 184.0,
                    211.0, 103.0, 166.0, 65.0, 174.0, 213.0,
                    68.0, 100.0, 134.0, 290.0, 186.0, 111.0,
                    503.0, 328.3, 205.7, 112.7, 57.0, 27.0])

input_dim = 1024
n_l1 = 50
n_l2 = 20
z_dim = 1

VIS = True

def get_colours(image, ts, start=None, end=None):
    """
    Get colours of a colour chart image.
    """

    colours = []

    for i in range(len(ts.references)):
        anno = ts.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)
        colours.append([sample_bgr[0], sample_bgr[1], sample_bgr[2]])

    return np.array(colours)[start:end]

def estimate_coefficients1d(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    n = tf.cast(tf.shape(x), tf.float32)
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = tf.reduce_sum(y * x) - mean_y * mean_x * n
    SS_xx = tf.reduce_sum(x * x) - mean_x * mean_x * n

    # calculating regression coefficients
    b_1 = tf.div_no_nan(SS_xy, SS_xx)
    b_0 = mean_y - b_1 * mean_x
    return b_1, b_0

def dense(x, n1, n2, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2], trainable=False)
        bias = tf.get_variable("bias", shape=[n2], trainable=False)
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def decoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim, n_l2, 'd_dense_2'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_3'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output

def initialize_uninitialized(sess):
  global_vars = tf.global_variables()
  is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
  not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))


def calibrate_crf(sess, jpg_intensities, true_values):
    """
    Calibrate a camera response function by using the true intensity values of a Macbeth colour chart obtained from raw images.
    """
    xx, yy = jpg_intensities.shape

    with tf.variable_scope(tf.get_variable_scope()):
        params = tf.Variable(tf.random.normal(
                    [1, z_dim], mean=0.0, stddev=1.0, dtype=tf.float32), name='Decoder_input')
        decoder_crf = decoder(params, reuse=True)

    decoder_crf = tf.gather(decoder_crf, 0)

    cost_items = []
    for idx in range(xx):
        # transform intensity to index space and calculate the CRF-corrected intensity
        intensity_idxs = jpg_intensities[idx] * 1024 - 1
        intensity_idxs = np.array(intensity_idxs, dtype=np.int)
        interpreted = tf.gather(decoder_crf, intensity_idxs)

        # calculate the MSEs between the CRF-corrected and true intensity values
        W_intensity, b_intensity = estimate_coefficients1d(
                    interpreted, true_values)
        aligned = interpreted * W_intensity + b_intensity
        cost_items.append(tf.reduce_mean(tf.square(tf.subtract(aligned, true_values))))
    loss = tf.global_norm([cost_items])

    # training algorithm
    random_init_num = 10
    escape_rate_crf = 1E-6
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # initializing the variables
    init = tf.initialize_variables([params])

    initialize_uninitialized(sess)

    res_crf = None
    res_params = None
    min_error = sys.maxsize

    # Perform random initialisation
    for j in range(random_init_num):
        print('=====================')
        print('Random init num: ' + str(j))
        print('=====================')

        sess.run(init)

        epoch = 600
        prev_training_cost = sys.maxsize

        for step in range(epoch):
            _, training_cost = sess.run([optimizer, loss])

            if math.isinf(training_cost) or math.isnan(training_cost):
                break

            if np.abs(prev_training_cost - training_cost) <= escape_rate_crf:
                break

            if step % 100 == 0:
                print('cost: ' + str(training_cost))

            prev_training_cost = training_cost

        if min_error > prev_training_cost:
            min_error = prev_training_cost

            tmp_params = params.eval(session=sess)
            res_params = tmp_params
            res_params = res_params.reshape((res_params.size))
            res_crf = decoder_crf.eval(session=sess)

            print('=====Optimal CRF found=====')
            print('Param: ' + str(res_params))
            print('Cost: ' + str(min_error))
            print('===========================')

    return res_crf


    

if __name__ == '__main__':
    with open('./dataset/modified_Middlebury/tags.json') as json_data:
        obj = json.load(json_data)[0]
        ts = TrainingSet(obj)
        root = './dataset/modified_Middlebury'
        dirs = os.listdir(root)
        rmses = []

        sess = tf.Session()
        with tf.variable_scope(tf.get_variable_scope()):
            decoder(tf.ones(shape=(1, z_dim)))

        saver = tf.train.Saver()
        saver.restore(sess, save_path=tf.train.latest_checkpoint('./model/'))

        for dir in dirs:
            if not os.path.isdir(root+'/'+dir):
                continue

            print('###################################')
            print('Camera name: ' + dir)
            print('###################################')
        
            image_names = os.listdir(root+'/'+dir)

            # collection intensities of raw images
            raw_colours_b = []
            raw_colours_g = []
            raw_colours_r = []
            raw_colours_intensity = []
            for name in image_names:
                if not 'raw' in name or not '.png' in name:
                    continue
                image = cv2.imread(
                    root + '/' + dir + '/' + name, cv2.IMREAD_COLOR)
                colours = get_colours(image, ts, 0, 24) / 255.0
                raw_colours_b.append(colours[:,0])
                raw_colours_g.append(colours[:,1])
                raw_colours_r.append(colours[:,2])
                raw_colours_intensity.append((colours[:,0] + colours[:,1] + colours[:,2])/3)
            raw_colours_intensity = np.array(raw_colours_intensity)
            raw_colours_intensity_flat = np.reshape(raw_colours_intensity, (-1))

            # collecting intensities of jpg images
            jpg_colours_b = []
            jpg_colours_g = []
            jpg_colours_r = []
            jpg_colours_intensity = []
            for name in image_names:
                if not 'jpg' in name or not '.png' in name:
                    continue
                image = cv2.imread(
                    root + '/' + dir + '/' + name, cv2.IMREAD_COLOR)
                colours = get_colours(image, ts, 0, 24) / 255.0
                jpg_colours_b.append(colours[:,0])
                jpg_colours_g.append(colours[:,1])
                jpg_colours_r.append(colours[:,2])
                jpg_colours_intensity.append((colours[:,0] + colours[:,1] + colours[:,2])/3)
            jpg_colours_intensity_np = np.array(jpg_colours_intensity)

            # Perform calibration using four calibration images
            crf = calibrate_crf(sess, jpg_colours_intensity_np[0:4, :], true_intensities)

            x = np.linspace(0, 1, len(crf))
            if VIS:
                x = np.linspace(0, 1, crf.size)
                plt.plot(x, crf, color='g', linewidth=1.0)
                plt.plot(x, x, color='y')
                plt.scatter(jpg_colours_intensity_np,raw_colours_intensity)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.show()

            # compare the CRF-corrected jpg intensities with the raw intensities
            raw = np.interp(jpg_colours_intensity, x, crf)
            rmses.append(np.sqrt(mean_squared_error(raw[4:,:], raw_colours_intensity[4:,:])))

        print(len(rmses))
        print('mean: '+str(np.mean(rmses)))
        print('median: '+str(np.median(rmses)))
        print('std: '+str(np.std(rmses)))
        print('max: '+str(np.max(rmses)))
        print('95%: '+str(np.percentile(rmses, 95)))
