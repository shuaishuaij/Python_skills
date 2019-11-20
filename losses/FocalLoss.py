#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
'''
@author: jwli9
@contact: jwli9@iflytek.com
Better to run than curse the road !
'''
import tensorflow as tf


def multi_category_focal_loss2_fixed(y_true, y_pred):
        '''
        :param y_true: [1,0,0,1,1,1,0,1,0,0,1] or [[0.1,0.2], [0.2,0.4]]
        :param y_pred: [0.12,0.24,0.23,...] or [[1,0], [0,1]]
        :return: focal loss
        '''
        epsilon = 1.e-7
        gamma=2.
        alpha = tf.constant(0.5, dtype=tf.float32)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

if __name__ == '__main__':
        pass



































































































