#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
'''
@author: Jiawei Adrian Li
@contact: jli@jacobs-alumni.de / muhanxiaoquan@gmail.com
Better to run than curse the road !
'''

# import necessary packages
import tensorflow as tf
import argparse
import sys
import io
import time
import numpy as np
from models import Model  # import main model
from models.model_configs import params_model # import model parameters configuration
from data_loading.batch_iterator import batch_iterator # import batch iterator for training and evaluating
from data_loading.file_reader import process_file, vocab_builder, read_vocab_file # tools for data loading
from GPUmanager import GPUManager  # gpu manager (optional)
from omnibox import *              # toolkits (optional)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # some printing settings
np.set_printoptions(linewidth=236, precision=4, threshold=10000)

class Config:
        '''
        define paths, hyper params and so forth
        '''
        data_dir = '\DATA\WebQA.v1.0\WebQA.v1.0'
        train = jpath(data_dir, 'train_dataset_PNratio_0.5.pkl')
        dev = jpath(data_dir, 'dev.pkl')
        test = jpath(data_dir, 'test.pkl')
        vocab = jpath(data_dir, 'vocab.pkl')
        #experiment settings
        is_train = True
        model_ckpt = 'ckpt/model_ckpt'
        sim_mode = 'sim2'
        max_seq_len = 512
        train_epoch = 1000
        train_batch_size = 512
        eval_epoch = 5


def evaluate(model, sess, k,v,y, bz, mode):
        '''
        some evaluation function
        '''
        batcher = batch_iterator(k,v,y, batch_size=bz) # here we use batch_iterator to traversing whole eval data
        total_loss = []
        for kb, vb, yb in batcher.yield_batch(): # batch size batch
                feed_dict = {model.input_k:kb,
                             model.input_v:vb,
                             model.input_y:yb}
                if mode=='sim1':
                        loss = sess.run(model.sim1_loss, feed_dict=feed_dict)
                        total_loss.append(loss)
                if mode == 'sim2':
                        loss = sess.run(model.sim2_loss, feed_dict=feed_dict)
                        total_loss.append(loss)
        return np.mean(total_loss)   # return some metrics you want


def main(config=Config):
        '''
        main training pipeline
        :param config: config for data paths and so forth
        '''
        model_path = config.model_ckpt  # model save path
        gm = GPUManager()
        with gm.auto_choice():
                configProto = tf.ConfigProto(allow_soft_placement=True)
                configProto.gpu_options.allow_growth = True
                sess = tf.Session(config=configProto)
                # construct computational Graph
                # data loading
                word2id = read_vocab_file(config.vocab)
                k_train, v_train, y_train = process_file(config.train, word2id, config.max_seq_len)
                k_dev, v_dev, y_dev = process_file(config.dev, word2id, config.max_seq_len)
                k_test, v_test, y_test = process_file(config.test, word2id, config.max_seq_len)

                # init model
                model = Model(params_model)

                # init saver
                mysaver = tf.train.Saver(tf.trainable_variables())

                # do training
                if config.is_train:
                        # init all variables
                        sess.run(tf.global_variables_initializer())
                        # load old model if finetune
                        ckpt = tf.train.latest_checkpoint(model_path)
                        if ckpt is not None:
                                mysaver.restore(sess, ckpt)
                        # base loss for recording
                        best_loss = 100000.
                        train_loss = 0.0

                        # begin epochs iteration
                        for epoch in range(config.train_epoch):
                                epoch_total_loss = []  # record epoch average loss
                                count = 0
                                # define the batch iterators
                                batcher = batch_iterator(k_train, v_train, y_train, batch_size=config.train_batch_size)
                                for k_batch, v_batch, y_batch in batcher.yield_batch():
                                        feed_dict = {model.input_k: k_batch,
                                                     model.input_v: v_batch,
                                                     model.input_y: y_batch}

                                        # loss and opt
                                        if config.sim_mode == 'sim1':
                                                fetches = {'opt': model.opt1,
                                                           'loss': model.sim1_loss}
                                                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                                                train_loss = result['loss']
                                                epoch_total_loss.append(train_loss)
                                        elif config.sim_mode == 'sim2':
                                                fetches = {'opt': model.opt2,
                                                           'loss': model.sim2_loss}
                                                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                                                train_loss = result['loss']
                                                epoch_total_loss.append(train_loss)

                                        # (optional) can also eval during batch iterations if the data is so big
                                        print('\r[Train]:Epoch-{},batch-{}/{},current avg mse-{}'.format(epoch, count, batcher.num_batch, np.mean(epoch_total_loss)), end='')
                                        sys.stdout.flush()
                                        time.sleep(0.01)
                                        if count % 100 == 0 and count != 0: # every 100 batches and exclude the first batch
                                                dev_loss = evaluate(model,sess,k_dev, v_dev, y_dev, 512, mode)
                                                if dev_loss < best_loss:
                                                        best_loss = dev_loss
                                                        mysaver.save(sess=sess, save_path=config.model_ckpt)
                                                        print('\nUpdated model!')
                                                        print("[Eval]:Epoch-{},batch-{}/{},eval average mse loss:{}\n".format(epoch, count, batcher.num_batch, dev_loss), end='')
                                                print("\r[Eval]:Epoch-{},batch-{}/{},eval average mse loss:{}\n".format(epoch, count, batcher.num_batch, dev_loss), end='')
                                                sys.stdout.flush()
                                                time.sleep(0.01)
                                        count += 1  # record batch idx

                                epoch_avg_loss = np.mean(epoch_total_loss)
                                # eval during an epoch and at the end of an epoch
                                if epoch % config.eval_epoch == 0:
                                        dev_loss = evaluate(model, sess, k_dev, v_dev, y_dev, 64, config.sim_mode)
                                        if dev_loss < best_loss:
                                                best_loss = dev_loss
                                                mysaver.save(sess=sess, save_path=config.model_ckpt)
                                                print('Updated model !')
                                        print("[Eval]: Epoch - {} , eval average mse loss: {}".format(epoch, dev_loss))
                                print("[train]: Epoch - {} , train average mse loss: {}".format(epoch, epoch_avg_loss))


                # do testing / predicting
                elif not config.is_train:
                        ckpt = tf.train.latest_checkpoint(model_path)
                        if ckpt is not None:
                                mysaver.restore(sess, ckpt)
                        else:
                                raise FileNotFoundError('Cannot load model ckpt, plz check model path')

                        test_loss = evaluate(model, sess, k_test, v_test, y_test, 64, config.sim_mode)
                        print("[Test]: test mse: %.4f" % (test_loss))