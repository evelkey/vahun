#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
import numpy as np
from vahun.corpus import Corpus
from vahun.genetic import evolution
from vahun.genetic import experiment
from vahun.tools import Timer
from vahun.tools import explog
from vahun.autoencoder import Autoencoder_ffnn
from vahun.variational_autoencoder import Variational_autoencoder
from vahun.tools import show_performance
from vahun.tools import show_reconstruction
import argparse


def main(args=None):
    encode=args.encoding_dim
    dictsize=args.corp_len
    popsize=args.pop_size
    corp_path=args.corp_path

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    corp=Corpus(corpus_path=args.corp_path,language="Hun",size=dictsize,encoding_len=args.feature_len,corpus_stream=None,printer=False)
    all_features=corp.featurize_data_charlevel_onehot(corp.hun_lower_unique)
    train=all_features[0:int(len(all_features)*0.8)]
    test=all_features[int(len(all_features)*0.8):len(all_features)]
    x_train = train.reshape((len(train), np.prod(train.shape[1:])))
    x_test = test.reshape((len(test), np.prod(test.shape[1:])))
    
    
    testcorp=Corpus(corpus_path=args.corp_path,language="Hun",size=1000000,encoding_len=args.feature_len,corpus_stream=args.infile,printer=False)
    testdata=corp.featurize_data_charlevel_onehot(corp.hun_lower_unique)
    testdata= testdata.reshape((len(testdata), np.prod(testdata.shape[1:])))
    
    logger=explog(encoder_type="Demo_"+str(encode),
              encoding_dim=encode,feature_len=10,
              lang="Hun",unique_words=0,
              name="auto_demo_uni"+str(encode),population_size=popsize,
              words=len(corp.hun_lower_unique))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    exp=experiment(encoded_width=10,layermax=10,layermin=2,maxw=10,minw=3,out_dim=380)
    #exp.weights=[348, 254, 10, 254, 348, 360]
    exp.weights=[args.encoding_dim, 380]
    exp.len=len(exp.weights)
    
    if args.encoder_type==0:
        encoder=Autoencoder_ffnn(experiment=exp,
                     logger=logger,tf_session=sess,
                     inputdim=380,
                     layerlist=exp.weights,
                     encode_index=int(exp.len/2),
                     optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                     nonlinear=tf.sigmoid)
    else:
        encoder=Variational_autoencoder(logger=logger,tf_session=sess,
                             inputdim=380,
                             encoding_size=args.encoding_dim,
                             optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                             nonlinear=tf.sigmoid)
    encoder.train(x_train,x_test,512,80)
    show_reconstruction(encoder,testdata,corp,length=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder command line runner')
    parser.add_argument("--encoder_type", dest="encoder_type", default=0, type=int, help="0=fully connected ffnn autoencoder, 1=variational ffnn autoencoder")
    parser.add_argument('--corp_path', dest='corp_path', type=str,default='/mnt/permanent/Language/Hungarian/Corp/Webkorpusz/webkorpusz.wpl',help='Path to the Corpus.')
    parser.add_argument("--encoding_dim", dest="encoding_dim", default=160, type=int, help='Encoding dimension')
    parser.add_argument("--corp_len", dest="corp_len", default=2000000, type=int, help="Words to read from corpus")
    parser.add_argument("--feature_len", dest="feature_len", default=10, type=int, help="Feature size")
    parser.add_argument('--infile', type=argparse.FileType('r'),default='-',help="Input stream")
    
    args = parser.parse_args()

    main(args)