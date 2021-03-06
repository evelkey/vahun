import sys
import tensorflow as tf
from vahun.corpus import TrainXY_Corpus as Corpus
import numpy as np
from vahun.tools import Timer
from vahun.tools import explog
#from vahun.autoencoder import Autoencoder_ffnn
from vahun.tools import show_performance
from vahun.genetic import Settings
from vahun.tools import get_reconstruction

from vahun.Autoencoder_FFNN import Autoencoder_FFNN
from vahun.Autoencoder_Variational import Autoencoder_Variational


import argparse


def main(args=None):
    timer=Timer()

    size=args.corp_len
    corpus_list=['/mnt/store/velkey/mnsz2/webcorp.1M.CV',
                 '/mnt/store/velkey/mnsz2/brain_split.200k.maxlen20',
                 '/mnt/store/velkey/mnsz2/webcorp.full.enfilt.segmented']

    corpus=Corpus(corpus_path=corpus_list[args.type],size=size)
    
    
    exps = []
    for i in range(1,50):
        exps.append([0,args.type,i*20,args.feature_len*len(corpus.abc)])
        exps.append([1,args.type,i*20,args.feature_len*len(corpus.abc)])
        
    for exper in exps:
        exper=[int(item) for item in exper]
        layerlist=exper[2:]
    
        name=(str("uniq_"+("variational_" if exper[0]==1 else "autoencoder_")+
             ("CV_" if exper[1]==0 else ("segmented_" if exper[1]==1 else "even_odd_"))))
              
        logger=explog(encoder_type=name,
                  encoding_dim=min(layerlist),
                  feature_len=args.feature_len,
                  lang=corpus_list[args.type],
                  unique_words=len(corpus.wordlist),
                  name=name,
                  population_size=0,
                  words=len(corpus.wordlist))
              
        for k in range(2):
            print("starting experiment: ",exper)
            timer.add("experiment")

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            if(exper[0]==1):
                encoder=Autoencoder_Variational(
                    logger=logger,tf_session=sess,
                    inputdim=len(corpus.abc)*args.feature_len,
                    layerlist=layerlist,
                    encode_index=1,corpus=corpus,
                    optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                    nonlinear=tf.sigmoid,disp_step=50,
                    charnum=len(corpus.abc))
            else:
                encoder=Autoencoder_FFNN(
                     logger=logger,tf_session=sess,
                     inputdim=len(corpus.abc)*args.feature_len,
                     layerlist=layerlist,
                     encode_index=1,corpus=corpus,
                     optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                     nonlinear=tf.sigmoid,disp_step=50,
                     charnum=len(corpus.abc))

            encoder.train(corpus.x_train,corpus.x_valid,corpus.x_test,
                          512,40,
                          corpus.y_train,corpus.y_valid,corpus.y_test)

            print("Finished in:", timer.get("experiment") ,"s")
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder experiment')
    parser.add_argument("--corp_len", dest="corp_len", default=400000, type=int, help="Words to read from corpus")
    parser.add_argument("--feature_len", dest="feature_len", default=20, type=int, help="Feature size")
    parser.add_argument("--type", dest="type", type=int, help="type")

    args = parser.parse_args()

    main(args)