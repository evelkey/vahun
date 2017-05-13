import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from vahun.corpus import Corpus as Corpus
import numpy as np
from vahun.tools import Timer
from vahun.tools import explog
#from vahun.autoencoder import Autoencoder_ffnn
from vahun.tools import show_performance
from vahun.genetic import Settings
from vahun.tools import get_reconstruction
import argparse
from vahun.Autoencoder_FFNN import Autoencoder_FFNN
from vahun.Autoencoder_Variational import Autoencoder_Variational
import fileinput


def main(args=None):
    timer=Timer()
    corpus=Corpus(' +-.abCcDdeEfgGhijkLlmNnopqrsSTtuvwxyZzáéíóöúüőű',20)
    encode=args.encode
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logger=explog(encoder_type="demo_autoencoder_segmented_"+str(encode),
                  encoding_dim=encode,
                  feature_len=20,
                  lang="HUN",
                  unique_words=0,
                  name="demo_autoencoder_top_segmented_"+str(encode),
                  population_size=0,
                  words=0,path="/home/velkey/tmp/")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    encoder=Autoencoder_FFNN(
                     logger=logger,tf_session=sess,
                     inputdim=len(corpus.abc)*20,
                     layerlist=[encode,len(corpus.abc)*20],
                     encode_index=1,corpus=corpus,
                     optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                     nonlinear=tf.sigmoid,disp_step=40,
                     charnum=len(corpus.abc))

    encoder.load(args.graph_path)
    inputlist=[str(line).replace('\n','') for line in fileinput.input('-')]
    result=encoder.get_reconstruction_splitbrain(inputlist,corpus,inputlist)

    with open('/mnt/store/velkey/output/log.txt', "a") as myfile:
        for it in result:
            print(it[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation command line runner')
    parser.add_argument('-i','--input', type=argparse.FileType('r'),default='-',help="Input stream")
    
    parser.add_argument('--graph_path', dest='graph_path', type=str,default="/mnt/store/velkey/graphs/auto500.graph",help='Path to the Corpus.')
    parser.add_argument('--encode', dest='encode', type=int,help='Number of neurons in encoder layer')
    
    args = parser.parse_args()

    main(args)