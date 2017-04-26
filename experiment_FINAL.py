import sys
import tensorflow as tf
from vahun.Text import Text
import numpy as np
from vahun.tools import Timer
from vahun.tools import explog
from vahun.autoencoder import Autoencoder_ffnn
from vahun.variational_autoencoder import Variational_autoencoder
from vahun.genetic import evolution
from vahun.genetic import experiment
from vahun.tools import show_performance
from vahun.genetic import Settings
import argparse


def main(args=None):
    timer=Timer()

    size=200000
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    corpuses=[Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20'),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20'),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20.digraph_repl'),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20.digraph_repl')]
    print("Corpus list ready")
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder experiment')
    parser.add_argument("--corp_len", dest="corp_len", default=200000, type=int, help="Words to read from corpus")
    parser.add_argument("--feature_len", dest="feature_len", default=20, type=int, help="Feature size")
    parser.add_argument("--from", dest="from", type=int, help="lower boundary")
    parser.add_argument("--to", dest="to",  type=int, help="upper boundary")

    args = parser.parse_args()

    main(args)