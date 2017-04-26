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

timer=Timer()

size=200000
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

corpuses=[Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20'),
          Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20'),
          Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20.digraph_repl'),
          Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20.digraph_repl')]

exps = []
ranger=range(20,30)
i=0
with open('/mnt/store/velkey/experiments') as f:
    for line in f:
        if(i in ranger):
            exps.append(line.strip().split('\t'))
        i+=1


for exper in exps:
    exper=[int(item) for item in exper]
    layerlist=exper[3:]
    settings=Settings(layerlist)
    if exper[1]==0 and exper[2]==0:
        corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20'
        typ=0
    if exper[1]==1 and exper[2]==0:
        corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20'
        typ=1
    if exper[1]==0 and exper[2]==1:
        corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20.digraph_repl'
        typ=2
    if exper[1]==1 and exper[2]==1:
        corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20.digraph_repl'
        typ=3
    
    corpus=corpuses[typ]
    name=(str("uniq_"+("variational_" if exper[0]==1 else "autoencoder_")+
                    ("top_" if exper[1]==1 else "random_")+
                    ("bigraph_" if exper[2]==1 else "uni_")))
    logger=explog(encoder_type=name,
              encoding_dim=0,
              feature_len=20,
              lang=corpus_path,
              unique_words=len(set(corpus.wordlist)),
              name=name,
              population_size=0,
              words=len(corpus.wordlist))
    print("starting experiment: ",exper)
    timer.add("experiment")
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    if(exper[0]==1):
        encoder=Variational_autoencoder(logger=logger,
                                        tf_session=sess,
                                        inputdim=len(corpus.abc)*20,
                                        encoding_size=settings.weights[0],
                                        corpus=corpus,
                                        optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                                        nonlinear=tf.sigmoid)
    else:
        encoder=Autoencoder_ffnn(
                         experiment=settings,
                         logger = logger,
                         tf_session=sess,
                         inputdim = len(corpus.abc)*20,
                         layerlist = settings.weights,
                         encode_index = int(len(settings.weights)/2),
                         corpus = corpus,
                         optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                         nonlinear = tf.sigmoid,
                         charnum=len(corpus.abc))
    
    encoder.train(corpus.x_train,corpus.x_valid,corpus.x_test,512,80)
    
    print("Finished in:", timer.get("experiment") ,"s")