import tensorflow as tf

from corpus import Corpus
import numpy as np

from tools import Timer
from tools import explog
from autoencoders import Autoencoder_ffnn
from genetic import evolution
from genetic import experiment

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction=1
corp_path='/home/velkey/corp/webkorpusz.wpl'
corp=Corpus(corpus_path=corp_path,language="Hun",size=100000,encoding_len=10)
all_features=corp.featurize_data_charlevel_onehot(corp.hun_lower)
train=all_features[0:int(len(all_features)*0.8)]
test=all_features[int(len(all_features)*0.8):len(all_features)]
x_train = train.reshape((len(train), np.prod(train.shape[1:])))
x_test = test.reshape((len(test), np.prod(test.shape[1:])))
print(x_train.shape)
logger=explog(encoder_type="Autoencoder",
              encoding_dim=120,feature_len=10,
              lang="Hun",unique_words=len(set(corp.full)),
              name="Autoencoder_evolution",population_size=40,
              words=len(corp.hun_lower))


x23=evolution(x_train,x_test,40,120,360,config,logger=logger)
for i in range(10):        
    x23.evolve()
    logger.logline("evolution.log",["generation",i,"grade",x23.grade()])