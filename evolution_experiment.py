import tensorflow as tf
import numpy as np
from vahun.corpus import Corpus
from vahun.tools import Timer
from vahun.tools import explog
from vahun.autoencoder import Autoencoder_ffnn
from vahun.genetic import evolution
from vahun.genetic import experiment
import argparse


def main(args=None):
	encode=args.encoding_dim
	dictsize=args.corp_len
	popsize=args.pop_size
	corp_path=args.corp_path

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True


	corp=Corpus(corpus_path=corp_path,language="Hun",size=dictsize,encoding_len=args.feature_len)
	all_features=corp.featurize_data_charlevel_onehot(corp.mark_list(corp.hun_lower_unique))
	train=all_features[0:int(len(all_features)*0.8)]
	test=all_features[int(len(all_features)*0.8):len(all_features)]
	x_train = train.reshape((len(train), np.prod(train.shape[1:])))
	x_test = test.reshape((len(test), np.prod(test.shape[1:])))
	print(x_train.shape)

	logger=explog(encoder_type="autoencoder_unique_with_mark"+str(encode),
				  encoding_dim=encode,feature_len=args.feature_len,
				  lang="Hun",unique_words=len(set(corp.full)),
				  name="autoencoder_unique_"+str(encode),population_size=popsize,
				  words=len(corp.hun_lower))


	x23=evolution(x_train,x_test,popsize,encode,args.feature_len*38,config,logger=logger)
	for i in range(3):        
		x23.evolve()
		logger.logline("evolution.log",["generation",i,"grade",x23.grade()])
		
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Autoencoder experiment runner')
	parser.add_argument('--corp_path', dest='corp_path', type=str,default='/mnt/permanent/Language/Hungarian/Corp/Webkorpusz/webkorpusz.wpl',	 help='Path to the Corpus.')
	parser.add_argument("--encoding_dim", dest="encoding_dim", default=60, type=int, help='Encoding dimension')
	parser.add_argument("--corp_len", dest="corp_len", default=2000000, type=int, help="Words to read from corpus")
	parser.add_argument("--pop_size", dest="pop_size", default=10, type=int, help="Population size")
	parser.add_argument("--feature_len", dest="feature_len", default=10, type=int, help="Feature size")
	
	args = parser.parse_args()
	main(args)