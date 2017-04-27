import scipy
import tensorflow as tf
import numpy as np
import collections
import Levenshtein

class Variational_autoencoder():
    def __init__(self,
                 tf_session, inputdim,
                 logger,
                 encoding_size,corpus,
                 optimizer = tf.train.AdamOptimizer(),
                 nonlinear=tf.sigmoid,
                 disp_step=20,
                charnum=0,
                maxlen=20):
        """
        """
        self.encoding_size=encoding_size
        self.charnum=charnum
        self.maxlen=maxlen
        self.logger=logger

        self.corpus=corpus
        
        self.n_input = inputdim
        self.n_hidden=encoding_size
        self.display_step=disp_step

        network_weights = self._initialize_weights()
        self.weights = network_weights  

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        
        self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])
        self.encoded=self.z_mean
        
        # sample from gaussian distribution
        eps = tf.random_normal([tf.shape(self.x)[0], self.n_hidden], 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

        # cost
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        #tf stuff init
        init = tf.global_variables_initializer()
        self.sess = tf_session
        self.sess.run(init)
        self.saver = tf.train.Saver()
        
        self.size=encoding_size
        

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(self.xavier_init(self.n_input, self.n_hidden))
        all_weights['log_sigma_w1'] = tf.Variable(self.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X,Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,self.y: Y})
        return cost

    def calc_total_cost(self, X,Y=None,batch=2048):
        if Y is None:
            Y=X
        cost=0
        start=0
        for i in range(int(len(X)/batch)):
            if start+batch >= len(X):
                start=0
            start+=batch
            batch_xs = X[start:(start + batch)]
            batch_ys = Y[start:(start + batch)]
            cost+=self.sess.run(self.cost, feed_dict = {self.x: batch_xs,self.y: batch_ys})
        return cost

    def encode(self, X):
        return self.sess.run(self.encoded, feed_dict={self.x: X})

    def decode(self, encoded = None):
        if encoded is None:
            encoded = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.encoded: encoded})

    def reconstruct(self, X,batch=512):
        start=0
        reconstructionl=np.zeros([int(len(X)),self.charnum*self.maxlen])
        for i in range(int(len(X)/batch)):
            if start+batch >= len(X):
                batch_xs = X[start:]
                start=0
            else:
                batch_xs = X[start:(start + batch)]
                start+=batch
            leng=len(batch_xs)
            cur=self.sess.run(self.reconstruction, feed_dict = {self.x: batch_xs})
            reconstructionl[i*batch:i*batch+leng,:]=(np.reshape(cur,(leng,self.charnum*self.maxlen)))
        if len(X)<batch:
            return self.recon(X)
        return reconstructionl
    
    def recon(self,X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X})
    
    def char_accuracy(self,data):
        accuracy_max=len(data)*self.maxlen
        accuracy=len(data)*self.maxlen
        
        reconstructed=self.reconstruct(data)
        for i in range(len(data)):
            a=data[i].reshape((self.maxlen,self.charnum))
            b=reconstructed[i].reshape((self.maxlen,self.charnum))
            for j in range(self.maxlen):
                if (a[j,:].argmax()!=b[j,:].argmax()):
                    accuracy-=1
        return accuracy/accuracy_max
    
    def word_accuracy(self,data):
        accuracy_max=len(data)
        accuracy=len(data)
        
        reconstructed=self.reconstruct(data)
        for i in range(len(data)):
            a=data[i].reshape((self.maxlen,self.charnum))
            b=reconstructed[i].reshape((self.maxlen,self.charnum))
            for j in range(self.maxlen):
                if (a[j,:].argmax()!=b[j,:].argmax()):
                    accuracy-=1
                    break
        return accuracy/accuracy_max
    
    def levenshtein_distance(self,data):
        fulldist=0
        avgdist=0
        reconstructed=self.reconstruct(data)
        dists=[Levenshtein.distance(self.corpus.defeaturize_data_charlevel_onehot([           data[i].reshape((self.maxlen,self.charnum))])[0],self.corpus.defeaturize_data_charlevel_onehot([reconstructed[i].reshape((self.maxlen,self.charnum))])[0]) for i in range(len(data))]
            
        return np.average(dists)
            
        return np.sum(dists),np.average(dists)
    
    def train(self,Y_train,X_valid,X_test,batch_size,max_epochs,X_train=None):
        if X_train is None:
            X_train=Y_train
        breaker=False
        testlog=collections.deque(maxlen=30)
        self.logger.logline("train.log",["START"])
        self.logger.logline("train.log",["config"]+[self.n_hidden,self.n_input])
        
        total_batch = int(max_epochs*len(X_train) / batch_size)
        start=0
        for i in range(total_batch):
            start+=batch_size
            if start+batch_size >= len(X_train):
                start=0
            batch_xs = X_train[start:(start + batch_size)]
            batch_ys = Y_train[start:(start + batch_size)]
            cost = self.partial_fit(batch_xs,batch_ys)
            #avg_cost += cost/ batch_size
            if i % self.display_step==0:
                testloss=self.calc_total_cost(X_valid)
                #early stop
                self.logger.logline("train.log",["batch",i,"valid_loss",testloss])
                testlog.append(testloss)

                if len(testlog)>20:
                    breaker=True
                    for j in range(10):
                         if testlog[-j]<testlog[-10-j]:
                            breaker=False
                else:
                    breaker=False

                if breaker:
                    self.logger.logline("early_stop.log",["STOPPED"])
                    self.logger.logline("early_stop.log",["survived",i])
                    self.logger.logline("early_stop.log",["train_cost",self.calc_total_cost(X_train)])
                    self.logger.logline("early_stop.log",["valid_last_results"]+list(testlog))
                    break
        self.logger.logline("train.log",["STOP"])
        #train_loss,test_loss,train_char_acc,train_word_acc,test_char_acc,test_word_acc,config
        self.logger.logline("accuracy.log",
                            [self.calc_total_cost(X_train),
                             self.calc_total_cost(X_valid),
                             self.calc_total_cost(X_test),
                             self.char_accuracy(X_train[:int(len(X_train)*0.7)]),
                             self.word_accuracy(X_train[:int(len(X_train)*0.7)]),
                             self.char_accuracy(X_valid),
                             self.word_accuracy(X_valid),
                             self.char_accuracy(X_test),
                             self.word_accuracy(X_test),
                             self.levenshtein_distance(X_train),
                             self.levenshtein_distance(X_valid),
                             self.levenshtein_distance(X_test),
                             self.encoding_size,self.charnum*self.maxlen])
                          
    def xavier_init(self,fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)
    
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
    def save(self,path):
        self.saver.save(self.sess, path)
        
    def load(self,path):
        self.saver.restore(self.sess, path)

