import scipy
import tensorflow as tf
import numpy as np
import collections
import Levenshtein

class Autoencoder_ffnn():
    def __init__(self,experiment,
                 tf_session, inputdim,
                 logger,
                 layerlist,encode_index,corpus,
                 optimizer = tf.train.AdamOptimizer(),
                 nonlinear=tf.nn.relu,
                 disp_step=30,
                charnum=0,
                maxlen=20):
        """
        """
        self.charnum=charnum
        self.maxlen=maxlen
        self.corpus=corpus
        self.experiment=experiment
        self.logger=logger
        
        self.layerlist=layerlist
        self.layernum=len(layerlist)
        self.n_input = inputdim
        self.encode_index=encode_index
        self.display_step=disp_step

        network_weights = self._initialize_weights()
        self.weights = network_weights  

        self._create_layers(nonlinear)

        # cost
        self.cost =  0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf_session
        self.sess.run(init)
        self.saver = tf.train.Saver()
        
        self.size=0
        nums=[self.n_input,layerlist]
        for i in range(1,len(nums)):
            self.size+=4*layerlist[i]*layerlist[i-1]
        

    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['w1']=tf.Variable(self.xavier_init(self.n_input, self.layerlist[0]))
        all_weights['b1'] = tf.Variable(tf.random_normal([self.layerlist[0]], dtype=tf.float32))
        
        for i in range(1,self.layernum):
            all_weights['w'+str(i+1)]=tf.Variable(self.xavier_init(self.layerlist[i-1], self.layerlist[i]))
            all_weights['b'+str(i+1)] = tf.Variable(tf.random_normal([self.layerlist[i]], dtype=tf.float32))

        return all_weights
    
    def _create_layers(self,nonlinearity=tf.nn.relu):
        """
        """
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        layer=nonlinearity(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.encoded=layer
        for i in range(1,self.layernum-1):
            if i==self.encode_index:
                self.encoded=layer
            layer=nonlinearity(tf.add(tf.matmul(layer, self.weights['w'+str(i+1)]), self.weights['b'+str(i+1)]))
            
        self.reconstruction=tf.add(tf.matmul(layer, self.weights['w'+str(self.layernum)]), self.weights['b'+str(self.layernum)])

    def partial_fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
        return cost

    def calc_total_cost(self, X,Y=None,batch=2048):
        if Y==None:
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
    
    def train(self,X_train,X_valid,X_test,batch_size,max_epochs,Y_train=None):
        if Y_train==None:
            Y_train=X_train
            
        breaker=False
        testlog=collections.deque(maxlen=30)
        self.logger.logline("train.log",["START"])
        self.logger.logline("train.log",["config"]+self.layerlist)
        total_batch = int(max_epochs*len(X_train) / batch_size)
        # Loop over all batches
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
                    self.logger.logline("early_stop.log",["config"]+self.layerlist)
                    self.logger.logline("early_stop.log",["train_cost",self.calc_total_cost(X_train)])
                    self.logger.logline("early_stop.log",["valid_last_results"]+list(testlog))
                    break
        self.logger.logline("train.log",["STOP"])
        #train_loss,test_loss,train_char_acc,train_word_acc,test_char_acc,test_word_acc,config
        self.logger.logline("accuracy.log",
                            [self.calc_total_cost(X_train),
                             self.calc_total_cost(X_valid),
                             self.calc_total_cost(X_test),
                             self.char_accuracy(X_train),
                             self.word_accuracy(X_train),
                             self.char_accuracy(X_valid),
                             self.word_accuracy(X_valid),
                             self.char_accuracy(X_test),
                             self.word_accuracy(X_test),
                             self.levenshtein_distance(X_train),
                             self.levenshtein_distance(X_valid),
                             self.levenshtein_distance(X_test)]+self.layerlist)
                          
    
    def xavier_init(self,fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)
    def save(self,path):
        self.saver.save(self.sess, path)
        
    def load(self,path):
        self.saver.restore(self.sess, path)

