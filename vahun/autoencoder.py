import scipy
import tensorflow as tf
import numpy as np
import collections
import Levenshtein
from vahun.genetic import Settings

class Autoencoder():
    """
        abstract class
    """
    def __init__(self,
                 tf_session, inputdim,
                 logger,
                 layerlist,
                 encode_index,
                 corpus,
                 optimizer = tf.train.AdamOptimizer(),
                 nonlinear=tf.nn.sigmoid,
                 disp_step=30,
                charnum=0):
        
        self.charnum=charnum
        self.maxlen=int(inputdim/charnum)
        self.corpus=corpus
        self.logger=logger
        
        self.layerlist=layerlist
        self.layernum=len(layerlist)
        self.n_input = inputdim
        self.encode_index=encode_index
        self.display_step=disp_step
        self.nonlinearity=nonlinear
        self.optimizer_function=optimizer
        
        self.experiment=Settings(self.layerlist)
          
        self.create_graph()
        
        self.size=0
        nums=[self.n_input,layerlist]
        for i in range(1,len(nums)):
            self.size+=4*layerlist[i]*layerlist[i-1]
            
        init = tf.global_variables_initializer()
        self.sess = tf_session
        self.sess.run(init)
        self.saver = tf.train.Saver()
        

    def create_graph(self):
        raise NotImplementedError() #class is abstract!!!!!
        
    def partial_fit(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
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
    
        for i in range(int(len(X)/batch)+1):
            if start+batch >= len(X):
                batch_xs = X[start:]
                start=0
            else:
                batch_xs = X[start:(start + batch)]
                start+=batch
            leng=len(batch_xs)
            cur = self.sess.run(self.reconstruction, feed_dict = {self.x: batch_xs})
            
            y=np.reshape(cur,(leng,self.charnum*self.maxlen))
            reconstructionl[i*batch:i*batch+leng,:] = y
            
        if len(X)<batch:
            return self.recon(X)
        return reconstructionl
    
    def recon(self,X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X})
    
    def reconstruction_accuracy(self,dataX,dataY=None):
        """
        @return char accuracy, word accuracy, levenshtein avg error
        """
        if dataY is None:
            dataY=dataX

        success=False
        while(not success):
            try:
                reconstructed=self.reconstruct(dataX)
                success=True
            except:
                print("memoryerror during train reconstruction,using only the half data!")
                dataY=dataY[0:int(len(dataY)/2)]
                dataX=dataX[0:int(len(dataY)/2)]
                
        accuracy_max=len(dataX)*self.maxlen
        char_accuracy=len(dataX)*self.maxlen
        word_accuracy=len(dataX)
        
        levenshtein_error_sum=0
        
        for i in range(len(dataX)):
            a=dataY[i].reshape((self.maxlen,self.charnum))
            b=reconstructed[i].reshape((self.maxlen,self.charnum))
            
            word_ok=True
            
            for j in range(self.maxlen):
                if (a[j,:].argmax()!=b[j,:].argmax()):
                    char_accuracy-=1
                    word_ok=False
            if (not word_ok):
                word_accuracy-=1   
                
            levenshtein_error_sum+=Levenshtein.distance(
                self.corpus.defeaturize_data_charlevel_onehot([a])[0],
                self.corpus.defeaturize_data_charlevel_onehot([b])[0])
                
        characc=char_accuracy/accuracy_max
        wordacc=word_accuracy/len(dataX)

        return characc,wordacc,levenshtein_error_sum/len(dataX)
    
    def train(self,
              X_train,X_valid,X_test,
              batch_size=512,max_epochs=50,
              Y_train=None, Y_valid=None, Y_test=None):
        
        if Y_train is None:
            Y_train = X_train
        if Y_valid is None:
            Y_valid = X_valid
        if Y_test is None:
            Y_test = X_test
            
        breaker=False
        validlog=collections.deque(maxlen=30)
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
            #print(batch_xs.shape,batch_ys.shape)
            cost = self.partial_fit(batch_xs,batch_ys)
            #avg_cost += cost/ batch_size
            if i % self.display_step==0:
                validloss=self.calc_total_cost(X=X_valid,Y=Y_valid)
                #early stop
                self.logger.logline("train.log",["batch",i,"valid_loss",validloss])
                validlog.append(validloss)

                if len(validlog)>20:
                    breaker=True
                    for j in range(10):
                         if validlog[-j]<validlog[-10-j]:
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

        Vchar,Vword,Vleven=self.reconstruction_accuracy(dataX=X_valid,dataY=Y_valid)
        Testchar,Testword,Testleven=self.reconstruction_accuracy(dataX=X_test,dataY=Y_test)
        Trainchar,Trainword,Trainleven=self.reconstruction_accuracy(dataX=X_train,dataY=Y_train)
        
        self.logger.logline("accuracy.log",
                            [self.calc_total_cost(X_train),
                             self.calc_total_cost(X_valid),
                             self.calc_total_cost(X_test),
                             Trainchar,
                             Trainword,
                             Vchar,
                             Vword,
                             Testchar,
                             Testword,
                             Trainleven,
                             Vleven,
                             Testleven]+self.layerlist)                  
    
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
            
    def get_reconstruction_splitbrain(self,dataX,corp,dataY=None):
        if dataY is None:
            dataY=dataX
            
        enc_list=[]
        result=[]

        if isinstance(dataX,list):
            handmade=corp.featurize_data_charlevel_onehot(dataX)
            dataX=handmade.reshape((len(handmade), np.prod(handmade.shape[1:])))
        if isinstance(dataY,list):
            handmade=corp.featurize_data_charlevel_onehot(dataY)
            dataY=handmade.reshape((len(handmade), np.prod(handmade.shape[1:])))

        a=dataX
        b=self.reconstruct(a)
        
        inputlen=int(self.n_input/self.charnum)
        for i in range(len(dataX)):
            xa=corp.defeaturize_data_charlevel_onehot(
                [a[i].reshape(inputlen,self.charnum)])[0]
            y=corp.defeaturize_data_charlevel_onehot(
                [b[i].reshape(inputlen,self.charnum)])[0]
            ya=corp.defeaturize_data_charlevel_onehot(
                [dataY[i].reshape(inputlen,self.charnum)])[0]
            l=Levenshtein.distance(ya,y)
            stri=""
            striorig=""
            fullword=""
            origword=""
            for i in range(inputlen):
                stri+=' '
                striorig+=' '
            stri=list(stri)
            striorig=list(striorig)
            for i in range(1,inputlen):
                if xa[-i]=='_':
                    stri[-i]=y[-i]
                    striorig[-i]=ya[-i]
                else:
                    stri[-i]=xa[-i]
                    striorig[-i]=xa[-i]
            for i in range(inputlen):
                fullword+=stri[i]
                origword+=striorig[i]
            result.append([xa,ya,y,origword,fullword,l])
        return result