import tensorflow as tf
import numpy as np
from vahun.autoencoder import Autoencoder

class Autoencoder_FFNN(Autoencoder):          

    def create_graph(self):
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self._create_layers()
        # cost
        self.cost =  0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0))
        self.optimizer = self.optimizer_function.minimize(self.cost)
    
    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['w1']=tf.Variable(self.xavier_init(self.n_input, self.layerlist[0]))
        all_weights['b1'] = tf.Variable(tf.random_normal([self.layerlist[0]], dtype=tf.float32))
        
        for i in range(1,self.layernum):
            all_weights['w'+str(i+1)]=tf.Variable(self.xavier_init(self.layerlist[i-1], self.layerlist[i]))
            all_weights['b'+str(i+1)] = tf.Variable(tf.random_normal([self.layerlist[i]], dtype=tf.float32))

        return all_weights
    
    def _create_layers(self):
        """
        """

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        layer=self.nonlinearity(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.encoded=layer
        for i in range(1,self.layernum-1):
            if i==self.encode_index:
                self.encoded=layer
            layer=self.nonlinearity(tf.add(tf.matmul(layer, self.weights['w'+str(i+1)]), self.weights['b'+str(i+1)]))
            
        self.reconstruction=tf.add(tf.matmul(layer, self.weights['w'+str(self.layernum)]), self.weights['b'+str(self.layernum)])
