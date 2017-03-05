import scipy
import tensorflow as tf

class Autoencoder_ffnn():
    def __init__(self,experiment,tf_session, inputdim,layerlist,encode_index,optimizer = tf.train.AdamOptimizer(),nonlinear=tf.nn.relu):
        """
        """
        self.experiment=experiment
        
        self.layerlist=layerlist
        self.layernum=len(layerlist)
        self.n_input = inputdim
        self.encode_index=encode_index
        self.display_step=10

        network_weights = self._initialize_weights()
        self.weights = network_weights  

        self._create_layers(nonlinear)

        # cost
        self.cost =  0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf_session
        self.sess.run(init)
        
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
        layer=nonlinearity(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))

        for i in range(1,self.layernum-1):
            if i==self.encode_index:
                self.encoded=layer
            layer=nonlinearity(tf.add(tf.matmul(layer, self.weights['w'+str(i+1)]), self.weights['b'+str(i+1)]))
            
        self.reconstruction=tf.add(tf.matmul(layer, self.weights['w'+str(self.layernum)]), self.weights['b'+str(self.layernum)])

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def encode(self, X):
        return self.sess.run(self.encoded, feed_dict={self.x: X})

    def decode(self, encoded = None):
        if encoded is None:
            encoded = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.encoded: encoded})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})
    
    def train(self,X_train,X_test,batch_size,max_epochs):
        breaker=False
        testlog=collections.deque(maxlen=30)
        
        for epoch in range(max_epochs):
            avg_cost = 0.
            total_batch = int(len(X_train) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(X_train, batch_size)
                cost = self.partial_fit(batch_xs)
                avg_cost += cost/ batch_size
                
                #early stop
                testlog.append(self.calc_total_cost(X_test))
                for i in range(8):
                    if len(testlog)>20 and testlog[-i]>=testlog[-10-i]*0.995:
                        breaker=True
                    else:
                        breaker=False
                if breaker:
                    print("STOPPED OVERFIT")
                    break
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            if breaker:
                break
                
    def get_random_block_from_data(self,data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]
    
    def xavier_init(self,fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)
