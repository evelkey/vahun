from random import randint, random
import collections
from autoencoders import Autoencoder_ffnn
import tensorflow as tf
from tools import Timer

class experiment:
    
    def __init__(self,out_dim,minw,maxw,encoded_width,layermin=1,layermax=5):
        self.len=randint(layermin,layermax)*2
        self.weights=[randint(minw,maxw) for n in range(self.len)]
        self.weights[int(self.len/2-1)]=encoded_width
        self.weights[-1]=out_dim
        
    def set(self,weights):
        self.len=len(weights)
        self.weights=weights
        
class evolution:
    
    def __init__(self,x_train,x_test,
                 population_size,encoder,
                 dim,config,logger,repeat_runs=3,
                 epoch=30,batch=512,disp_freq=1):
        """
        """
        self.timer=Timer()
        self.logger=logger
        
        self.encoded_width=encoder
        self.dim=dim
        self.min=10
        self.max=200
        self.repeat_runs=repeat_runs
        
        self.config=config
        self.training_epochs = epoch
        self.batch_size = batch
        self.display_step = disp_freq
        self.x_train=x_train
        self.x_test=x_test
        
        self.learnrate=0.001
        self.batchsize=512
        self.maxepoch=100
        self.optimizer=tf.train.AdamOptimizer(learning_rate = self.learnrate)
        
        
        self.retain_p=0.2
        self.random_select_p=0.05
        self.mutate_p=0.1
        self.mutate_len_p=0.1
        self.mutate_width_p=0.4
        self.population_size=population_size
        self.population=self.gen_population(population_size)
        
        self.target=0
        

        
    def ekv(self,e):
        return e
       
    
    def gen_population(self,count):
        """
        count: the number of individuals in the population
        """
        self.sess = tf.Session(config=self.config)
        self.timer.add("gen")
        print("Initializing new population")
        population=[]
        self.logger.logline("population.log",["len","weigths...->"])
        for x in range(count):
            exp=experiment(out_dim=self.dim,minw=self.min,
                           maxw=self.max,
                           encoded_width=self.encoded_width)
            population.append(Autoencoder_ffnn(experiment=exp,
                                               tf_session=self.sess,inputdim=self.dim,
                                               layerlist=exp.weights,
                                               encode_index=int(exp.len/2-1),
                                               optimizer = self.optimizer))
            self.logger.logline("population.log",[population[x].experiment.len]+\
                                population[x].experiment.weights)
        print("new gen took: ",self.timer.get("gen")," s")
        
        return population
    
    def new_generation(self,experiments):
        """
        
        """
        self.timer.add("new")
        self.sess.close()
        self.sess = tf.Session(config=self.config)
        
        print("New generation is being created.")
        
        population=[]
        for x in range(len(experiments)):
    
            population.append(Autoencoder_ffnn(experiment=experiments[x],tf_session=self.sess,inputdim=self.dim,layerlist=experiments[x].weights,
                                               encode_index=int(experiments[x].len/2-1),
                                               optimizer = self.optimizer))
        print("new gen took: ",self.timer.get("new")," s")
        return population
    
    def train_population(self):
        self.population_fitness=[]
        for individual in self.population:
            sum_cost=0
            for i in range(self.repeat_runs): #average the model's fitness
                individual.train(self.x_train,self.x_test,self.batchsize,self.maxepoch)
                sum_cost+=individual.calc_total_cost(self.x_test)
            self.population_fitness.append(sum_cost/self.repeat_runs)
        return self.population_fitness
    

    def grade(self):
        'Find average fitness for a population.'
        summed = sum(self.population_fitness)
        self.graded= summed / (self.population_size * 1.0)
        return self.graded
    
    def mutate(self,group):
        for individual in group:
            if self.mutate_p > random():
                if self.mutate_len_p>random():
                    if random()<0.5:
                        individual.len+=2
                        individual.weights=[randint(self.min, self.max),randint(self.min, self.max)]+individual.weights
                        individual.weights[int(individual.len/2-1)]=self.encoded_width
                    else :
                        if individual.len!=2:
                            individual.len-=2
                            individual.weights=individual.weights[2:]
                            individual.weights[int(individual.len/2-1)]=self.encoded_width
                if self.mutate_width_p>random():
                    pos_to_mutate = randint(0,individual.len-2)
                    if pos_to_mutate!=int(individual.len/2-1):
                        if 0.5>random():
                            individual.weights[pos_to_mutate] +=20
                        else:
                            individual.weights[pos_to_mutate] -=20
        self.mutants=group
        return group

    def evolve(self):
        self.train_population()
        
        #select top individs
        graded = [(self.population_fitness[x], self.population[x].experiment) for x in range(self.population_size)]
        graded = [ x[1] for x in sorted(graded)]
        retain_length = int(len(graded)*self.retain_p)
        parents = graded[:retain_length]
        
        
        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if self.random_select_p > random():
                parents.append(individual)
        
        # mutate 
        mutants=self.mutate(parents)
       
        # crossover parents to create children (aka sex)
        mutants_length = len(mutants)
        desired_length = self.population_size - mutants_length
        children = []
        while len(children) < desired_length:
            male = randint(0, mutants_length-1)
            female = randint(0, mutants_length-1)
            if male != female:
                male = mutants[male]
                female = mutants[female]
                
                child=experiment(out_dim=self.dim,minw=self.min,maxw=self.max,encoded_width=self.encoded_width)
                weights = male.weights[:int(male.len/2-1)]+female.weights[int(female.len/2-1):]
                child.set(weights)
                children.append(child)
                
        mutants.extend(children)
        
        self.population=self.new_generation(mutants)
        return mutants