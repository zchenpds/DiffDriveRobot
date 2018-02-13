# -*- coding: utf-8 -*-
"""
Formation control learning from raw visual observation

Created on Mon Dec 11 10:10:18 2017

@author: jesse
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import os
#import tensorflow.contrib.slim as slim

class DeepFCL:
    
    def __init__(self, obs_dim1, obs_dim2, act_dim, img_chnl):
        
        # observation and state space dimension        
        self.obs_dim = obs_dim1*obs_dim2*img_chnl # height * width * channel
        self.act_dim = act_dim 
        self.batchsize = 32 #256
       
        # input variables
        self.obs_var = tf.placeholder(shape=[None,obs_dim1*obs_dim2*img_chnl], dtype=tf.float32, name="obs_var")
        #self.obs_var = tf.placeholder(tf.float32, shape=(None,obs_dim,obs_dim))
        #self.goal_trj = tf.placeholder(shape=[None,2], dtype=tf.float32, name="goal_trj")
        #self.is_training = tf.placeholder(shape=[],  dtype=tf.bool, name="train_cond")
        
        # ------- Define Observation-State Mapping Using Convolutional Network -----------------------
        
        # network parameters
        conv1_num = 32
        conv2_num = 16
        conv3_num = 16
        fc1_num = 32  
        
        # resize the array of flattened input
        self.imageIn = tf.reshape(self.obs_var, shape=[-1,obs_dim1,obs_dim2,img_chnl])
               
        # convolutions acti: ReLU and spatial softmax
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn,#tf.expand_dims(self.obs_var,3),
                                                     num_outputs=conv1_num,
                                                     kernel_size=[8,8],
                                                     stride=[4,4],
                                                     padding='VALID',
                                                     biases_initializer=None)
        # max pooling
        #self.conv1 = tf.layers.max_pooling2d(self.conv1,2,1)
        
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1,
                                                     num_outputs=conv2_num,
                                                     kernel_size=[4,4],
                                                     stride=[2,2],
                                                     padding='VALID',
                                                     biases_initializer=None)
        
        #self.conv2 = tf.layers.max_pooling2d(self.conv2,2,1)
                                                     
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2,
                                                     num_outputs=conv3_num,
                                                     kernel_size=[3,3],
                                                     stride=[1,1],
                                                     padding='VALID',
                                                     biases_initializer=None)
                                                     
        #self.conv3 = tf.layers.max_pooling2d(self.conv3,2,1)
        
        # output layer
        self.convout = tf.contrib.layers.flatten(self.conv3)                                                    
                                                           
        # fully-connected (acti: ReLU)
        self.W1 = tf.Variable(tf.random_normal([self.convout.get_shape().as_list()[1], fc1_num]))
        self.b1 = tf.Variable(tf.random_normal([fc1_num]))
        self.fc1 = tf.nn.relu(tf.matmul(self.convout, self.W1) + self.b1) 
        
        self.W2 = tf.Variable(tf.random_normal([fc1_num, act_dim]))
        self.b2 = tf.Variable(tf.random_normal([act_dim]))
        # output layer (acti: linear)
        self.out = tf.matmul(self.fc1, self.W2) + self.b2
        #self.out = tf.nn.relu(tf.matmul(self.fc1, self.W2) + self.b2)
                
        # ------- Define Loss Function ---------------------------
        self.targetOut = tf.placeholder(shape=[None,2], dtype=tf.float32)
        self.out_error = tf.norm(self.targetOut - self.out, ord=2, axis=1) 
        
        # total loss
        self.loss = tf.reduce_mean(self.out_error)
        
        # Training Functions
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        
        
    def learn(self, observations, actions, epi_starts):
        
        # Prepre Training Data -------------------------------------------
        # normalize observation input
        self.mean_obs = np.mean(observations, axis=0, keepdims=True)
        self.std_obs = np.std(observations, ddof=1)
        observations = (observations - self.mean_obs) / self.std_obs
        
        # number of samples in total
        num_samples = observations.shape[0] - 1
        
        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not epi_starts[i + 1]], dtype='int32')
        np.random.shuffle(indices)

        # split indices into minibatches
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batchsize]))
                         for start_idx in range(0, num_samples - self.batchsize + 1, self.batchsize)]
        
        # Training -------------------------------------------------------
        init = tf.global_variables_initializer()
        num_epochs = 1000
        saver = tf.train.Saver()
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir + '/models/')
        
        with tf.Session() as sess:
            sess.run(init)
            loss_hist = []            
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                epoch_batches = 0                
                enumerated_minibatches = list(enumerate(minibatchlist))
                np.random.shuffle(enumerated_minibatches)
                
                for i, batch in enumerated_minibatches:                    
                    _ , tmp_loss = sess.run([self.train_op,self.loss], feed_dict = {
                                                                        self.obs_var: observations[batch],                                                                        
                                                                        self.targetOut: actions[batch] })                                                                      
                    epoch_loss += tmp_loss
                    epoch_batches += 1                    
                    loss_hist.append(epoch_loss / epoch_batches)
                    
                # print results for this epoch
                if (epoch+1) % 5 ==0:
                    print("Epoch {:3}/{}, loss:{:.4f}".format(epoch+1, num_epochs, epoch_loss / epoch_batches))                    
                
            # save the updated model
            print('saving learned model')
            saver.save(sess, os.path.join(save_path, 'model_epi' + str(epoch)))
            predicted_action = sess.run(self.out, feed_dict={self.obs_var: observations})
        plt.close("Learned Policy")
        
        return predicted_action, loss_hist
        
        
    def test(self, observations):
        observations = (observations - self.mean_obs) / self.std_obs
        saver = tf.train.Saver()
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir + '/models/')
        num_epochs = 1000
        with tf.Session() as sess:
            # load the model and output action
            saver.restore(sess, os.path.join(save_path, 'model_epi' + str(num_epochs-1)))
            act_output = sess.run(self.out, feed_dict = {self.obs_var: observations})
            
        return act_output
        
        
if __name__ == '__main__':
    
    print('\nFormation Control Task\n')

    print('Loading and displaying training data ... ')
    training_data = np.load('fcl_data11.npz')
    #plot_observations(training_data['observations'], name="Observation Samples (Subset of Training Data) -- Simple Navigation Task")

    print('Learning a policy ... ')
    fcl = DeepFCL(50, 50, 2, 1)
    [training_ctrls, loss_hist] = fcl.learn(training_data['observations'],training_data['actions'],training_data['epi_starts'])
#    plot_representation(training_states, training_data['rewards'],
#                            name='Observation-State-Mapping Applied to Training Data -- Simple Navigation Task',
#                            add_colorbar=True)
    
    print('Loading and displaying testing data ... ')
    testing_data = np.load('fcl_data1.npz')
    
    print('Testing a policy ... ')    
    testing_ctrls = fcl.test(testing_data['observations'])
    
        
        
        
        
        
