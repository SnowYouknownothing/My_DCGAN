# -*- coding: utf-8 -*-
'''
Created on 2017年12月20日
@author: Administrator
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
minst=input_data.read_data_sets('minst_data')
x_real=(minst.train.images-0.5)*2
y_real=minst.train.labels
# print(x_real.shape,y_real.shape)
x_fake=np.random.uniform(-1,1,[55000,100])
x=tf.placeholder(tf.float32,[None,784],name='inputs_real_1')
inputs_real=tf.reshape(x,[-1,28,28,1],name='inputs_real')
inputs_fake=tf.placeholder(tf.float32,[None,100],name='inputs_fake')
with tf.variable_scope('G'):
    G_layer1=tf.layers.dense(inputs_fake,4*4*512,activation=tf.nn.relu)
    G_layer1=tf.reshape(G_layer1,[-1,4,4,512])  
    G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.8)
    G_layer2=tf.layers.conv2d_transpose(G_layer1, filters=256, kernel_size=4, strides=1, padding='valid', activation=tf.nn.relu)
    G_layer2=tf.nn.dropout(G_layer2,keep_prob=0.8)
    G_layer3=tf.layers.conv2d_transpose(G_layer2, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
#     G_layer3=tf.layers.conv2d_transpose(G_layer2, filters=128, kernel_size=3,strides=2, padding='same')
#     G_layer3=tf.layers.batch_normalization(G_layer3,training=True)
#     G_layer3=tf.maximum(0.01*G_layer3,G_layer3)
    G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.8)
    
    
    G_outputs=tf.layers.conv2d_transpose(G_layer3, 1, 3, 2, padding='same', activation=tf.nn.tanh)
#     G_logits=tf.layers.conv2d_transpose(G_layer3,1,3,2,'same')
#     G_outputs=tf.tanh(G_logits)
#A=tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, padding, data_format, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, trainable, name, reuse)

with tf.variable_scope('D'):
# real
    D_layer1=tf.layers.conv2d(inputs_real,128,3,2,padding='same',activation=tf.nn.relu,name='1')
    D_layer1=tf.nn.dropout(D_layer1, keep_prob=0.8)
#     D_layer1=tf.layers.max_pooling2d(D_layer1, 2, 2, padding='same')

    D_layer2=tf.layers.conv2d(D_layer1,256,3,2,padding='same',activation=tf.nn.relu,name='2')
    D_layer2=tf.nn.dropout(D_layer2, keep_prob=0.8)
    
    D_layer3=tf.layers.conv2d(D_layer2,512,3,2,padding='same',activation=tf.nn.relu,name='3')
    D_layer3=tf.nn.dropout(D_layer3, keep_prob=0.8)
    
    flatten=tf.reshape(D_layer3,[-1,4*4*512])
    logits_D=tf.layers.dense(flatten,1,name='4')
    outputs=tf.sigmoid(logits_D)

# fake
    D_layer1_f=tf.layers.conv2d(G_outputs,128,3,2,padding='same',activation=tf.nn.relu,name='1',reuse=True)
    D_layer1_f=tf.nn.dropout(D_layer1_f, keep_prob=0.8)
    
    D_layer2_f=tf.layers.conv2d(D_layer1_f,256,3,2,padding='same',activation=tf.nn.relu,name='2',reuse=True)
    D_layer2_f=tf.nn.dropout(D_layer2_f, keep_prob=0.8)
    
    D_layer3_f=tf.layers.conv2d(D_layer2_f,512,3,2,padding='same',activation=tf.nn.relu,name='3',reuse=True)
    D_layer3_f=tf.nn.dropout(D_layer3_f, keep_prob=0.8)
    
    flatten_f=tf.reshape(D_layer3_f,[-1,4*4*512])
    logits_f=tf.layers.dense(flatten_f,1,name='4',reuse=True)
    outputs_f=tf.sigmoid(logits_f)

# g_output=get_G(inputs_noise,is_train=True)
# d_logits_real,d_outputs_real=get_D(inputs_real)
# d_logits_fake,d_outputs_fake=get_D(g_output,reuse=True)


d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs),
                                                              logits=logits_D))

d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(outputs_f),
                                                              logits=logits_f))

d_loss=tf.add(d_loss_real,d_loss_fake)

g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs_f),
                                                           logits=logits_f))

# train_vars=tf.trainable_variables()
# g_vars=[var for var in train_vars if var.name.startswith('G')]
# d_vars=[var for var in train_vars if var.name.startswith('D')]
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=d_vars)
#     g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=g_vars)
# d_opt=tf.train.AdamOptimizer(0.0001).minimize(d_loss)
# g_opt=tf.train.AdamOptimizer(0.0001).minimize(g_loss)
d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D')) 
g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))
    
list_g_loss,list_d_loss,list_g=[],[],[]
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        for i in range(1):
            o_G_outputs,o_g_loss,o_d_loss=sess.run([g_opt,d_opt,G_outputs,g_loss,d_loss],
                                                   {x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})[2:]
        list_g_loss.append(o_g_loss)
        list_d_loss.append(o_d_loss) 
        list_g.append(o_G_outputs) 
        print(epoch)
    saver.save(sess,'net3/my_net.ckpt')
# b=np.reshape(o_G_outputs[50],[28,28])
# plt.figure()
# plt.imshow(b,cmap='Greys_r')
# plt.show()
# 
# 
# list_g=np.array(list_g)
# for i in range(10):
#     for j in range(3):
#         a=np.reshape(list_g[i,30*j],[28,28])
#         plt.figure()
#         plt.imshow(a,cmap='Greys_r')
#         plt.show() 
# print(o_g_outputs.shape)
# plt.imshow(o_g_outputs[-1])
# plt.show()
plt.figure()
plt.plot(list_d_loss,'red')
plt.plot(list_g_loss,'green')
plt.show()