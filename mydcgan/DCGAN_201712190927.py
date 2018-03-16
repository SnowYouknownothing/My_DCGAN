# -*- coding: utf-8 -*-
'''
Created on 2017年12月19日
@author: Administrator
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
minst=input_data.read_data_sets('minst_data')
x_real=(minst.train.images-0.5)*2
y_real=minst.train.labels
print(x_real.shape,y_real.shape)

# bacth=100
# n_bacth=len(x_real)/100
# x_real=np.array(x_real)

# plt.imshow(x_real[4])
# plt.show()
x_fake=np.random.uniform(-1,1,[55000,100])

# plt.imshow(x_fake[4])
# plt.show()

x=tf.placeholder(tf.float32,[None,784],name='inputs_real_1')
inputs_real=tf.reshape(x,[-1,28,28,1],name='inputs_real')

inputs_fake=tf.placeholder(tf.float32,[None,100],name='inputs_fake')

def get_G(inputs_img,output_dim=1,is_train=True):
    with tf.variable_scope('G',reuse=(not is_train)):
        G_layer1=tf.layers.dense(inputs_img,4*4*512)
        G_layer1=tf.reshape(G_layer1,[-1,4,4,512])
        G_layer1=tf.layers.batch_normalization(G_layer1,training=is_train)
        G_layer1=tf.maximum(0.01*G_layer1,G_layer1)
        G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.8)
        
        G_layer2=tf.layers.conv2d_transpose(G_layer1, filters=256, kernel_size=4,
                                          strides=1, padding='valid')
        G_layer2=tf.layers.batch_normalization(G_layer2,training=is_train)
        G_layer2=tf.maximum(0.01*G_layer2,G_layer2)
        G_layer2=tf.nn.dropout(G_layer2,keep_prob=0.8)
        
        G_layer3=tf.layers.conv2d_transpose(G_layer2, filters=128, kernel_size=3,
                                          strides=2, padding='same')
        G_layer3=tf.layers.batch_normalization(G_layer3,training=is_train)
        G_layer3=tf.maximum(0.01*G_layer3,G_layer3)
        G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.8)
        
        logits=tf.layers.conv2d_transpose(G_layer3,output_dim,3,2,'same')
        G_outputs=tf.tanh(logits)
    return G_outputs

def get_D(inputs_img,reuse=False):  
    with tf.variable_scope('D',reuse=reuse):
        D_layer1=tf.layers.conv2d(inputs_img,128,3,2,padding='same')
        D_layer1=tf.maximum(0.01*D_layer1,D_layer1)
        D_layer1=tf.nn.dropout(D_layer1, keep_prob=0.8)
        
        D_layer2=tf.layers.conv2d(D_layer1,256,3,2,padding='same')
        D_layer2=tf.layers.batch_normalization(D_layer2,training=True)
        D_layer2=tf.maximum(0.01*D_layer2,D_layer2)
        D_layer2=tf.nn.dropout(D_layer2, keep_prob=0.8)
        
        D_layer3=tf.layers.conv2d(D_layer2,512,3,2,padding='same')
        D_layer3=tf.layers.batch_normalization(D_layer3,training=True)
        D_layer3=tf.maximum(0.01*D_layer3,D_layer3)
        D_layer3=tf.nn.dropout(D_layer3, keep_prob=0.8)
        
        flatten=tf.reshape(D_layer3,(-1,4*4*512))
        logits=tf.layers.dense(flatten,1)
        outputs=tf.sigmoid(logits)
    return logits,outputs

def get_loss(inputs_real,inputs_noise,image_depth=1,smooth=0.1):
    g_output=get_G(inputs_noise,image_depth,is_train=True)
    d_logits_real,d_outputs_real=get_D(inputs_real)
    d_logits_fake,d_outputs_fake=get_D(g_output,reuse=True)
    
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_outputs_fake)*(1-smooth),
                                                                  logits=d_logits_fake))
    d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_outputs_real)*(1-smooth),
                                                                  logits=d_logits_real))
    d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_outputs_fake),
                                                                  logits=d_outputs_fake))
    
    d_loss=tf.add(d_loss_real,d_loss_fake)
    
    return g_loss, d_loss

def get_op(g_loss,d_loss,lr=0.001):
    train_vars=tf.trainable_variables()
    g_vars=[var for var in train_vars if var.name.startswith('G')]
    d_vars=[var for var in train_vars if var.name.startswith('D')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_opt=tf.train.AdamOptimizer(lr,0.4).minimize(d_loss,var_list=g_vars)
        g_opt=tf.train.AdamOptimizer(lr,0.4).minimize(g_loss,var_list=d_vars)
    return g_opt,d_opt

list_g_loss,list_d_loss,list_out=[],[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    g_loss,d_loss=get_loss(inputs_real,inputs_fake)
    g_opt,d_opt=get_op(g_loss,d_loss)
#     G_outputs=get_G(inputs_fake)
    for epoch in range(5):
        for i in range(550):
            o_d_loss,o_g_loss=sess.run([d_opt,g_opt],{x:x_real[i*100:(i+1)*100],
                                                     inputs_fake:x_fake[i*100:(i+1)*100]})
        list_g_loss.append(o_g_loss)
        list_d_loss.append(o_d_loss)
#         list_out.append(o_g_outputs)
# print(o_g_outputs.shape)
# 
# 
# plt.imshow(o_g_outputs[-1])
# plt.show()
# for i in range(5):
#     for j in range(3):
#         a=tf.reshape(list_out[i,30*j],[28,28])
#         plt.imshow(a)
#         plt.show()
plt.figure()
plt.plot(list_d_loss,'red')
plt.plot(list_g_loss,'greed')
plt.show()









       
