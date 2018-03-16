# -*- coding: utf-8 -*-
'''
Created on 2017年12月21日
@author: Administrator
'''
from numpy import array, reshape
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from matplotlib.pyplot import autumn

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
x_train,y_train=[],[]
for i in range(1,6):
    datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
    for j in range(len(datadict[b'data'])):
        x_train.append(datadict[b'data'][j])
        y_train.append(datadict[b'labels'][j])
# print(len(x_test),len(x_test[0]),len(y_test))

datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
x_test,y_test=datadict[b'data'],datadict[b'labels']
print(len(x_test),len(x_test[0]),len(y_test))
print(len(x_train),len(y_train))
# 
# x_train=np.reshape(x_train, [50000,3,32,32] )
# x_train=np.transpose(x_train, [0,2,3,1])

# plt.imshow(x_train[4])
# plt.show() 
# print(y_train[5])

x_list=[]
for i in range(len(y_train)):
    if y_train[i]==1:
        x_list.append(x_train[i])
x_real=(np.array(x_list)-127.5)/127.5



# from sklearn.preprocessing import MinMaxScaler
# minmax=MinMaxScaler()
# 
# x_real=minmax.fit_transform(x_real)

x_fake=np.random.uniform(-1,1,[5000,100])

x=tf.placeholder(tf.float32,[None,3072],name='inputs_real_1')
inputs_real=tf.reshape(x,[-1,32,32,3],name='inputs_real')
inputs_fake=tf.placeholder(tf.float32,[None,100],name='inputs_fake')

with tf.variable_scope('G'):
    G_layer1=tf.layers.dense(inputs_fake,4*4*512,activation=tf.nn.relu)
    G_layer1=tf.reshape(G_layer1,[-1,4,4,512])  
    G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.5)
    
    G_layer2=tf.layers.conv2d_transpose(G_layer1, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    G_layer2=tf.nn.dropout(G_layer2,keep_prob=0.5)
    
    G_layer3=tf.layers.conv2d_transpose(G_layer2, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.5)
    
    G_outputs=tf.layers.conv2d_transpose(G_layer3, 3, 3, 2, padding='same', activation=tf.nn.tanh)


with tf.variable_scope('D'):
# real
    D_layer1=tf.layers.conv2d(inputs_real,128,3,2,padding='same',activation=tf.nn.relu,name='1')
    D_layer1=tf.nn.dropout(D_layer1, keep_prob=0.5)
  
    D_layer2=tf.layers.conv2d(D_layer1,256,3,2,padding='same',activation=tf.nn.relu,name='2')
    D_layer2=tf.nn.dropout(D_layer2, keep_prob=0.5)
      
    D_layer3=tf.layers.conv2d(D_layer2,512,3,2,padding='same',activation=tf.nn.relu,name='3')
    D_layer3=tf.nn.dropout(D_layer3, keep_prob=0.5)
      
    flatten=tf.reshape(D_layer3,[-1,4*4*512])
    logits_D=tf.layers.dense(flatten,1,name='4')
    outputs=tf.sigmoid(logits_D)

# fake
    D_layer1_f=tf.layers.conv2d(G_outputs,128,3,2,padding='same',activation=tf.nn.relu,name='1',reuse=True)
    D_layer1_f=tf.nn.dropout(D_layer1_f, keep_prob=0.5)
      
    D_layer2_f=tf.layers.conv2d(D_layer1_f,256,3,2,padding='same',activation=tf.nn.relu,name='2',reuse=True)
    D_layer2_f=tf.nn.dropout(D_layer2_f, keep_prob=0.5)
      
    D_layer3_f=tf.layers.conv2d(D_layer2_f,512,3,2,padding='same',activation=tf.nn.relu,name='3',reuse=True)
    D_layer3_f=tf.nn.dropout(D_layer3_f, keep_prob=0.5)
      
    flatten_f=tf.reshape(D_layer3_f,[-1,4*4*512])
    logits_f=tf.layers.dense(flatten_f,1,name='4',reuse=True)
    outputs_f=tf.sigmoid(logits_f)
  
# g_output=get_G(inputs_noise,is_train=True)
# d_logits_real,d_outputs_real=get_D(inputs_real)
# d_logits_fake,d_outputs_fake=get_D(g_output,reuse=True)
  
  
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs)*0.9,
                                                              logits=logits_D))
  
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(outputs_f),
                                                              logits=logits_f))
  
d_loss=tf.add(d_loss_real,d_loss_fake)
  
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs_f)*0.9,
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
        for i in range(50):
            o_G_outputs,o_g_loss,o_d_loss=sess.run([g_opt,d_opt,G_outputs,g_loss,d_loss],
                                                   {x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})[2:]
            list_g_loss.append(o_g_loss)
            list_d_loss.append(o_d_loss) 
            list_g.append(o_G_outputs) 
        print(epoch)       
    saver.save(sess,'net10_cifar/my_net.ckpt')
# b=np.reshape(o_G_outputs[50],[28,28])
# plt.figure()
# plt.imshow(b,cmap='Greys_r')
# plt.show()

plt.figure()
plt.plot(list_d_loss,'red')
plt.plot(list_g_loss,'green')
plt.show()


def transpose_x(a):
    a=(np.array(a)*127.5)+127.5
    a=np.reshape(a,[len(a),3072])
    a=np.reshape(a,[len(a),3,32,32] )
    a=np.transpose(a, [0,2,3,1])
    return a

def figure(l):
    for i in [0,len(l)*0.25,len(l)*0.5,len(l)*0.75,-1]:
        plt.imshow(l[i])
        plt.show()

x_real=transpose_x(x_real)
list_g=transpose_x(list_g)

figure(list_g)
figure(x_real)


# fig,axes=plt.subplots(nrows=5, ncols=20, sharex=True, sharey=True, figsize=(20,5))
# imgs=list_g[4899:4999]
# for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100]],axes):
#     for img,ax in zip(image,row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.show()
# print(o_g_outputs.shape)
# plt.imshow(o_g_outputs[-1])
# plt.show()






















