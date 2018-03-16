# -*- coding: utf-8 -*-
'''
Created on 2017年12月26日
@author: Administrator
'''

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

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

x_list=[]
for i in range(len(y_train)):
    if y_train[i]==1:
        x_list.append(x_train[i])

# x_real=np.transpose(np.reshape(np.array(x_list),[5000,3,32,32]),[0,2,3,1])
# fig,axes=plt.subplots(nrows=5, ncols=20, sharex=True, sharey=True, figsize=(20,5))
# imgs=x_real[-120:-20]
# for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100]],axes):
#     for img,ax in zip(image,row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.show()
#                 





x_real=((np.array(x_list))/127.5)-1
x_fake=np.random.uniform(-1,1,[5000,100])

x=tf.placeholder(tf.float32,[None,3072],name='inputs_real_1')
inputs_real=tf.reshape(x,[-1,32,32,3],name='inputs_real')
inputs_fake=tf.placeholder(tf.float32,[None,100],name='inputs_fake')

with tf.variable_scope('G'):
    G_layer1=tf.layers.dense(inputs_fake,4*4*1024)
    G_layer1=tf.reshape(G_layer1,[-1,4,4,1024])
    G_layer1=tf.layers.batch_normalization(G_layer1)
    G_layer1=tf.nn.relu(G_layer1)
    G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.8)
    
    G_layer2=tf.layers.conv2d_transpose(G_layer1, filters=512, kernel_size=3, strides=2, padding='same')
    G_layer2=tf.layers.batch_normalization(G_layer2)
    G_layer2=tf.nn.relu(G_layer2)
    G_layer2=tf.nn.dropout(G_layer2,keep_prob=0.8) 
    
    G_layer3=tf.layers.conv2d_transpose(G_layer2, filters=256, kernel_size=3, strides=2, padding='same',)
    G_layer3=tf.layers.batch_normalization(G_layer3)
    G_layer3=tf.nn.relu(G_layer3)
    G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.8)
      
    G_outputs=tf.layers.conv2d_transpose(G_layer3, 3, 3, 2, padding='same', activation=tf.tanh)

with tf.variable_scope('D'):

    conv1=tf.layers.conv2d(inputs_real, filters=2*64, kernel_size=3, strides=2, padding='same',name='1')
    layer1=tf.maximum(0.01*conv1,conv1)
    layer1=tf.nn.dropout(layer1, keep_prob=0.8)
        
    conv2=tf.layers.conv2d(layer1, filters=2*128, kernel_size=3, strides=2, padding='same',name='2')
    layer2=tf.layers.batch_normalization(conv2)
    layer2=tf.maximum(0.01*layer2,layer2)
    layer2=tf.nn.dropout(layer2, keep_prob=0.8)

    conv3=tf.layers.conv2d(layer2, filters=2*256, kernel_size=3, strides=2, padding='same',name='3')
    layer3=tf.layers.batch_normalization(conv3)
    layer3=tf.maximum(0.01*layer3,layer3)
    layer3=tf.nn.dropout(layer3, keep_prob=0.8)   

    conv4=tf.layers.conv2d(layer3, filters=2*512, kernel_size=3, strides=1, padding='same',name='4')
    layer4=tf.layers.batch_normalization(conv4)
    layer4=tf.maximum(0.01*layer4,layer4)
    layer4=tf.nn.dropout(layer4, keep_prob=0.8)
    
    full_conect=tf.reshape(layer4,[-1,2*2*2*512])
    logits_D=tf.layers.dense(full_conect,1,name='5')
    outputs=tf.nn.sigmoid(logits_D)

# fake
    conv1_f=tf.layers.conv2d(G_outputs, filters=2*64, kernel_size=3, strides=2, padding='same',name='1',reuse=True)
    pool1_f=tf.maximum(0.01*conv1_f,conv1_f)
    layer1_f=tf.nn.dropout(pool1_f, keep_prob=0.8)
    
    conv2_f=tf.layers.conv2d(layer1_f, filters=2*128, kernel_size=3, strides=2, padding='same',name='2',reuse=True)
    pool2_f=tf.layers.batch_normalization(conv2_f)
    pool2_f=tf.maximum(0.01*pool2_f,pool2_f)
    layer2_f=tf.nn.dropout(pool2_f, keep_prob=0.8)

    conv3_f=tf.layers.conv2d(layer2_f, filters=2*256, kernel_size=3, strides=2, padding='same',name='3',reuse=True)
    pool3_f=tf.layers.batch_normalization(conv3_f)
    pool3_f=tf.maximum(0.01*pool3_f,pool3_f)
    layer3_f=tf.nn.dropout(pool3_f, keep_prob=0.8)   

    conv4_f=tf.layers.conv2d(layer3_f, filters=2*512, kernel_size=3, strides=2, padding='same',name='4',reuse=True)
    pool4_f=tf.layers.batch_normalization(conv4_f)
    pool4_f=tf.maximum(0.01*pool4_f,pool4_f)
    layer4_f=tf.nn.dropout(pool4_f, keep_prob=0.8)
    
    full_conect_f=tf.reshape(layer4_f,[-1,2*2*2*512])  
    logits_f=tf.layers.dense(full_conect_f,1,name='5',reuse=True)
    outputs_f=tf.nn.sigmoid(logits_f)
  
  
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs),
                                                              logits=logits_D))
  
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(outputs_f),
                                                              logits=logits_f))
  
d_loss=tf.add(d_loss_real,d_loss_fake)
  
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs_f),
                                                           logits=logits_f))
#   
# train_vars=tf.trainable_variables()
# g_vars=[var for var in train_vars if var.name.startswith('G')]
# d_vars=[var for var in train_vars if var.name.startswith('D')]
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=d_vars)
#     g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=g_vars)

d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D')) 
g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))
      
list_g_loss,list_d_loss,list_g=[],[],[]
# saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        for i in range(50):
            o_G_outputs,o_g_loss,o_d_loss=sess.run([g_opt,d_opt,G_outputs,g_loss,d_loss],
                                                   {x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})[2:]
            list_g_loss.append(o_g_loss)
            list_d_loss.append(o_d_loss) 
            if epoch==0:
                if (i+1) % 5 == 0:
                    list_g.append(o_G_outputs) 
            print(i)
        print(epoch)       
#     saver.save(sess,'net10_cifar/my_net.ckpt')
# b=np.reshape(o_G_outputs[50],[28,28])
# plt.figure()
# plt.imshow(b,cmap='Greys_r')
# plt.show()


plt.figure()
plt.plot(list_d_loss,'red')
plt.plot(list_g_loss,'green')
plt.show()

# list_x=np.reshape((np.array(list_g)/2)+0.5,[1000,3072])
# for i in range(1000):
#     print(max(list_x[i]),'          ',min(list_x[i]))
    
list_g=np.reshape(np.array(list_g),[10,100,3072])
print(list_g.shape)
list_g=(np.array(list_g)/2)+0.5
index_error=0
for i in range(10):
    for j in range(100):
        for m in range(3072):
            if list_g[i,j,m]<0:
                list_g[i,j,m]=0
                index_error+=1

print(index_error)
list_g=np.array(list_g)

index_list=np.random.randint(1,100,20)
list_show=[]
for m in range(10):
    for n in index_list:
        list_show.append(list_g[m,n])
        
list_show=np.array(list_show)

list_show=np.transpose(np.reshape(list_show,[200,3,32,32]),[0,2,3,1])

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=list_show
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()



# def figure(l):
#     for i in [0,len(l)*0.25,len(l)*0.5,len(l)*0.75,-1]:
#         plt.imshow(l[i])
#         plt.show()


# 
# figure(list_g)
# figure(x_real)
