# -*- coding: utf-8 -*-
'''
Created on 2018年1月2日
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
# lr=tf.Variable(0.001,tf.float32)



with tf.variable_scope('G'):
    G_layer1=tf.layers.dense(inputs_fake,4*4*512)
    G_layer1=tf.reshape(G_layer1,[-1,4,4,512])  
    G_layer1=tf.layers.batch_normalization(G_layer1)
    G_layer1=tf.maximum(0.01*G_layer1,G_layer1)
    G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.5)
    
#     G_layer2=tf.layers.conv2d_transpose(G_layer1, filters=1024, kernel_size=2, strides=2, padding='same') 
#     G_layer2=tf.layers.batch_normalization(G_layer2)
#     G_layer2=tf.maximum(0.01*G_layer2,G_layer2)
#     G_layer2=tf.nn.dropout(G_layer2,keep_prob=0.5) 
    
    G_layer3=tf.layers.conv2d_transpose(G_layer1, filters=512, kernel_size=4, strides=1, padding='valid')
#     G_layer3=tf.layers.batch_normalization(G_layer3)
    G_layer3=tf.maximum(0.01*G_layer3,G_layer3)
    G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.5)
    
    G_layer4=tf.layers.conv2d_transpose(G_layer3, filters=256, kernel_size=3, strides=2, padding='same')
    G_layer4=tf.layers.batch_normalization(G_layer4)
    G_layer4=tf.maximum(0.01*G_layer4,G_layer4)
    G_layer4=tf.nn.dropout(G_layer4,keep_prob=0.5)
    
    G_outputs=tf.layers.conv2d_transpose(G_layer4, 1, 3, 2, padding='same', activation=tf.nn.tanh)
#     G_logits=tf.layers.conv2d_transpose(G_layer3,1,3,2,'same')
#     G_outputs=tf.tanh(G_logits)
#A=tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, padding, data_format, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, trainable, name, reuse)

with tf.variable_scope('D'):
# real
#     c1=tf.layers.conv2d(inputs_real, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,name='1')
#     p1=tf.layers.max_pooling2d(c1, pool_size=2, strides=2, padding='same')
#     c2=tf.layers.conv2d(p1, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,name='2')
#     p2=tf.layers.average_pooling2d(c2, pool_size=2, strides=2, padding='same')
#     c3=tf.layers.conv2d(p2, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,name='3')
#     p3=tf.layers.max_pooling2d(c1, pool_size=2, strides=2, padding='same')
#     c4=tf.layers.conv2d(p3, filters=512, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu,name='4')
#     p4=tf.layers.max_pooling2d(c4, pool_size=2, strides=2, padding='same')    
# 
#     full_conect=tf.reshape(p4, [-1,2*2*512])
#     f1=tf.layers.dense(full_conect,2048,tf.nn.relu,name='5')
#     f2=tf.nn.dropout(f1,keep_prob=0.5)
#     
#     f3=tf.layers.dense(f2,1024,tf.nn.relu,name='6')
#     f4=tf.nn.dropout(f3,keep_prob=0.5)  
    conv1=tf.layers.conv2d(inputs_real, filters=64, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='1')
    pool1=tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')
    layer1=tf.nn.dropout(pool1, keep_prob=0.5)
    
    conv2=tf.layers.conv2d(layer1, filters=128, kernel_size=3, strides=1, padding='same',name='2')
    pool2=tf.layers.batch_normalization(conv2)
    pool2=tf.maximum(0.01*pool2,pool2)
    pool2=tf.layers.max_pooling2d(pool2, pool_size=2, strides=2, padding='same')
    layer2=tf.nn.dropout(pool2, keep_prob=0.5)

    conv3=tf.layers.conv2d(layer2, filters=256, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='3')
    pool3=tf.layers.batch_normalization(conv3)
    pool3=tf.maximum(0.01*pool3,pool3)
    pool3=tf.layers.max_pooling2d(pool3, pool_size=2, strides=2, padding='valid')
    layer3=tf.nn.dropout(pool3, keep_prob=0.5)   

    conv4=tf.layers.conv2d(layer3, filters=512, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='4')
    pool4=tf.layers.batch_normalization(conv4)
    pool4=tf.maximum(0.01*pool4,pool4)    
    pool4=tf.layers.max_pooling2d(pool4, pool_size=2, strides=2, padding='same')
    layer4=tf.nn.dropout(pool4, keep_prob=0.5)
    
    full_conect=tf.reshape(layer4,[-1,2*2*512])
    output1=tf.layers.dense(full_conect, 1024, activation=tf.nn.relu,name='5')
    output2=tf.nn.dropout(output1,0.5)
 
#     output3=tf.layers.dense(output2, 1024, activation=tf.nn.relu,name='6')
#     output4=tf.nn.dropout(output3,0.5)

    logits_D=tf.layers.dense(output2,1,name='6')
    outputs=tf.nn.sigmoid(logits_D)

# fake
#     fc1=tf.layers.conv2d(G_outputs, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,name='1',reuse=True)
#     fp1=tf.layers.max_pooling2d(fc1, pool_size=2, strides=2, padding='same')
#     fc2=tf.layers.conv2d(fp1, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,name='2',reuse=True)
#     fp2=tf.layers.average_pooling2d(fc2, pool_size=2, strides=2, padding='same')
#     fc3=tf.layers.conv2d(fp2, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,name='3',reuse=True)
#     fp3=tf.layers.max_pooling2d(fc1, pool_size=2, strides=2, padding='same')
#     fc4=tf.layers.conv2d(fp3, filters=512, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu,name='4',reuse=True)
#     fp4=tf.layers.max_pooling2d(fc4, pool_size=2, strides=2, padding='same')    
# 
#     ffull_conect=tf.reshape(fp4, [-1,2*2*512])
#     ff1=tf.layers.dense(ffull_conect,2048,tf.nn.relu,name='5',reuse=True)
#     ff2=tf.nn.dropout(ff1,keep_prob=0.5)
#     
#     ff3=tf.layers.dense(ff2,1024,tf.nn.relu,name='6',reuse=True)
#     ff4=tf.nn.dropout(ff3,keep_prob=0.5)  
    conv1_f=tf.layers.conv2d(G_outputs, filters=64, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='1',reuse=True)
    pool1_f=tf.layers.max_pooling2d(conv1_f, pool_size=2, strides=2, padding='same')
    layer1_f=tf.nn.dropout(pool1_f, keep_prob=0.5)
    
    conv2_f=tf.layers.conv2d(layer1_f, filters=128, kernel_size=3, strides=1, padding='same',name='2',reuse=True)
    pool2=tf.layers.batch_normalization(conv2_f)
    pool2=tf.maximum(0.01*pool2,pool2)   
    pool2_f=tf.layers.max_pooling2d(pool2, pool_size=2, strides=2, padding='same')
    layer2_f=tf.nn.dropout(pool2_f, keep_prob=0.5)

    conv3_f=tf.layers.conv2d(layer2_f, filters=256, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='3',reuse=True)
    pool3=tf.layers.batch_normalization(conv3_f)
    pool3=tf.maximum(0.01*pool3,pool3)     
    pool3_f=tf.layers.max_pooling2d(pool3, pool_size=2, strides=2, padding='valid')
    layer3_f=tf.nn.dropout(pool3_f, keep_prob=0.5)   

    conv4_f=tf.layers.conv2d(layer3_f, filters=512, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu,name='4',reuse=True)
    pool4=tf.layers.batch_normalization(conv4_f)
    pool4=tf.maximum(0.01*pool4,pool4)     
    pool4_f=tf.layers.max_pooling2d(pool4, pool_size=2, strides=2, padding='same')
    layer4_f=tf.nn.dropout(pool4_f, keep_prob=0.5)
    
    full_conect_f=tf.reshape(layer4_f,[-1,2*2*512])
    output1_f=tf.layers.dense(full_conect_f, 1024, activation=tf.nn.relu,name='5',reuse=True)
    output2_f=tf.nn.dropout(output1_f,0.5)

#     output3_f=tf.layers.dense(output2_f, 1024, activation=tf.nn.relu,name='6',reuse=True)
#     output4_f=tf.nn.dropout(output3_f,0.5) 

    logits_f=tf.layers.dense(output2_f,1,name='6',reuse=True)
    outputs_f=tf.nn.sigmoid(logits_f)

d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs)*0.95,
                                                              logits=logits_D))

d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(outputs_f),
                                                              logits=logits_f))

d_loss=tf.add(d_loss_real,d_loss_fake)

g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(outputs_f)*0.95,
                                                           logits=logits_f))

# train_vars=tf.trainable_variables()
# g_vars=[var for var in train_vars if var.name.startswith('G')]
# d_vars=[var for var in train_vars if var.name.startswith('D')]
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=d_vars)
#     g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=g_vars)

d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D')) 
g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))


list_g_loss,list_d_loss,list_g=[],[],[]
j=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
#         sess.run(tf.assign(lr,0.001*(0.5**epoch)))
        for i in range(550):
            o_G_outputs,o_g_loss,o_d_loss=sess.run([g_opt,d_opt,G_outputs,g_loss,d_loss],
                                                   {x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})[2:]
            list_g_loss.append(o_g_loss)
            list_d_loss.append(o_d_loss)
            j+=1 
            if j%55==0:
                list_g.append(o_G_outputs) 
#             G_layer1,G_layer2,G_layer3,G_layer4,G_outputs=sess.run([G_layer1,G_layer2,G_layer3,G_layer4,G_outputs],{x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})
#             
            print(epoch,i)
#     saver.save(sess,'net9/my_net.ckpt')
# b=np.reshape(o_G_outputs[50],[28,28])
# plt.figure()
# plt.imshow(b,cmap='Greys_r')
# plt.show()
# print(G_layer1.shape,G_layer2.shape,G_layer3.shape,G_layer4.shape,G_outputs.shape)
# print(np.array(o_G_outputs).shape)
# 显示结果
plt.figure()
plt.plot(list_d_loss,'red')
plt.plot(list_g_loss,'green')
plt.show()


print(np.array(o_G_outputs).shape)


list_g=np.array(list_g)
index_list=np.random.randint(1,100,20)
list_show=[]
for m in range(10):
    for n in index_list:
        list_show.append(list_g[m,n])

list_show=(np.array(list_show)/2)+0.5

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=np.reshape(list_show,[200,28,28])
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

list_last_show=[]
for m in [-2,-1]:
    for n in range(100):
        list_last_show.append(list_g[m,n])

list_last_show=(np.array(list_last_show)/2)+0.5

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=np.reshape(list_last_show,[200,28,28])
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()