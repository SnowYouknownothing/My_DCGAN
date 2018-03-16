# -*- coding: utf-8 -*-
'''
Created on 2018年1月9日
@author: Administrator
'''
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import os
import tensorflow as tf

images=np.load('faces_kt1.npy')
print('load data successful!',images.shape)


inputs_real=tf.placeholder(tf.float32,[None,96,96,3],name='inputs_real')
inputs_fake=tf.placeholder(tf.float32,[None,100],name='inputs_fake')

def Gen(inputs):
    with tf.variable_scope('G'):
        filters_g=64
        G_layer1=tf.layers.dense(inputs,3*3*filters_g*16)
        G_layer1=tf.reshape(G_layer1,[-1,3,3,filters_g*16])  
        G_layer1=tf.layers.batch_normalization(G_layer1)
        G_layer1=tf.nn.relu(G_layer1)
        G_layer1=tf.nn.dropout(G_layer1,keep_prob=0.8)
        
        G_layer3=tf.layers.conv2d_transpose(G_layer1, filters_g*8, kernel_size=2, strides=2, padding='same')
        G_layer3=tf.layers.batch_normalization(G_layer3)
        G_layer3=tf.nn.relu(G_layer3)
        G_layer3=tf.nn.dropout(G_layer3,keep_prob=0.8)
        
        G_layer4=tf.layers.conv2d_transpose(G_layer3, filters_g*4, kernel_size=3, strides=2, padding='same')
        G_layer4=tf.layers.batch_normalization(G_layer4)
        G_layer4=tf.nn.relu(G_layer4)
        G_layer4=tf.nn.dropout(G_layer4,keep_prob=0.8)
        
        G_layer5=tf.layers.conv2d_transpose(G_layer4, filters_g*2, kernel_size=3, strides=2, padding='same')
        G_layer5=tf.layers.batch_normalization(G_layer5)
        G_layer5=tf.nn.relu(G_layer5)
        G_layer5=tf.nn.dropout(G_layer5,keep_prob=0.8)

        G_layer6=tf.layers.conv2d_transpose(G_layer5, filters_g, kernel_size=3, strides=2, padding='same')
        G_layer6=tf.layers.batch_normalization(G_layer6)
        G_layer6=tf.nn.relu(G_layer6)
        G_layer6=tf.nn.dropout(G_layer6,keep_prob=0.8)
        
        G_outputs_1=tf.layers.conv2d_transpose(G_layer6, 3, 3, 2, padding='same')
        G_outputs=tf.tanh(G_outputs_1)
    return G_outputs

def Dis(inputs,reuse_=False):
    with tf.variable_scope('D'):
        filters_d=64
        layer_1=tf.layers.conv2d(inputs, filters_d, 3, 2, 'same',name='1',reuse=reuse_)
        layer_1=tf.maximum(0.01*layer_1,layer_1)
        layer_1=tf.nn.dropout(layer_1,0.8)
    
        layer_2=tf.layers.conv2d(layer_1, filters_d*2, 3, 2, 'same',name='2',reuse=reuse_)
        layer_2=tf.layers.batch_normalization(layer_2,name='7',reuse=reuse_)
        layer_2=tf.maximum(0.01*layer_2,layer_2)
        layer_2=tf.nn.dropout(layer_2,0.8)
    
        layer_3=tf.layers.conv2d(layer_2, filters_d*4, 3, 2, 'same',name='3',reuse=reuse_)
        layer_1=tf.layers.batch_normalization(layer_1,name='8',reuse=reuse_)
        layer_3=tf.maximum(0.01*layer_3,layer_3)
        layer_3=tf.nn.dropout(layer_3,0.8)

        layer_4=tf.layers.conv2d(layer_3, filters_d*8, 3, 2, 'same',name='4',reuse=reuse_)
        layer_4=tf.layers.batch_normalization(layer_4,name='9',reuse=reuse_)
        layer_4=tf.maximum(0.01*layer_4,layer_4)
        layer_4=tf.nn.dropout(layer_4,0.8)  
              
        layer_5=tf.layers.conv2d(layer_4, filters_d*16, 3, 2, 'same',name='5',reuse=reuse_)
        layer_5=tf.layers.batch_normalization(layer_5,name='10',reuse=reuse_)
        layer_5=tf.maximum(0.01*layer_5,layer_5)
        layer_5=tf.nn.dropout(layer_5,0.8)
        
        layer_6=tf.reshape(layer_5, [-1,3*3*filters_d*16])
    
        logits=tf.layers.dense(layer_6,1,name='6',reuse=reuse_)

    return logits

G_outputs=Gen(inputs_fake)
logits_D=Dis(inputs_real)
logits_f=Dis(G_outputs, reuse_=True)

d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_D),logits=logits_D))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_f),logits=logits_f))
d_loss=tf.add(d_loss_real,d_loss_fake)
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_f),logits=logits_f))

# train_vars=tf.trainable_variables()
# g_vars=[var for var in train_vars if var.name.startswith('G')]
# d_vars=[var for var in train_vars if var.name.startswith('D')]
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     d_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(d_loss,var_list=d_vars)
#     g_opt=tf.train.AdamOptimizer(0.001,0.4).minimize(g_loss,var_list=g_vars)

d_opt=tf.train.AdamOptimizer(0.001,beta1=0.4).minimize(d_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D')) 
g_opt=tf.train.AdamOptimizer(0.001,beta1=0.4).minimize(g_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))

list_g_loss,list_d_loss,list_g,list_loss_real,list_loss_fake=[],[],[],[],[]
j,k,m=2,500,0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(j):
#         sess.run(tf.assign(lr,0.001*(0.5**epoch)))
        for i in range(k):
            x_fake=np.random.uniform(-1,1,[100,100])
            o_G_outputs,o_g_loss,o_d_loss,o_d_loss_real,o_d_loss_fake=sess.run([d_opt,g_opt,G_outputs,g_loss,d_loss,d_loss_real,d_loss_fake],
                                                   {inputs_real:images[i*100:(i+1)*100],inputs_fake:x_fake})[2:]
            list_g_loss.append(o_g_loss)
            list_d_loss.append(o_d_loss)
            list_loss_real.append(o_d_loss_real)
            list_loss_fake.append(o_d_loss_fake)
            m+=1 
            if m%(j*k/10)==0:
                list_g.append(o_G_outputs) 
                print('完成%s'%(100*m/(j*k))+'%！')
#             G_layer1,G_layer2,G_layer3,G_layer4,G_outputs=sess.run([G_layer1,G_layer2,G_layer3,G_layer4,G_outputs],{x:x_real[i*100:(i+1)*100],inputs_fake:x_fake[i*100:(i+1)*100]})
#             print('完成%s'%(100*m/(j*k))+'%！')
#     saver.save(sess,'net9/my_net.ckpt')
# b=np.reshape(o_G_outputs[50],[28,28])
# plt.figure()
# plt.imshow(b,cmap='Greys_r')
# plt.show()
# print(G_layer1.shape,G_layer2.shape,G_layer3.shape,G_layer4.shape,G_outputs.shape)
# print(np.array(o_G_outputs).shape)
# 显示结果
plt.figure()
plt.title('loss')
pl1=plt.plot(list_d_loss,'red')
pl2=plt.plot(list_g_loss,'k')#black
pl3=plt.plot(list_loss_real,'g')
pl4=plt.plot(list_loss_fake,'b')
plt.legend(('d_loss','g_loss','real_loss','fake_loss'),loc=0,ncol=2)
plt.show()


print(np.array(o_G_outputs).shape,np.max(o_G_outputs),np.min(o_G_outputs),np.average(o_G_outputs))


list_g=np.array(list_g)
index_list=np.linspace(0,100,21)[:-1]
list_show=[]
for m in range(10):
    for n in index_list:
        list_show.append(list_g[m,int(n)])

print(np.array(list_show).shape)
list_show=(np.array(list_show)/2)+0.5

count1,count2=0,0
for i in range(200):
    for j in range(96):
        for m in range(96):
            for n in range(3):
                if list_show[i,j,m,n]<0:
                    list_show[i,j,m,n]=0
                    count1+=1
                if list_show[i,j,m,n]>1:
                    list_show[i,j,m,n]=1
                    count2+=1

print(count1,count2)
print(np.max(list_show),np.min(list_show),np.average(list_show))

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=list_show
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

count3,count4=0,0
for i in range(200):
    for j in range(96):
        for m in range(96):
            for n in range(3):
                if list_last_show[i,j,m,n]<0:
                    list_last_show[i,j,m,n]=0
                    count3+=1
                if list_last_show[i,j,m,n]>1:
                    list_last_show[i,j,m,n]=1
                    count4+=1

print(count3,count4)
print(np.max(list_last_show),np.min(list_last_show),np.average(list_last_show))

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=list_last_show
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()