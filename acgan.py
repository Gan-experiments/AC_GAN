import tensorflow as tf
import numpy as np
import random
import datetime
from utils import*
import os  

from tfrec import TFRecordsReader

batch_size=64
class batchnorm():
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)
def lrelu(x, leak=0.2):
    return tf.maximum(x,x*leak)

def conv(x,num_filters,kernel=5,stride=[1,2,2,1],name="conv",padding='SAME'):
    with tf.variable_scope(name):
        w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        con=tf.nn.conv2d(x, w, strides=stride, padding=padding)
        return tf.reshape(tf.nn.bias_add(con, b),con.shape)
def fcn(x,num_neurons,name="fcn"):#(without batchnorm )
    with tf.variable_scope(name):

        w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b=tf.get_variable('b',shape=[num_neurons],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x,w)+b

def deconv(x,output_shape,kernel=5,stride=[1,2,2,1],name="deconv"):
    with tf.variable_scope(name):
        num_filters=output_shape[-1]
        w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b=tf.get_variable('b',shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        decon=tf.nn.conv2d_transpose(x, w, strides=stride,output_shape=output_shape)
        return tf.reshape(tf.nn.bias_add(decon, b),decon.shape)

def generator(z,y):
    with tf.variable_scope("generator"):
        y_onehot =    tf.one_hot(y,10)
        
        z_       =    tf.concat([z,y_onehot],1)

        h0       =    fcn(z_,num_neurons=512*4*4,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")
        
        h0       =    tf.nn.relu(gbn0(tf.reshape(h0,[batch_size,4,4,512]),train=True))

        gbn1     =    batchnorm(name="g_bn1")

        h1       =    tf.nn.relu(gbn1(deconv(h0,[batch_size,8,8,256],name="g_h1"),train=True))

        gbn2     =    batchnorm(name="g_bn2")

        h2       =    tf.nn.relu(gbn2(deconv(h1,[batch_size,16,16,128],name="g_h2"),train=True))

        gbn3     =    batchnorm(name="g_bn3")

        h3       =    tf.nn.relu(gbn3(deconv(h2,[batch_size,16*2,16*2,64],name="g_h3"),train=True))

#        gbn4     =    batchnorm(name="g_bn4") 

        h4       =    deconv(h3,[batch_size,16*2*2,16*2*2,1],name="g_h4")
	
	print "image generation till last layer done"        
        return tf.nn.tanh(h4)

def sampler(z,y):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()
        y_onehot =    tf.one_hot(y,10)
        
        z_       =    tf.concat([z,y_onehot],1)

        h0       =    fcn(z_,num_neurons=512*4*4,name="g_fcn0")

        gbn0     =    batchnorm(name="g_bn0")
        
        h0       =    tf.nn.relu(gbn0(tf.reshape(h0,[batch_size,4,4,512]),train=False))

        gbn1     =    batchnorm(name="g_bn1")

        h1       =    tf.nn.relu(gbn1(deconv(h0,[batch_size,8,8,256],name="g_h1"),train=False))

        gbn2     =    batchnorm(name="g_bn2")

        h2       =    tf.nn.relu(gbn2(deconv(h1,[batch_size,16,16,128],name="g_h2"),train=False))

        gbn3     =    batchnorm(name="g_bn3")

        h3       =    tf.nn.relu(gbn3(deconv(h2,[batch_size,16*2,16*2,64],name="g_h3"),train=False))

       # gbn4     =    batchnorm(name="g_bn4") 

        h4       =    deconv(h3,[batch_size,16*2*2,16*2*2,1],name="g_h4")
        print "sampling done"
        return tf.nn.tanh(h4)

def discriminator(imgs,reuse=False):

    with tf.variable_scope("discriminator") as scope:

        if reuse:

            scope.reuse_variables()

        h0       =    lrelu(conv(imgs,64,name="d_h0"))

        dbn1     =    batchnorm(name="d_bn1")
        
        h1       =    lrelu(dbn1(conv(h0,128,name="d_h1")))

        dbn2     =    batchnorm(name="d_bn2")

        h2       =    lrelu(dbn2(conv(h1,128*2,name="d_h2")))

        dbn3     =    batchnorm(name="d_bn3")

        h3       =    lrelu(dbn3(conv(h1,128*2*2,name="d_h3")))

        h4       =    tf.reshape(h3,[batch_size,-1])

        source   =    fcn(h4,1,name="d_source")

        classes  =    fcn(h4,10,name="d_classes")
	
	print "discrimination done"
        return source,classes



reader = TFRecordsReader(
            image_height=28,
            image_width=28,
            image_channels=1,
            image_format="bmp",
            directory="data/mnist",
            filename_pattern="*.tfrecords",
            crop=False,
            crop_height=64,
            crop_width=64,
            resize=True,
            resize_height=64,
            resize_width=64,
            num_examples_per_epoch=64)
images, labels = reader.inputs(batch_size=64)
float_images = tf.cast(images, tf.float32)
float_images = float_images / 127.5 - 1.0
images, labels = reader.inputs(batch_size=64)
float_images = tf.cast(images, tf.float32)
images = float_images / 127.5 - 1.0
with tf.device('/gpu:0'):
    z               =   tf.placeholder(tf.float32, [batch_size, 100], name='z')

    gamma	    =   tf.placeholder(dtype=tf.float32)
    epsilon	    =   tf.placeholder(dtype=tf.float32)
    labels_one_hot  =   tf.one_hot(labels,10)
    gen_imgs        =   generator(z=z,y=labels)
    real_source,real_class     =   discriminator(images,reuse=False)
    fake_source,fake_class     =   discriminator(gen_imgs,reuse=True)
    samples         =   sampler(z=z,y=labels)
    source_loss_fake     =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels=tf.zeros_like(fake_source)))

    source_loss_real     =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_source,labels=tf.ones_like(real_source)))

    class_loss_fake      =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_class,labels=labels_one_hot))

    class_loss_real      =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_class,labels=labels_one_hot))

    d_loss = source_loss_real + source_loss_fake + class_loss_real + class_loss_fake

    def g_l2_loss():
		return tf.nn.l2_loss(tf.reduce_mean(gen_imgs-images,axis=0))
   # g_loss = class_loss_real + class_loss_fake + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels=tf.ones_like(fake_source)))

    def g_loss_gan():
		return class_loss_real + class_loss_fake + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_source,labels=tf.ones_like(fake_source)))

    g_loss=tf.cond(epsilon<gamma,g_l2_loss,g_loss_gan) 
   
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'd_' in var.name]

    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_opt  = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(d_loss,var_list=d_vars)

    g_opt  = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(g_loss,var_list=g_vars)
    
    gamma_init=100.0
    
    factor=2.0

    init   = tf.global_variables_initializer()

    config = tf.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    with tf.Session(config=config) as sess:

        sess.run(init)
	
	tf.train.start_queue_runners(sess=sess)
	
	print "initialized all variables"

        for epoch in range(5000):
	
            eps = random.random()

	    print "in epoch ", epoch

            z_=np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)

            sess.run(d_opt,feed_dict={z:z_})

	    print "optimized d in epoch ",epoch

            sess.run(g_opt,feed_dict={z:z_,epsilon:eps,gamma:gamma_init})
            
            print "optimized g first time in epoch ",epoch

            sess.run(g_opt,feed_dict={z:z_,epsilon:eps,gamma:gamma_init})#similar to DCGAN

            print "optimized g second time in epoch ",epoch

            D_loss = sess.run(d_loss,feed_dict={z:z_})

	    print "d loss in epoch ",epoch," is ",D_loss
            
	    G_loss = sess.run(g_loss,feed_dict={z:z_,epsilon:eps,gamma:gamma_init})
		
	    print "g loss in epoch ",epoch," is ",G_loss
            
            gamma_init=gamma_init/factor

	    if epoch % 10 ==0:
                
                sample = sess.run(samples,feed_dict={z:z_})

                save_images(sample, image_manifold_size(sample.shape[0]),
                          './{}/{:02d}.png'.format("ac_out", epoch))
