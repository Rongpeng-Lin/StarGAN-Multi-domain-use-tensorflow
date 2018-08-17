import tensorflow as tf
import math

def conv(name,x,ker_size,outs,s,pad):
    ker = int(math.sqrt(ker_size))
    shape = [i.value for i in x.get_shape()]
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,shape[3],outs],
                            tf.float32,
                            tf.initializers.truncated_normal(stddev=0.01))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        if_pad = "SAME" if pad else "VALID"
        return tf.nn.conv2d(x,w,[1,s,s,1],if_pad)+b

def ins_norm(name,x):
    with tf.variable_scope(name):
        return tf.contrib.layers.instance_norm(x)
    
def relu(name,x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def block(name,x):
    with tf.variable_scope(name):
        bconv = conv(name+'conv',x,3*3,256,1,True)
        b_in = ins_norm(name+'ins',bconv)
        b_relu = relu(name+'relu',b_in)
        return b_relu
    
def lrelu(name,x):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x,alpha=0.01)
    
def sigmoid(name,x):
    with tf.variable_scope(name):
        return tf.nn.sigmoid(x)

def tanh(name,x):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)
    
def upsample(name,x,ker_size,outs,s,pad):
    ker = int(math.sqrt(ker_size))
    shape = [i.value for i in x.get_shape()]
    x_big = tf.image.resize_images(x,[int(s*shape[1]),int(s*shape[2])])
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,shape[3],outs],
                            tf.float32,
                            tf.initializers.truncated_normal(stddev=0.01))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        if_pad = "SAME" if pad else "VALID"
        return tf.nn.conv2d(x_big,w,[1,1,1,1],if_pad)+b
