import tensorflow as tf
import numpy as np
import scipy.misc as misc

##    preprocess
def process(name,dom,lraw):
    im_uint = tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(name),3),[256,256])   
    im_flo = tf.image.convert_image_dtype(im_uint,tf.float32)
    im_Flo = im_flo/127.5 - 1
    return im_Flo,dom,lraw

#  d: [batch,domins], l: [batch,all_features]
def decidel(batch, d, lab, f1_len, f2_len):
    num_fea = np.shape(lab)[1]
    ident = np.zeros([batch,num_fea],np.float32)
    for i in range(f1_len):
        ident[:,i] = d[:,0]
    for i in range(f1_len,num_fea):
        ident[:,i] = d[:,1] 
    tar = np.copy(lab)
    for i in range(batch):
        if np.argmax(d[i,:])==0:
            np.random.shuffle(tar[i,0:f1_len])
        else:
            np.random.shuffle(tar[i,f1_len:num_fea])
    return ident, tar

def saveim(fims,ep,num,path):
    fs = (fims+1)/2
    ba,hi,wi = np.shape(fs)[0],np.shape(fs)[1],np.shape(fs)[2]
    zeros = np.zeros([hi,int(ba*wi),3],np.float32)
    for i in range(ba):
        zeros[:,int(i*wi):int((i+1)*wi),:] = fims[i,:,:,:]
    misc.imsave(path+str(ep)+'_'+str(num)+'.jpg',zeros)
