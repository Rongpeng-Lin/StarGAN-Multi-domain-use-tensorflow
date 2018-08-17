import tensorflow as tf
import numpy as np
import scipy.misc as misc
import math,os
import functools
from utlis import *
from ops import *
from get_inputs import *

class StarGan:
    def __init__(self,args):                         
        self.batch_size = args.batch_size
        print(args.features_A)
        self.f1 = args.features_A[0].split(',')
        self.f2 = args.features_B[0].split(',')
        
        self.real_im = tf.placeholder(tf.float32,[None,256,256,3],'real_im')
        self.real_domin = tf.placeholder(tf.float32,[None,2],'real_domin')
        self.label_tar = tf.placeholder(tf.float32,[None,len(self.f1+self.f2)],'label_tar')  
        self.label_real = tf.placeholder(tf.float32,[None,len(self.f1+self.f2)],'label_real')
 
        self.ident = tf.placeholder(tf.float32,[None,len(self.f1+self.f2)],'ident')
        self.lr = tf.placeholder(tf.float32,name='lr')
        self.init_lr = args.lr
        
        self.txt1_dir = args.txt1_dir
        self.txt2_dir = args.txt2_dir

        self.imfile1 = args.imfile1
        self.imfile2 = args.imfile2
        self.epoch = args.epoch
        self.ld = args.ld
        self.adv_weight = args.adv_weight
        self.cls_weight = args.cls_weight
        self.recy_weight = args.recy_weight
        self.save_path = args.save_path
        self.save_iters = args.save_iters
        self.logdir = args.logdir
        self.ckpt_path = args.ckpt_path
        self.test_ckpt = args.test_ckpt
        self.test_imsdir = args.test_imsdir
        self.test_save_path = args.test_save_path
        self.test_target = args.test_target[0].split(',')
        
#  label: [batch,9]
    def G(self,im,one_domin,label,reu):
        with tf.variable_scope('G_net',reuse=reu):
            label_shape = [i.value for i in label.get_shape()]
            domin_shape = [i.value for i in one_domin.get_shape()]
            label2im = tf.tile(tf.reshape(label,[-1,1,1,label_shape[-1]]),[1,256,256,1])
            domin2im = tf.tile(tf.reshape(one_domin,[-1,1,1,domin_shape[-1]]),[1,256,256,1])
            x_ = tf.concat([im,label2im,domin2im],3)      
            conv1 = conv('conv1',x_,7*7,64,1,True)
            ins1 = ins_norm('ins1',conv1)
            relu1 = relu('relu1',ins1)
            conv2 = conv('conv2',relu1,4*4,128,2,True)
            ins2 = ins_norm('ins2',conv2)
            relu2 = relu('relu2',ins2)
            conv3 = conv('conv3',relu2,4*4,256,2,True)
            ins3 = ins_norm('ins3',conv3)
            relu3 = relu('relu3',ins3)
            block1 = block('block1',relu3)
            block2 = block('block2',block1)
            block3 = block('block3',block2)
            block4 = block('block4',block3)
            block5 = block('block5',block4)
            block6 = block('block6',block5)
            up_1 = upsample('up1',block6,4*4,128,2,True)
            ins_up1 = ins_norm('ins_up1',up_1)
            relu_up1 = relu('relu_up1',ins_up1)
            up_2 = upsample('up_2',relu_up1,4*4,64,2,True)
            ins_up2 = ins_norm('ins_up2',up_2)
            relu_up2 = relu('relu_up2',ins_up2)
            conv_end = conv('conv_end',relu_up2,7*7,3,1,True)
            tanh_end = tanh('tanh_end',conv_end)
#             shapes = self.look_shape(conv1,conv2,conv3,block1,block2,block3,block4,block5,block6,up_1,up_2,conv_end)
            return tanh_end
        
    def D(self,im,reu):
        with tf.variable_scope('D_net',reuse=reu):
            conv1 = conv('conv1',im,4*4,64,2,True)
            lrelu1 = lrelu('lrelu1',conv1)
            
            conv2 = conv('conv2',lrelu1,4*4,128,2,True)
            lrelu2 = lrelu('lrelu2',conv2)
            
            conv3 = conv('conv3',lrelu2,4*4,256,2,True)
            lrelu3 = lrelu('lrelu3',conv3)
            
            conv4 = conv('conv4',lrelu3,4*4,512,2,True)
            lrelu4 = lrelu('lrelu4',conv4)
            
            conv5 = conv('conv5',lrelu4,4*4,1024,2,True)
            lrelu5 = lrelu('lrelu5',conv5)
            
            conv6 = conv('conv6',lrelu5,4*4,2048,2,True)
            lrelu6 = lrelu('lrelu6',conv6)
            
            conv_torF = conv('conv_TorF',lrelu6,3*3,1,1,True)
            conv_TorF = sigmoid('conv_TorF',conv_torF)
            conv_cls = conv('conv_cls',lrelu6,4*4,len(self.f1+self.f2),1,False)
            conv_Cls = sigmoid('conv_Cls',tf.reshape(conv_cls,[-1,len(self.f1+self.f2)]))
            return conv_TorF,conv_Cls

    def cls_loss(self,probs,label_real,probs1,label_tar,ident):
        d_loss = ((-1)*tf.log(probs)*label_real-tf.log(1-probs)*(1-label_real))*ident
        D_loss = tf.reduce_mean(tf.reduce_sum(d_loss,1,True)/tf.reduce_sum(ident,1,True))
        
        g_loss = ((-1)*tf.log(probs1)*label_tar-tf.log(1-probs1)*(1-label_tar))*ident
        G_loss = tf.reduce_mean(tf.reduce_sum(g_loss,1,True)/tf.reduce_sum(ident,1,True))
        return D_loss,G_loss
  
    def cycle_loss(self,im1,im2):
        return tf.reduce_mean(tf.abs(im1-im2))
  
    def gradient_panalty(self,real,fake):
        shape = tf.shape(real)
        epsilong = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        _, var = tf.nn.moments(real, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(var)  # magnitude of noise decides the size of local region
        noise = 0.5 * x_std * epsilong  # delta in paper
        alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
        interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X
        TorF, _ = self.D(interpolated,True)

        grad = tf.gradients(TorF,interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(tf.layers.flatten(grad), axis=1) # l2 norm
        grad_penalty = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))
        return grad_penalty
    
    def adv_loss(self,r_or_f,r_or_f1):
        avdloss_d = tf.reduce_mean((-1)*tf.log(r_or_f)+(-1)*tf.log(1-r_or_f1))
        advloss_g = tf.reduce_mean((-1)*tf.log(r_or_f1))
        return avdloss_d,advloss_g
    
    def build(self):
        self.fake_im = self.G(self.real_im,self.real_domin,self.label_tar,False)
        self.recons_im = self.G(self.fake_im,self.real_domin,self.label_real,True)
        self.r_or_f,self.probs = self.D(self.real_im,False)
        self.r_or_f1,self.probs1 = self.D(self.fake_im,True)
        
        self.adv_D,self.adv_G = self.adv_loss(self.r_or_f,self.r_or_f1)
        
        self.class_D,self.class_G = self.cls_loss(self.probs,self.label_real,self.probs1,self.label_tar,self.ident)
        
        self.recyle_loss = self.cycle_loss(self.real_im,self.recons_im)
        self.d_sum_loss = self.adv_weight * self.adv_D + self.cls_weight * self.class_D
        self.g_sum_loss = self.adv_weight * self.adv_G + self.cls_weight * self.class_G + self.recy_weight * self.recyle_loss       
        
    def train(self):
        self.build()
        G_vars = [var for var in tf.trainable_variables() if 'G_net' in var.name]
        D_vars = [var for var in tf.trainable_variables() if 'D_net' in var.name]
        G_optim = tf.train.AdamOptimizer(self.lr,beta1=0.5,beta2=0.999).minimize(self.g_sum_loss,var_list=G_vars)
        D_optim = tf.train.AdamOptimizer(self.lr,beta1=0.5,beta2=0.999).minimize(self.d_sum_loss,var_list=D_vars)
        
        im_names, domin, real_label = get_inputs(self.txt1_dir,self.txt2_dir,self.f1,self.f2,self.imfile1,self.imfile2) 
        data = tf.data.Dataset.from_tensor_slices((im_names, domin, real_label))
        data = data.map(process)
        data = data.shuffle(1000).batch(self.batch_size).repeat()
        Data = data.make_one_shot_iterator()
        Ims, Domin, Real_label = Data.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Graph = tf.summary.FileWriter(self.logdir,sess.graph)
            Saver = tf.train.Saver(max_to_keep=10)
            for ep in range(self.epoch):
                lear_rate = self.init_lr-ep*(self.init_lr/self.epoch) if ep<=(self.epoch-1) else self.init_lr/self.epoch
                for num in range(len(im_names)//self.batch_size):
                    ims,dom,real_l = sess.run([Ims,Domin,Real_label])
                    ident, l_tar = decidel(self.batch_size, dom, real_l, len(self.f1), len(self.f2))
                    dict_D = {self.real_im:ims,self.label_real:real_l,self.label_tar:l_tar,self.real_domin:dom,self.ident:ident,self.lr:lear_rate}
                    _, dsum, advd, classd, pprobs, pprobs1 = sess.run([D_optim, self.d_sum_loss, self.adv_D, self.class_D, self.probs,self.probs1],feed_dict=dict_D)
                    print('current_epoch: ',ep)
                    print('  total_loss_D: %f, adv_loss_d: %f, class_loss_d: %f'%(dsum,advd,classd))
#                     _,D_Loss,gp = sess.run([D_optim,self.d_sum_loss,self.GP],feed_dict=dict_D                             
                    dict_G = {self.real_im:ims,self.label_real:real_l,self.label_tar:l_tar,self.real_domin:dom,self.ident:ident,self.lr:lear_rate}
                    _, gsum, advg, classg, Fims = sess.run([G_optim, self.g_sum_loss, self.adv_G, self.class_G, self.fake_im],feed_dict=dict_G)
                    print('  total_loss_G: %f, adv_loss_g: %f, class_loss_g: %f'%(gsum, advg, classg))
                    if int(ep*len(im_names)+num*self.batch_size)%self.save_iters==0:
                        print('save images at epoch: %d, training numbers is %d'%(ep,num))
                        saveim(Fims,ep,num,self.save_path)
                Saver.save(sess, self.ckpt_path, global_step=ep)
                print('save_success! ')
            return True
        
    def read_testims(self):
        ims_dir = [self.test_imsdir+imname for imname in os.listdir(self.test_imsdir)]
        zeros = np.zeros([len(ims_dir),256,256,3])
        for i in range(len(ims_dir)):
            im = misc.imread(ims_dir[i])
            if np.shape(im)!=(256,256,3):
                im = misc.imresize(im,[256,256])
            zeros[i,:,:,:] = im
        Dict = self.get_domin_target(self.test_target,len(ims_dir))
        return zeros,Dict
    
    def get_domin_target(self,features,test_batch):
        fea_f1 = []
        fea_f2 = []
        for feature in features:
            if feature in self.f1:
                fea_f1.append(feature)
            else:
                fea_f2.append(feature)
        if fea_f1:
            domin = np.zeros([test_batch,2])
            domin[:,0] = 1.
            empty_f1 = np.zeros([test_batch,len(self.f1+self.f2)],np.float32)
            for fea in fea_f1:
                empty_f1[:,self.f1.index(fea)] = 1.
        if fea_f2: 
            domin2 = np.zeros([test_batch,2])
            domin2[:,1] = 1.
            empty_f2 = np.zeros([test_batch,len(self.f1+self.f2)],np.float32)
            for fea in fea_f2:
                empty_f2[:,len(self.f1)+self.f2.index(fea)] = 1.
        if fea_f1:
            if fea_f2:
                sin_or_dou = 2
                return {'sin_or_dou':sin_or_dou,'Dom1':domin,'Label_tar1':empty_f1,'Dom2':domin2,'Label_tar2':empty_f2}
            else:
                sin_or_dou = 1
                return {'sin_or_dou':sin_or_dou,'Dom':domin,'Label_tar':empty_f1}
        else:
            sin_or_dou = 1
            return {'sin_or_dou':sin_or_dou,'Dom':domin2,'Label_tar':empty_f2}
    
    def test(self):
        self.build()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Saver = tf.train.Saver()
            im_in,Dict = self.read_testims()
            if Dict['sin_or_dou']==2:
                Saver.restore(sess,tf.train.latest_checkpoint(self.test_ckpt))
                print('load_success')
                test_dict = {self.real_im:im_in,self.label_tar:Dict['Label_tar1'],self.real_domin:Dict['Dom1']}
                fake_ims = sess.run(self.fake_im,feed_dict=test_dict)
                test_dict2 = {self.real_im:fake_ims,self.label_tar:Dict['Label_tar2'],self.real_domin:Dict['Dom2']}
                Fake_ims = sess.run(self.fake_im,feed_dict=test_dict2)
                saveim(Fake_ims,0,0,self.test_save_path)
                print('save_success')
            else:
                Saver.restore(sess,tf.train.latest_checkpoint(self.test_ckpt))
                print('load_success')
                test_dict = {self.real_im:im_in,self.label_tar:Dict['Label_tar'],self.real_domin:Dict['Dom']}
                fake_ims = sess.run(self.fake_im,feed_dict=test_dict)
                saveim(fake_ims,0,0,self.test_save_path)
                print('save_success')
