import tensorflow as tf
import numpy as np

import pickle
import normalization
import cv2
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
class autoencoder():
    def __init__(
            self,
            train_data = None,
            batch_size = 16,
            learning_rate = 0.0001,
            training_epochs = 20,
            time_scale = 20,
            param_file = False,
            is_train = True
                 ):

        self.train = train_data
        self.batch_size = batch_size
        self.lr = learning_rate
        self.learning_rate=learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.time_scale = time_scale

        self.build()
        print "Neural networks build!"
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./params/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                print "learning initialization!"
                self.learn()
        else:
            self.saver.restore(self.sess, "./params/train.ckpt")
            self.show()

    def build(self):

            self.input = tf.placeholder(tf.float32, shape = [None, 30, self.time_scale, 4], name='csi_input')
            self.tag = tf.placeholder(tf.float32, shape = [None, 120, 160, 1], name ='image_origin')
            self.output= tf.placeholder(tf.float32, shape = [None, 120, 160,1], name='image_output')
        

            with tf.variable_scope('CNN'):
                alpha = 0.01
                w_initializer = tf.random_normal_initializer(0.,0.1)
                b_initializer = tf.constant_initializer(0.1)

                self.W_e_conv1 = tf.get_variable('w1', [3, 3, 4, 16], initializer=w_initializer)
                b_e_conv1 = tf.get_variable('b1', [16, ], initializer=b_initializer)
                self.conv1 = tf.nn.relu(tf.add(self.conv2d(self.input, self.W_e_conv1), b_e_conv1))
                print self.conv1.shape

                self.W_e_conv2 = tf.get_variable('w2', [3, 3, 16, 32], initializer=w_initializer)
                b_e_conv2 = tf.get_variable('b2', [32, ], initializer=b_initializer)
                self.conv2 = tf.nn.relu(tf.add(self.conv2d(self.conv1, self.W_e_conv2), b_e_conv2))
                print self.conv2.shape

                self.W_e_conv3 = tf.get_variable('w3', [3, 3, 32, 64], initializer=w_initializer)
                b_e_conv3 = tf.get_variable('b3', [64, ], initializer=b_initializer)
                self.conv3 = tf.nn.relu(tf.add(self.conv2d(self.conv2, self.W_e_conv3), b_e_conv3))
                print self.conv3.shape
            

                self.W_e_conv4 = tf.get_variable('w4', [2, 2, 64, 128], initializer=w_initializer)
                b_e_conv4 = tf.get_variable('b4', [128, ], initializer=b_initializer)
                self.conv4 = tf.nn.relu(tf.add(tf.nn.conv2d(self.conv3, self.W_e_conv4, strides=[1,1,1,1], padding='SAME'), b_e_conv4))
                print self.conv4.shape
                self.conv4 = tf.reshape(self.conv4, [-1, 4 * 5 * 128])

                self.w2 = tf.get_variable('w5', [4 * 5 * 128, 8 * 10 * 128], initializer=w_initializer, )
                self.b2 = tf.get_variable('b5', [8 * 10 * 128, ], initializer=b_initializer,)
                encoder = tf.nn.relu(tf.matmul(self.conv4, self.w2) + self.b2)
                encoder = tf.reshape(encoder, [-1, 8, 10, 128])

                decoder_add = tf.image.resize_images(encoder, size=(15, 20),
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                self.W_d_conv_add = tf.get_variable('w_d_add', [1, 1, 128, 64], initializer=w_initializer)
                decoder_add = tf.nn.conv2d(decoder_add, self.W_d_conv_add, strides=[1, 1, 1, 1], padding='SAME', )
                decoder_add = tf.maximum(alpha * decoder_add, decoder_add)


                decoder_1 = tf.image.resize_images(decoder_add, size=(30, 40),
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                self.W_d_conv1 = tf.get_variable('w_d_1', [3, 3, 64, 32], initializer=w_initializer)
                decoder_1 = tf.nn.conv2d(decoder_1, self.W_d_conv1, strides=[1, 1, 1, 1], padding='SAME', )
                decoder_1 = tf.maximum(alpha * decoder_1, decoder_1)

                decoder_2 = tf.image.resize_images(decoder_1, size=(60, 80),
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                self.W_d_conv2 = tf.get_variable('w_d_2', [3, 3, 32, 8], initializer=w_initializer)
                decoder_2 = tf.nn.conv2d(decoder_2, self.W_d_conv2, strides=[1, 1, 1, 1], padding='SAME', )
                decoder_2 = tf.maximum(alpha * decoder_2, decoder_2)
                print decoder_2.shape

                decoder_3 = tf.image.resize_images(decoder_2, size=(120, 160),
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                self.W_d_conv3 = tf.get_variable('w_d_3', [3, 3, 8, 1], initializer=w_initializer)
                decoder_3 = tf.nn.conv2d(decoder_3, self.W_d_conv3, strides=[1, 1, 1, 1], padding='SAME', )
                decoder_3 = tf.maximum(alpha * decoder_3, decoder_3)
                print decoder_3.shape

                self.output = tf.reshape(decoder_3, [-1, 120, 160,1])
                max = tf.reduce_max(self.output)
                min  = tf.reduce_min(self.output)
                self.output=(self.output-min)/(max-min)
                self.output=tf.clip_by_value(self.output, 1e-7, 0.9999999 )


            with tf.variable_scope('loss'):
               
                alpha, beta, rho = 5e-6, 7.5e-6, 0.08
                Wset = [self.W_e_conv1, self.W_e_conv2, self.W_e_conv3, self.w2, self.W_d_conv1,self.W_d_conv2,self.W_d_conv3]
                self.loss =tf.reduce_mean(-tf.reduce_sum(self.tag*tf.log(self.output)+(1.0-self.tag)*tf.log(1.0-self.output)))
                
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def batch_Convert(self, csidata, image, csi_index_list):
        csidata_batch, image_batch = None, None


        for index in range(len(image)):
          
            xs = csidata[:,csi_index_list[index]-self.time_scale+1:csi_index_list[index]+1 ,:]
            ys = image[index]
         
            if (index)%4==0:
                
                csidata_batch = np.array([xs]) if csidata_batch is None else np.append(csidata_batch, [xs], axis=0)
                image_batch = np.array([ys]) if image_batch is None else np.append(image_batch, [ys], axis= 0)
    
        return csidata_batch, image_batch

    def learn(self):
        stop_flag=0
        for j in range(self.training_epochs):
            for train_data in self.train:
                print train_data[2].shape
                xs = train_data[0].astype(np.float32)
                xs = np.nan_to_num(xs)

                batch_xs, batch_ys = self.batch_Convert(xs, train_data[1], train_data[2])
                print batch_xs.shape
                batch_xs = np.reshape(batch_xs, [-1, 30, self.time_scale, 4])

                batch_ys = batch_ys.astype(np.float32)
                batch_ys = np.nan_to_num(batch_ys)
                batch_ys = np.reshape(batch_ys, [-1, 120, 160, 1])
                batch_ys = np.clip(batch_ys, 1e-7, 0.9999999)
                for i in range(2000):
                        loss = 0
                        _, c ,output,tag= self.sess.run([self.optimizer, self.loss,self.output,self.tag], feed_dict={self.input: batch_xs, self.tag: batch_ys})
                        if i==10:
                           pass
                        loss += c
                        if math.isnan(loss) is True:
                            stop_flag=1
                            break
                        if np.any(np.isnan(batch_xs)):
                            print "Input Nan Type Error!! "
                        if np.any(np.isnan(batch_ys)):
                            print "Tag Nan Type Error!! "
                        if i % 5 == 0:
                           print("Total Epoch:", '%d' % (j), "Pic Rpoch:",'%d' % (i), "total cost=", "{:.9f}".format(loss))
                if stop_flag==1:
                    break
            if stop_flag==1:
                break
        print("Optimization Finished!")
        self.saver.save(self.sess, "./params/train.ckpt")

    def show(self):
        """
        display the performance of autoencoder
        :return: a autoencoder model using unsupervised learning
        """
        count = 0
        for train_data in self.train:
          
            xs = train_data[0].astype(np.float32)
            xs = np.nan_to_num(xs)
            batch_xs, batch_ys = self.batch_Convert(xs, train_data[1], train_data[2])
       
            batch_xs = np.reshape(batch_xs, [-1, 30, self.time_scale, 4])

            output = self.sess.run(self.output, feed_dict={self.input: batch_xs})
        
            output = np.reshape(output, (-1, 120, 160, 1))
          
            for i in range(len(output)):
                output[i] = output[i] * 255
                output1 = output[i].astype(np.uint8)
                # cv2.imshow("Image", output1)
                # cv2.waitKey(0)
                # cv2.imwrite('generator/'+str(count)+'.jpg',output1)
                #
                #
                batch_ys[i] = batch_ys[i] * 255
                target = batch_ys[i].astype(np.uint8)
                cv2.imwrite('target/' + str(count) + '.jpg', target)
                count += 1


    def conv2d(self,  x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    def deconv2d(self, x,W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding = 'SAME')

def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho
def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

def batchNormalization(data):
    for each_item in range(len(data)):
        data[each_item] = normalization.MINMAXNormalization(data[each_item])

def package(train_data):
 
 
    csi_rx1,csi_rx2, image ,index= train_data[0], train_data[1],train_data[2],train_data[3]
    tn_data = np.append(csi_rx1, csi_rx2, axis=0)
    tn_data = np.transpose(tn_data, [1,2,0])


    return [tn_data, image,index]


if __name__ =="__main__":
    np.set_printoptions(threshold=np.inf)
    train_data=[]
    for i in range( ):#data range
        index=i+1
        if index == :#abnormal data
            pass
        elif  index == :#test data
             pass
            # with open('../data_523/data_index/training_data_' + str(index) + '.pkl', 'rb') as handle:
            #     data_temp = pickle.load(handle)
            # batchNormalization(data_temp[0])
            # #batchNormalization(data_temp[1])
            # data_nor = package(data_temp)
            # train_data.append(data_nor)
        else:
            with open('../data_523/data_index_dwt/training_data_' + str(index) + '.pkl', 'rb') as handle:
                data_temp = pickle.load(handle)
            batchNormalization(data_temp[0])
            #batchNormalization(data_temp[1])
            data_nor = package(data_temp)
            train_data.append(data_nor)


    print len(train_data)
    autoencoder(train_data=train_data)






