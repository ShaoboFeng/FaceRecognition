#encoding=utf8
import dlib
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import sys
import time
from skimage import data,transform,draw,io

class Cnn:

	def __init__(self,imgs,labs):
		self._images = imgs
		self._labels = labs
		self._index_in_epoch = 0
		self._num_examples = len(labs)
		self._epochs_completed = 0
		print('there is a ' + str(len(imgs))+ ' images')

	def next_batch(self,batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples: # epoch中的句子下标是否大于所有语料的个数，如果为True,开始新一轮的遍历
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples) # arange函数用于创建等差数组
			np.random.shuffle(perm)  # 打乱
			self._images = np.array(self._images)[perm]
			self._labels = np.array(self._labels)[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

	def conv2d(self,x,w):
		return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
	
	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
	
	def cnn_train(self,test_input,test_output):
		#输入输出层
		tf_input = tf.placeholder(tf.float32,[None,80,60,3],name="tf_input")
		tf_output = tf.placeholder(tf.float32,[None,2])
		#tf_output_one_hot = tf.one_hot(tf_output,2)
		#初始化权重
		w_conv1 = tf.Variable(tf.truncated_normal([5,5,3,32],stddev = 0.1))
		#初始化偏置项
		b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))

		x_image=tf.reshape(tf_input,[-1,80,60,3])
		#第一层卷积\池化
		h_conv1 = tf.nn.relu(self.conv2d(x_image,w_conv1)+b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)	

		#第二层操作	
		#初始化权重
		w_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
		#初始化偏置项
		b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
		#第二层卷积\池化
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1,w_conv2)+b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		#全链接层
		#初始化权重
		w_fc1 = tf.Variable(tf.truncated_normal([20*15*64,1024],stddev = 0.1))
		#初始化偏置项
		b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
		h_pool2_flat = tf.reshape(h_pool2,[-1,20*15*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

		#输出层
		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

		#初始化权重
		w_fc2 = tf.Variable(tf.truncated_normal([1024,2],stddev = 0.1))
		#初始化偏置项
		b_fc2 = tf.Variable(tf.constant(0.1,shape=[2]))
		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2,name='y_conv')

		#下面就是一样的东西了,可以使用不通的优化函数
		#计算交叉熵的代价函数
		cross_entropy = -tf.reduce_sum(tf_output*tf.log(y_conv))
		#使用优化算法使得代价函数最小化
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		#找出预测正确的标签
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(tf_output,1))
		#得出通过正确个数除以总数得出准确率
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		saver = tf.train.Saver()
		sess = tf.InteractiveSession()
		sess.run(tf.initialize_all_variables())
		#每100次迭代输出一次日志，共迭代20000次
		for i in range(8000):
			batch = self.next_batch(100)
			if i%50 == 0:
				train_accuracy = accuracy.eval(feed_dict={tf_input:batch[0], tf_output: batch[1], keep_prob: 1.0})
				print str(time.time()) + " step %d, training accuracy %g"%(i, train_accuracy)
			train_step.run(feed_dict={tf_input: batch[0], tf_output: batch[1], keep_prob: 0.5})
			if train_accuracy > 0.95 :
				saver.save(sess,"model/model.ckpt")
				break

		print "test accuracy %g" % accuracy.eval(feed_dict={tf_input: test_input, tf_output: test_output, keep_prob: 1.0})
		saver.save(sess,"model/model.ckpt")

		

class FaceRec:
	def face_detector(self,image):
		'''face detector'''
		detector = dlib.get_frontal_face_detector()
		dets = detector(image,1)
		rect = dlib.rectangle
		rects = []
		for d in dets:
			rects.append((rect.top(d),rect.left(d),rect.bottom(d),rect.right(d)))
		return rects

	def normalize_face(self,filename):
		img = cv2.imread(filename)
		if 'MyFace' not in filename:
			return cv2.resize(img,(60,80))
		faces = self.face_detector(img)
		for face in faces:
			faceimage = img[face[0]:face[2],face[1]:face[3],:]
			faceimage = cv2.resize(faceimage,(60,80))
			#图像灰化
			#faceimage = cv2.cvtColor(faceimage,cv2.COLOR_BGR2GRAY)
			return faceimage
	def readtraindata(self):
		myface_path = "./face_photos/MyFace"
		otherface_path = "./face_photos/others"
		return self.readdata(myface_path,otherface_path)
		
	def readtestdata(self):
		myface_path = "./face_photos/test_MyFace"
		otherface_path = "./face_photos/test_other"
		return self.readdata(myface_path,otherface_path)
		
	def readdata(self,myface_path,otherface_path):
		imgs = []
		labs = []
		for filename in os.listdir(myface_path):
			if filename.endswith('.jpg'):
				filename = myface_path+'/'+filename
				imgs.append(self.normalize_face(filename))
				labs.append([0,1])
		for filename in os.listdir(otherface_path):
			if filename.endswith('.jpg'):
				filename = otherface_path+'/'+filename
				imgs.append(self.normalize_face(filename))
				labs.append([1,0])
		return imgs,labs

	def train(self):
		'''this method only run once for train a model'''
		imgs,labs = self.readtraindata()
		print('load train data ok')
		#cnn 神经网络算法
		#print((imgs,labs))
		cnn = Cnn(imgs,labs)
		test_imgs,test_labs = self.readtestdata()
		print('load test data ok')
		cnn.cnn_train(test_imgs,test_labs)

	def recognition(self):
		'''through recongnite a photo, make sure there is contain me in the pic?'''
		test_imgs,test_labs = self.readtestdata()
		tf_output = tf.Variable(tf.zeros([2]))
		tf_input = tf.placeholder(tf.float32,[None,80,60,3])
		saver = tf.train.import_meta_graph("model/model.ckpt.meta")
		saver = tf.train.Saver()
		with tf.Session() as sess:
			#sess.run(tf.global_variables_initializer())
			saver.restore(sess,"model/model.ckpt")
			graph = tf.get_default_graph()
			tf_input = graph.get_tensor_by_name("tf_input:0") 
			tf_output = graph.get_tensor_by_name("y_conv:0")
			
			ret = sess.run(tf_output,feed_dict={tf_input:test_imgs})
			print(ret)
			print(tf.argmax(classification_result,1),eval())

			for i in ret:
				print(i)


def test_findface():
	facerec = FaceRec()
	img = io.imread("./test_photo/timg.jpeg")
	faces = facerec.face_detector(img)
	'''show one face'''
	for face in faces:
		faceimage = img[face[0]:face[2],face[1]:face[3],:]
		faceimage = transform.resize(faceimage,(80,60))
		#图像灰化
		faceimage = cv2.cvtColor(faceimage,cv2.COLOR_BGR2GRAY)
		plt.imshow(faceimage)
		plt.show()
	return

	'''show all face'''
	for face in faces:
		print(face)
		rr,cc = draw.line(face[0],face[1],face[2],face[1])
		draw.set_color(img,(rr,cc),[0,0,255])
		rr,cc = draw.line(face[0],face[1],face[0],face[3])
		draw.set_color(img,(rr,cc),[0,0,255])
		rr,cc = draw.line(face[2],face[1],face[2],face[3])
		draw.set_color(img,(rr,cc),[0,0,255])
		rr,cc = draw.line(face[0],face[3],face[2],face[3])
		draw.set_color(img,(rr,cc),[0,0,255])
	plt.imshow(img)
	plt.show()

if __name__=="__main__":
	print("start program")
	#test_findface()
	print sys.argv
	if len(sys.argv) != 2:
		exit(1)
	facerec = FaceRec()
	if 'train' in sys.argv[1]:
		facerec.train()

	if 'test' in sys.argv[1]:
		facerec.recognition()
		exit(1)
		cap = cv2.VideoCapture(0)
		while cap.isOpened():
			ret,img = cap.read()
			faces = facerec.face_detector(img)
			fs = []
			for face in faces:
				faceimage = img[face[0]:face[2],face[1]:face[3],:]
				faceimage = cv2.resize(faceimage,(60,80))
				fs.append(faceimage)
			#facerec.recognition(fs)

			cv2.imshow("me",faceimage)
			if(cv2.waitKey(1) & 0xFF) == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
		pass

