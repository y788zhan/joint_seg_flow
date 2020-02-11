import tensorflow as tf
from deconvLayer import *

def flowRefinementConcat(prev,skip,flow):
        prev = tf.image.resize_images(prev, skip.shape[1:3])
	with tf.variable_scope(None,default_name="flowRefinementConcat"):
		with tf.variable_scope(None,default_name="upsampleFlow"):
			upsample = deconvLayer(flow,4,2,2)
		upsample = tf.image.resize_images(upsample, skip.shape[1:3])
		return tf.concat([prev,skip,upsample],3)
