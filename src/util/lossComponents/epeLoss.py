import tensorflow as tf
from components import *
import numpy as np

ground_truth = np.expand_dims(np.load("/home/y788zhang/joint_seg_flow/src/train_hs_lambda10000.npy"), 0)[:,16:-16,11:-11,:]
ground_truth = tf.cast(ground_truth, tf.float32)

def epeLoss(flow):
	with tf.variable_scope(None, default_name="epeLoss"):
		size = [flow.shape[1].value, flow.shape[2].value]
		scale = 448 / size[0]
        # usually not needed. only used for multiscale experiments
		flowDiff = flow - tf.image.resize_bilinear(ground_truth / (2 ** (scale - 1)), size)
		flowDist = charbonnierLoss(flowDiff, 1, 1, 0.001)
		return tf.reduce_sum(flowDist, axis=3, keep_dims=True)
