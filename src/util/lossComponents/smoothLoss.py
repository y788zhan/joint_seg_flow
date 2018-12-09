import tensorflow as tf
from components import *
import numpy as np

def make_kernel(width, horizontal=True):
	kernel = []
	for _ in xrange(2 * width + 1):
		kernel.append([0] * (2 * width + 1))
	kernel[width][width] = 1
	if horizontal:
		kernel[width][-1] = -1
	else:
		kernel[-1][width] = -1
	return kernel

def make_kernels(width):
	return tf.transpose(
		tf.constant(
			[[make_kernel(width, True)], [make_kernel(width, False)]],
			dtype=tf.float32),
		perm=[3,2,1,0])

def make_mask(kernel_width, height, width, horizontal = True):
	mask = np.ones((height, width))
	if horizontal:
		for i in xrange(kernel_width):
			mask[:,width - i - 1] = 0
	else:
		for i in xrange(kernel_width):
			mask[height - i - 1,:] = 0
	mask = tf.cast(mask, tf.float32)
	return tf.stack([mask, mask]) # batch_size = 2


MAX_WIDTH = 40
# KERNELS = [make_kernels(i) for i in xrange(1, MAX_WIDTH + 1)]
X_MASKS = [make_mask(i, 480, 854, True) for i in xrange(1, MAX_WIDTH + 1)]
Y_MASKS = [make_mask(i, 480, 854, False) for i in xrange(1, MAX_WIDTH + 1)]


def smoothLossMaskCorrection(validMask):
	"""
	makes correct mask for smoothness based on a valid pixel mask
	if any invalid pixel is within the inclusion kernel, ignore
	"""

	inclusionKernel = tf.transpose(tf.constant([\
		[ \
			[ \
				[0,0,0],\
				[0,1,1],\
				[0,1,0]\
			] \
		] \
	],dtype=tf.float32),perm=[3,2,1,0])

	maskCor = tf.nn.conv2d(validMask,inclusionKernel,[1,1,1,1],padding="SAME")
	maskCor = tf.greater_equal(maskCor,2.95)
	maskCor = tf.cast(maskCor,tf.float32)

	return maskCor




def smoothLoss(flow,gt,alpha,beta,validPixelMask=None,img0Grad=None,boundaryAlpha=0,verbose=False):
	"""
	smoothness loss, includes boundaries if img0Grad != None
	"""
	kernel = tf.transpose(tf.constant([\
		[ \
			[ \
				[0,0,0],\
				[0,1,-1],\
				[0,0,0]\
			] \
		], \
		[ \
			[ \
				[0,0,0],\
				[0,1,0],\
				[0,-1,0]\
			] \
		] \
	],dtype=tf.float32),perm=[3,2,1,0])

	with tf.variable_scope(None,default_name="smoothLoss"):
		u = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
		v = tf.slice(flow,[0,0,0,1],[-1,-1,-1,-1])
		robustLoss = None

		flowShape = flow.get_shape()
		if verbose:
			tf.summary.image("smooth_flow", flowToRgb1(flow))
		for i in xrange(MAX_WIDTH):
			x_mask = X_MASKS[i]
			y_mask = Y_MASKS[i]
			# gtMask = tf.nn.conv2d(gt,kernel,[1,1,1,1],padding="SAME",dilations=[1, 1, 1, 1])
			gtMask = tf.nn.atrous_conv2d(gt, kernel, rate=i+1, padding="SAME")
			gtMask = 1 - tf.square(gtMask)
			gtMask = tf.stack([gtMask[:,:,:,0] * x_mask, gtMask[:,:,:,1] * y_mask], axis=-1)
			neighborDiffU = tf.nn.atrous_conv2d(u, kernel, rate=i+1, padding="SAME") * gtMask
			neighborDiffV = tf.nn.atrous_conv2d(v, kernel, rate=i+1, padding="SAME") * gtMask
			# neighborDiffU = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",dilations=[1, 1, 1, 1]) * gtMask
			# neighborDiffV = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME",dilations=[1, 1, 1, 1]) * gtMask

			diffs = tf.concat([neighborDiffU,neighborDiffV],3)
			dists = tf.reduce_sum(tf.abs(diffs),axis=3,keep_dims=True)
			if robustLoss is None:
				robustLoss = charbonnierLoss(dists,alpha,beta,0.001)
			else:
				# robustLoss += charbonnierLoss(dists,alpha * (1 - 0.8 * i / (MAX_WIDTH - 1)), beta,0.001)
				robustLoss += charbonnierLoss(dists, alpha, beta, 0.001)
		# if not img0Grad == None:
		# 	dMag = tf.sqrt(tf.reduce_sum(img0Grad**2, axis=3, keep_dims=True))
		# 	mask = tf.exp(-boundaryAlpha*dMag)
		# 	robustLoss *= mask

			# debug
			# tf.summary.image("boundaryMask", mask)
		return robustLoss
		
		if validPixelMask is None:
			return robustLoss
		else:
			return robustLoss*smoothLossMaskCorrection(validPixelMask)


import math
def flowToRgb1(flow,zeroFlow="saturation"):
	with tf.variable_scope(None,default_name="flowToRgb"):
		mag = tf.sqrt(tf.reduce_sum(flow**2,axis=-1))
		ang180 = tf.atan2(flow[:,:,:,1],flow[:,:,:,0])
		ones = tf.ones_like(mag)

		# fix angle so righward motion is red
		ang = ang180*tf.cast(tf.greater_equal(ang180,0),tf.float32)
		ang += (ang180+2*math.pi)*tf.cast(tf.less(ang180,0),tf.float32)

		# normalize for hsv
		largestMag = tf.reduce_max(mag,axis=[1,2])
		magNorm = tf.stack([mag[0,:,:] / largestMag[0], mag[1,:,:] / largestMag[1]], axis=0)
		angNorm = ang/(math.pi*2)

		if zeroFlow == "value":
				hsv = tf.stack([angNorm,ones,magNorm],axis=-1)
		elif zeroFlow == "saturation":
				hsv = tf.stack([angNorm,magNorm,ones],axis=-1)
		else:
				assert("zeroFlow mode must be {'value','saturation'}")
		rgb = tf.image.hsv_to_rgb(hsv)
		return rgb

