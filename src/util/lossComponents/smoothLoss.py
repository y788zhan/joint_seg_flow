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
	# return tf.expand_dims(mask, 0)
	return tf.stack([mask, mask]) # batch_size = 2


MAX_WIDTH = 1
# KERNELS = [make_kernels(i) for i in xrange(1, MAX_WIDTH + 1)]
#X_MASKS = [make_mask(i, H, W, True) for i in xrange(1, MAX_WIDTH + 1)]
#Y_MASKS = [make_mask(i, H, W, False) for i in xrange(1, MAX_WIDTH + 1)]


KERNEL = tf.transpose(tf.constant([[[[0,-1,0],
                                     [0,1,0],
                                     [0,0,0]]],
                                   [[[0,0,0],
                                     [0,1,-1],
                                     [0,0,0]]],
                                   [[[0,0,0],
                                     [0,1,0],
                                     [0,-1,0]]],
                                   [[[0,0,0],
                                     [-1,1,0],
                                     [0,0,0]]]],
                                  dtype=tf.float32),perm=[3,2,1,0])


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



def make_gt_masks(gt_mask, w):
    H, W = gt_mask.shape[1], gt_mask.shape[2]
    normalizer = tf.zeros((2, H, W, 2)) # batch size 2

    multiplier_masks = []
    for i in range(w):
        multiplier_masks.append([])

    for i in range(w):
        j = i + 1
        multiplier_masks[i].append(
            tf.concat([
                tf.tile(tf.expand_dims(gt_mask[:,:,j:W,0], axis=-1), (1,1,1,2)),
                tf.zeros((2, H, j, 2), dtype=tf.float32)], axis=2))

        multiplier_masks[i].append(
            tf.concat([
                tf.zeros((2, j, W, 2), dtype=tf.float32),
                tf.tile(tf.expand_dims(gt_mask[:,0:(H-j),:,1], axis=-1), (1,1,1,2))], axis=1))

        multiplier_masks[i].append(
            tf.concat([
                tf.zeros((2, H, j, 2), dtype=tf.float32),
                tf.tile(tf.expand_dims(gt_mask[:,:,0:(W-j),2], axis=-1), (1,1,1,2))], axis=2))

        multiplier_masks[i].append(
            tf.concat([
                tf.tile(tf.expand_dims(gt_mask[:,j:H,:,3], axis=-1), (1,1,1,2)),
                tf.zeros((2, j, W, 2), dtype=tf.float32)], axis=1))

        normalizer += multiplier_masks[i][0] + multiplier_masks[i][1] + multiplier_masks[i][2] + multiplier_masks[i][3]

    # Remove 0's in normalizer
    normalizer += 4 * w * tf.cast(tf.equal(normalizer, 0), dtype=tf.float32)

    return multiplier_masks, normalizer


def fixed_point_update(flow, gamma, itr, multiplier_masks, normalizer):
    H, W = flow.shape[1], flow.shape[2]
    flow_copy1 = flow * 1.0
    flow_copy2 = flow * 1.0
    flow_copy1 = tf.stop_gradient(flow_copy1)
    flow_copy2 = tf.stop_gradient(flow_copy2)

    for k in range(itr):
        temp = tf.zeros_like(flow)
        for i in range(MAX_WIDTH):
            j = i + 1
            temp += tf.concat([flow_copy1[:,:,j:W,:],
                               tf.zeros((2, H, j, 2), dtype=tf.float32)], axis=2) * multiplier_masks[i][0]

            temp += tf.concat([tf.zeros((2, j, W, 2), dtype=tf.float32),
                               flow_copy1[:,0:(H-j),:,:]], axis=1) * multiplier_masks[i][1]

            temp += tf.concat([tf.zeros((2, H, j, 2), dtype=tf.float32),
                               flow_copy1[:,:,0:(W-j),:]], axis=2) * multiplier_masks[i][2]

            temp += tf.concat([flow_copy1[:,j:H,:,:],
                               tf.zeros((2, j, W, 2), dtype=tf.float32)], axis=1) * multiplier_masks[i][3]

        temp += gamma * flow_copy2
        flow_copy1 = temp / (normalizer + gamma)
    return flow_copy1


def smoothLoss(flow, gt, alpha, beta, verbose=False):
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

        gtMask = tf.nn.atrous_conv2d(gt, kernel, rate=1, padding="SAME")
        gtMask = 1 - tf.square(gtMask)

        flowShape = flow.get_shape()
	tf.summary.image("smooth_flow", flowToRgb1(flow))
        neighborDiffU = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME") * gtMask
        neighborDiffV = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME") * gtMask

        diffs = tf.concat([neighborDiffU,neighborDiffV],3)
	dists = charbonnierLoss(diffs, alpha, beta, 0.001)
	robustLoss = tf.reduce_sum(dists, axis=3, keep_dims=True)

        tf.summary.scalar("smooth_loss", tf.reduce_mean(robustLoss[:,2:-2,2:-2,:]))
	
	return robustLoss


def clampLoss(flow, fhat):
	diff = flow - fhat
	loss = charbonnierLoss(diff, 1, 1, 0.001)
	# loss = tf.reduce_sum(dist, axis=3, keep_dims=True)

	tf.summary.scalar("clamping", tf.reduce_mean(loss))
	return loss


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

