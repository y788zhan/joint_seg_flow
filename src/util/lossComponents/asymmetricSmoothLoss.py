import tensorflow as tf
from components import *
from smoothLoss import *

def asymmetricSmoothLoss(flow,gt,instanceParams,occMask,validPixelMask,img0Grad=None,boundaryAlpha=0, backward=False):
	"""
	modifies gradients so that smoothness can only go from non-occluded to occluded areas
	"""
	with tf.variable_scope(None,default_name="asymmetricSmoothLoss"):
		alpha = instanceParams["smoothParams"]["robustness"]
		beta = instanceParams["smoothParams"]["scale"]
		occAlpha = instanceParams["smoothOccParams"]["robustness"]
		occBeta = instanceParams["smoothOccParams"]["scale"]

		# non occluded
		#nonOccSmooth = clampLoss(flow, gt, alpha,beta,occMask,img0Grad,boundaryAlpha, verbose=not backward)
		nonOccSmooth = smoothLoss(flow, gt, alpha, beta)
		return nonOccSmooth

import math
def vis_flow(flow,zeroFlow="saturation"):
    with tf.variable_scope(None,default_name="flowToRgb"):
        mag = tf.sqrt(tf.reduce_sum(flow**2,axis=-1))
        ang180 = tf.atan2(flow[:,:,:,1],flow[:,:,:,0])
        ones = tf.ones_like(mag)

        # fix angle so righward motion is red
        ang = ang180*tf.cast(tf.greater_equal(ang180,0),tf.float32)
        ang += (ang180+2*math.pi)*tf.cast(tf.less(ang180,0),tf.float32)

        # normalize for hsv
        largestMag = tf.reduce_max(mag,axis=[1,2])
	magNorm = tf.stack([mag[0,:,:]/largestMag[0], mag[1,:,:]/largestMag[1]], axis=0)
	# magNorm = mag/largestMag
        angNorm = ang/(math.pi*2)

	if zeroFlow == "value":
                hsv = tf.stack([angNorm,ones,magNorm],axis=-1)
        elif zeroFlow == "saturation":
                hsv = tf.stack([angNorm,magNorm,ones],axis=-1)
        else:
                assert("zeroFlow mode must be {'value','saturation'}")
        rgb = tf.image.hsv_to_rgb(hsv)
        return rgb

