import tensorflow as tf
from components import *
from photoLoss import *
from gradLoss import *
from smoothLoss import *
from smoothLoss2nd import *
from asymmetricSmoothLoss import *

def unsupFlowLoss(flow,flowB,frame0,frame1,validPixelMask,instanceParams, backward=False):
	with tf.variable_scope(None,default_name="unsupFlowLoss"):
		# hyperparams
		photoAlpha = instanceParams["photoParams"]["robustness"]
		photoBeta = instanceParams["photoParams"]["scale"]

		smoothReg = instanceParams["smoothParams"]["weight"]

		smooth2ndReg = instanceParams["smooth2ndParams"]["weight"]
		smooth2ndAlpha = instanceParams["smooth2ndParams"]["robustness"]
		smooth2ndBeta = instanceParams["smooth2ndParams"]["scale"]

		gradReg = instanceParams["gradParams"]["weight"]
		gradAlpha = instanceParams["gradParams"]["robustness"]
		gradBeta = instanceParams["gradParams"]["scale"]

		boundaryAlpha = instanceParams["boundaryAlpha"]
		lossComponents = instanceParams["lossComponents"]

		# gamma over lambda
		GOL = 10.0

		# helpers
		size = [flow.shape[1], flow.shape[2]]
		scale = 448 / size[0].value
		rgb0 = tf.image.resize_bilinear(frame0["rgbNorm"], size)
		rgb1 = tf.image.resize_bilinear(frame1["rgbNorm"], size)
		grad0 = tf.image.resize_bilinear(frame0["grad"], size)
		grad1 = tf.image.resize_bilinear(frame1["grad"], size)
		gt = tf.image.resize_bilinear(frame0["gt"], size)
		gt1 = tf.image.resize_bilinear(frame1["gt"], size)
		# if not backward:
		# 	tf.summary.image("rgb0", rgb0)
		# 	tf.summary.image("rgb1", rgb1)
		# 	tf.summary.image("gt", gt)
		# masking from simple occlusion/border
		#occMask = borderOcclusionMask(flow) # occ if goes off image
		occInvalidMask = 1#validPixelMask*occMask # occluded and invalid
		
		# loss components
		photo = photoLoss(flow,rgb0,rgb1,photoAlpha,photoBeta)

		# occMask = occluMask(flow, flowB, alpha2 = 0.5 / (scale ** 2), backward=backward)
		# photo = photo * (1.0 - occMask) + 100 * occMask
		if flow.shape[1] == 448:
			tf.summary.image("occlusion_mask", occMask)

		# grad = gradLoss(flow,grad0,grad1,gradAlpha,gradBeta)
		imgGrad = None
		if lossComponents["boundaries"]:
			imgGrad = grad0

		if lossComponents["asymmetricSmooth"]:
			smoothMasked = asymmetricSmoothLoss(flow,gt,instanceParams,None,validPixelMask,imgGrad,boundaryAlpha, backward,GOL=GOL)

		# apply masking
		photoMasked = photo * occInvalidMask
		# gradMasked = grad * occInvalidMask

		# average spatially
		photoAvg = tf.reduce_mean(photoMasked,reduction_indices=[1,2])
		smoothAvg = tf.reduce_mean(smoothMasked,reduction_indices=[1,2])

		# summaries ----------------------------
		photoLossName = "photoLossB" if backward else "photoLossF"
		smoothLossName = "smoothLossB" if backward else "smoothLossF"
		if scale == 1:
			tf.summary.scalar(photoLossName,tf.reduce_mean(photoAvg))
			# tf.summary.scalar(smoothLossName,tf.reduce_mean(smoothAvg))
		smoothAvg = smoothAvg*smoothReg

		return photoAvg, smoothAvg


def occluMask(flowF, flowB, alpha1=0.01, alpha2=0.5, backward=False):
    flowBWarp = flowWarp(flowB, flowF)
    lhs = tf.reduce_sum(tf.square(tf.abs(flowF + flowBWarp)), axis=-1, keepdims=True)
    rhs = alpha1 * (
        tf.reduce_sum(tf.square(tf.abs(flowF)), axis=-1, keepdims=True) + tf.reduce_sum(
            tf.square(tf.abs(flowBWarp)), axis=-1, keepdims=True)) + alpha2

    ret = tf.cast(tf.less_equal(lhs, rhs), tf.float32)
    return 1.0 - ret



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
