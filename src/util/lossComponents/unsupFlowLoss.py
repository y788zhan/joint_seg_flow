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

		# helpers
        size = [flow.shape[1], flow.shape[2]]
		rgb0 = tf.images.bilinear_resize(frame0["rgbNorm"], size)
		rgb1 = tf.images.bilinear_resize(frame1["rgbNorm"], size)
		grad0 = tf.images.bilinear_resize(frame0["grad"], size)
		grad1 = tf.images.bilinear_resize(frame1["grad"], size)
		gt = tf.images.bilinear_resize(frame0["gt"], size)
		gt1 = tf.images.bilinear_resize(frame1["gt"], size)
		# if not backward:
		# 	tf.summary.image("rgb0", rgb0)
		# 	tf.summary.image("rgb1", rgb1)
		# 	tf.summary.image("gt", gt)
		# masking from simple occlusion/border
		#occMask = borderOcclusionMask(flow) # occ if goes off image
		occInvalidMask = 1#validPixelMask*occMask # occluded and invalid
		
		# loss components
		photo = photoLoss(flow,rgb0,rgb1,photoAlpha,photoBeta)

		flowF = flow * 1.0
		flowBCopy = flowB * 1.0
		flowF = tf.stop_gradient(flowF)
		flowBCopy = tf.stop_gradient(flowBCopy)
		occMask = occluMask(flowF, flowBCopy, backward=backward)
		photo = photo * (1.0 - occMask)
		tf.summary.image("occlusion_mask", occMask)

		seg = photoLoss(flow, gt, gt1, 1, 1)
		#seg = seg * 0
		seg = seg * 1e4 * (1.0 - occMask)

		# grad = gradLoss(flow,grad0,grad1,gradAlpha,gradBeta)
		imgGrad = None
		if lossComponents["boundaries"]:
			imgGrad = grad0

		if lossComponents["asymmetricSmooth"]:
			smoothMasked, gtMask = asymmetricSmoothLoss(flow,gt,instanceParams,None,validPixelMask,imgGrad,boundaryAlpha, backward)
		else:
			smoothMasked = smoothLoss(flow,smoothAlpha,smoothBeta,validPixelMask,imgGrad,boundaryAlpha)
		# smooth2ndMasked = smoothLoss2nd(flow,smooth2ndAlpha,smooth2ndBeta,validPixelMask,imgGrad,boundaryAlpha)

		# apply masking
		photoMasked = photo * occInvalidMask * gtMask
		# gradMasked = grad * occInvalidMask

		# average spatially
		photoAvg = tf.reduce_mean(photoMasked,reduction_indices=[1,2])
		# gradAvg = tf.reduce_mean(gradMasked,reduction_indices=[1,2])
		smoothAvg = tf.reduce_mean(smoothMasked,reduction_indices=[1,2])
		# smooth2ndAvg = tf.reduce_mean(smooth2ndMasked,reduction_indices=[1,2])
		segAvg = tf.reduce_mean(seg, reduction_indices=[1,2])

		# weight loss terms
		# gradAvg = gradAvg*gradReg
		# smooth2ndAvg = smooth2ndAvg*smooth2ndReg

		# summaries ----------------------------
		photoLossName = "photoLossB" if backward else "photoLossF"
		smoothLossName = "smoothLossB" if backward else "smoothLossF"
		tf.summary.scalar(photoLossName,tf.reduce_mean(photoAvg))
		tf.summary.scalar("segLossF", tf.reduce_mean(segAvg))
		# tf.summary.scalar(smoothLossName,tf.reduce_mean(smoothAvg))
		smoothAvg = smoothAvg*smoothReg
		# final loss
		# finalLoss = photoAvg + smoothAvg
		return photoAvg, smoothAvg, segAvg


def occluMask(flowF, flowB, alpha1=0.01, alpha2=0.1, backward=False):
    flowBWarp = flowWarp(flowB, flowF)
    if not backward:
        tf.summary.image("warped_backwards_flow", flowToRgb1(flowBWarp))
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
