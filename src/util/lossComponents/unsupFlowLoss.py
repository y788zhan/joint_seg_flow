import tensorflow as tf
from components import *
from photoLoss import *
from gradLoss import *
from smoothLoss import *
from smoothLoss2nd import *
from asymmetricSmoothLoss import *
from epeLoss import *
def unsupFlowLoss(flow,flowB,fhat,frame0,frame1,validPixelMask,instanceParams, gol, backward=False):
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

		tf.summary.scalar("gol", tf.reduce_mean(gol))

		rgb0 = frame0["rgbNorm"]
		rgb1 = frame1["rgbNorm"]
		gt = frame0["gt"]

		occInvalidMask = 1#validPixelMask*occMask # occluded and invalid

		photo = photoLoss(flow, rgb0, rgb1, photoAlpha, photoBeta)
		tf.summary.scalar("flow_ploss", tf.reduce_mean(photo))
		imgGrad = None
		if lossComponents["boundaries"]:
			imgGrad = grad0

		smoothMasked = asymmetricSmoothLoss(flow,gt,instanceParams,None,validPixelMask,imgGrad,boundaryAlpha, backward)
		clamp = clampLoss(flow, fhat) * gol
		clamp = tf.reduce_mean(clamp, axis=-1, keepdims = True)

		# assign lossAvg to this to train epe against HS solution
		# epe = epeLoss(flow)
		# epe = 1.0e4 * tf.reduce_mean(epe, reduction_indices=[1,2])

		lossAvg = tf.reduce_mean(smoothMasked + clamp,reduction_indices=[1,2])
		lossAvg = lossAvg*smoothReg

		cgrad = tf.gradients(clamp, flow)
		tf.summary.scalar("clamp_gradient", smoothReg * tf.reduce_mean(tf.abs(cgrad[0])))
		sgrad = tf.gradients(smoothMasked, flow)
		tf.summary.scalar("smooth_gradient", smoothReg * tf.reduce_mean(tf.abs(sgrad[0])))

		return lossAvg


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
