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
		rgb0 = frame0["rgbNorm"]
		rgb1 = frame1["rgbNorm"]
		grad0 = frame0["grad"]
		grad1 = frame1["grad"]
		gt = frame0["gt"]
		if not backward:
			tf.summary.image("rgb0", frame0["rgb"])
			tf.summary.image("rgb1", frame1["rgb"])
			tf.summary.image("gt", gt)
		# masking from simple occlusion/border
		occMask = borderOcclusionMask(flow) # occ if goes off image
		occInvalidMask = validPixelMask*occMask # occluded and invalid
		gt = gt / 255.0
		
		# loss components
		photo = photoLoss(flow,rgb0,rgb1,photoAlpha,photoBeta)
		# grad = gradLoss(flow,grad0,grad1,gradAlpha,gradBeta)
		imgGrad = None
		if lossComponents["boundaries"]:
			imgGrad = grad0

		if lossComponents["asymmetricSmooth"]:
			smoothMasked = asymmetricSmoothLoss(flow,gt,instanceParams,occMask,validPixelMask,imgGrad,boundaryAlpha, backward)
		else:
			smoothMasked = smoothLoss(flow,smoothAlpha,smoothBeta,validPixelMask,imgGrad,boundaryAlpha)
		# smooth2ndMasked = smoothLoss2nd(flow,smooth2ndAlpha,smooth2ndBeta,validPixelMask,imgGrad,boundaryAlpha)

		# apply masking
		photoMasked = photo * occInvalidMask
		# gradMasked = grad * occInvalidMask

		# average spatially
		photoAvg = tf.reduce_mean(photoMasked,reduction_indices=[1,2])
		# gradAvg = tf.reduce_mean(gradMasked,reduction_indices=[1,2])
		smoothAvg = tf.reduce_mean(smoothMasked,reduction_indices=[1,2])
		# smooth2ndAvg = tf.reduce_mean(smooth2ndMasked,reduction_indices=[1,2])

		# weight loss terms
		# gradAvg = gradAvg*gradReg
		smoothAvg = smoothAvg*smoothReg
		# smooth2ndAvg = smooth2ndAvg*smooth2ndReg

		# summaries ----------------------------
		photoLossName = "photoLossB" if backward else "photoLossF"
		smoothLossName = "smoothLossB" if backward else "smoothLossF"
		tf.summary.scalar(photoLossName,tf.reduce_mean(photoAvg))
		tf.summary.scalar(smoothLossName,tf.reduce_mean(smoothAvg))

		# final loss
		# finalLoss = photoAvg + smoothAvg
		return photoAvg, smoothAvg
		# if lossComponents["smooth2nd"]:
			# tf.summary.scalar("smooth2ndLoss",tf.reduce_mean(smooth2ndAvg))
			# finalLoss += smooth2ndAvg
		# if lossComponents["gradient"]:
			# tf.summary.scalar("gradLoss",tf.reduce_mean(gradAvg))
			# finalLoss += gradAvg
		return finalLoss
