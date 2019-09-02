import tensorflow as tf
from components import *
from lossComponents import *

class TrainingLoss:
	def __init__(self,instanceParams,networkF,networkB,trainingData,fhat):
		with tf.variable_scope(None,default_name="TrainingLoss"):
			# hyperparams
			weightDecay = instanceParams["weightDecay"]
			lossComponents = instanceParams["lossComponents"]

			# helpers
			frame0 = trainingData.frame0
			frame1 = trainingData.frame1
			vpm = trainingData.validMask

	                pLossTotalF = sLossTotalF = segLossTotalF = 0
            	        pLossTotalB = sLossTotalB = segLossTotalB = 0
	
                	predFlowF = networkF.flows[0]
                	predFlowB = networkB.flows[0]

			fhat_copy = fhat * 1.0
			fhat_copy = tf.stop_gradient(fhat_copy)

			gol = tf.placeholder(tf.float32, shape = fhat.get_shape().as_list())
			self.gol = gol

    			sLossF = unsupFlowLoss(predFlowF,predFlowB,fhat_copy,frame0,frame1,vpm,instanceParams,self.gol)
                	sLossTotalF += sLossF

            		recLossF = pLossTotalF + sLossTotalF + segLossTotalF
            		recLossB = pLossTotalB + sLossTotalB + segLossTotalB
			# final loss, schedule backward unsup loss
			recLossBWeight = 0.5  #  [0,0.5]
			self.recLossBWeight = recLossBWeight # used to set weight at runtime
			if lossComponents["backward"]:
				totalLoss = \
					recLossF*(1.0 - recLossBWeight) + \
					recLossB*recLossBWeight
				tf.summary.scalar("recLossF",tf.reduce_mean(recLossF*(1.0-recLossBWeight)))
				tf.summary.scalar("recLossB",tf.reduce_mean(recLossB*recLossBWeight))
			else:
				totalLoss = recLossF
				tf.summary.scalar("recLossF",tf.reduce_mean(recLossF))

			# pGrad = tf.gradients(pLossTotalF, networkF.flows[0])[0]
			# sGrad = tf.gradients(sLossTotalF, networkF.flows[0])[0]
			# segGrad = tf.gradients(segLossTotalF, predFlowF)[0]
			# tf.summary.image("photo_gradients", flowToRgb1(pGrad, 'saturation'))
			# tf.summary.image("smooth_gradients", flowToRgb1(sGrad, 'saturation'))
			# tf.summary.image("seg_gradients", flowToRgb1(segGrad, "saturation"))
			# tf.summary.scalar("mean_photo_grad", tf.reduce_mean(tf.abs(pGrad)))
			# tf.summary.scalar("mean_smooth_grad", tf.reduce_mean(tf.abs(sGrad)))
			# tf.summary.scalar("mean_seg_grad", tf.reduce_mean(tf.abs(segGrad)))
			self.loss = totalLoss
