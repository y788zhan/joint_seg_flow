import tensorflow as tf
from components import *
from lossComponents import *

class TrainingLoss:
	def __init__(self,instanceParams,networkF,networkB,trainingData):
		with tf.variable_scope(None,default_name="TrainingLoss"):
			# hyperparams
			weightDecay = instanceParams["weightDecay"]
			lossComponents = instanceParams["lossComponents"]

			# helpers
			predFlowF = networkF.flows[0]
			predFlowB = networkB.flows[0]
			frame0 = trainingData.frame0
			frame1 = trainingData.frame1
			vpm = trainingData.validMask

			# unsup loss
			pLossF, sLossF = unsupFlowLoss(predFlowF,predFlowB,frame0,frame1,vpm,instanceParams)
			recLossF = pLossF + sLossF
			if lossComponents["backward"]:
				pLossB, sLossB = unsupFlowLoss(predFlowB,predFlowF,frame1,frame0,vpm,instanceParams, backward=True)
				recLossB = pLossB + sLossB
			# weight decay
			with tf.variable_scope(None,default_name="weightDecay"):
				weightLoss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection("weights")]))*weightDecay

			# final loss, schedule backward unsup loss
			recLossBWeight = tf.placeholder(tf.float32,shape=[]) #  [0,0.5]
			self.recLossBWeight = recLossBWeight # used to set weight at runtime
			if lossComponents["backward"]:
				totalLoss = \
					recLossF*(1.0 - recLossBWeight) + \
					recLossB*recLossBWeight + \
					weightLoss
				tf.summary.scalar("recLossF",tf.reduce_mean(recLossF*(1.0-recLossBWeight)))
				tf.summary.scalar("recLossB",tf.reduce_mean(recLossB*recLossBWeight))
			else:
				totalLoss = recLossF + weightLoss
				tf.summary.scalar("recLossF",tf.reduce_mean(recLossF))

			tf.summary.scalar("weightDecay",tf.reduce_mean(weightLoss))
			tf.summary.scalar("totalLoss",tf.reduce_mean(totalLoss))

			pGrad = tf.gradients(pLossF, predFlowF)[0]
			sGrad = tf.gradients(sLossF, predFlowF)[0]
			tf.summary.image("photo_gradients", flowToRgb(pGrad, 'saturation'))
			tf.summary.image("smooth_gradients", flowToRgb(sGrad, 'saturation'))
			tf.summary.scalar("mean_photo_grad", tf.reduce_mean(tf.abs(pGrad)))
			tf.summary.scalar("mean_smooth_grad", tf.reduce_mean(tf.abs(sGrad)))
			self.loss = totalLoss
