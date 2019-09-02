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
			frame0 = trainingData.frame0
			frame1 = trainingData.frame1
			vpm = trainingData.validMask

			pLossTotalF = sLossTotalF = 0
			pLossTotalB = sLossTotalB = 0
	
			rel_weights = [1, 0.32, 0.3, 0.28, 0.08]

			# unsup loss
			# networkF.flows should contain multiscale outputs
			# turn on multiscale by iterating over these outputs
			for i in range(1): # range(len(networkF.flows)):
				predFlowF = networkF.flows[i]
				predFlowB = networkB.flows[i]

				if i > 0:
					predFlowF *= 20 / (2 ** i)
					predFlowB *= 20 / (2 ** i)

				pLossF, sLossF = unsupFlowLoss(predFlowF,predFlowB,frame0,frame1,vpm,instanceParams)
				pLossF *= rel_weights[i]
				sLossF *= rel_weights[i]

				pLossTotalF += pLossF
				sLossTotalF += sLossF

				if lossComponents["backward"]:
					pLossB, sLossB = unsupFlowLoss(predFlowB,predFlowF,frame1,frame0,vpm,instanceParams, backward=True)
					pLossB *= rel_weights[i]
					sLossB *= rel_weights[i]

					pLossTotalB += pLossB
					sLossTotalB += sLossB

			recLossF = pLossTotalF + sLossTotalF
			recLossB = pLossTotalB + sLossTotalB
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

			pGrad = tf.gradients(pLossF, predFlowF)[0]
			sGrad = tf.gradients(sLossF, predFlowF)[0]
			# tf.summary.image("photo_gradients", flowToRgb1(pGrad, 'saturation'))
			# tf.summary.image("smooth_gradients", flowToRgb1(sGrad, 'saturation'))
			# tf.summary.scalar("mean_photo_grad", tf.reduce_mean(tf.abs(pGrad)))
			# tf.summary.scalar("mean_smooth_grad", tf.reduce_mean(tf.abs(sGrad)))
			self.loss = totalLoss
