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

	                pLossTotalF = sLossTotalF = segLossTotalF = 0
            	        pLossTotalB = sLossTotalB = segLossTotalB = 0
	
        	        #rel_weights = [1, 0.32, 0.3, 0.28, 0.08]
			rel_weights = [1] * 5
			# unsup loss
			levels = len(networkF.flows) if instanceParams["multiscale"] else 1
            		for i in range(levels):
                		predFlowF = networkF.flows[i]
                		predFlowB = networkB.flows[i]

                		if i > 0:
                    			predFlowF *= 20 / (2 ** i)
                    			predFlowB *= 20 / (2 ** i)

    				pLossF, sLossF, segLossF = unsupFlowLoss(predFlowF,predFlowB,frame0,frame1,vpm,instanceParams)
                		pLossF *= rel_weights[i]
                		sLossF *= rel_weights[i]
                		segLossF *= rel_weights[i]

                		pLossTotalF += pLossF
                		sLossTotalF += sLossF
                		segLossTotalF += segLossF

    				if lossComponents["backward"]:
    					pLossB, sLossB, segLossB = unsupFlowLoss(predFlowB,predFlowF,frame1,frame0,vpm,instanceParams, backward=True)
                    			pLossB *= rel_weights[i]
					sLossB *= rel_weights[i]
                    			segLossB *= rel_weights[i]

                    			pLossTotalB += pLossB
                    			sLossTotalB += sLossB
                    			segLossTotalB += segLossB


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

			pGrad = tf.gradients(pLossF, predFlowF)[0]
			sGrad = tf.gradients(sLossF, predFlowF)[0]
			segGrad = tf.gradients(segLossF, predFlowF)[0]
			# tf.summary.image("photo_gradients", flowToRgb1(pGrad, 'saturation'))
			# tf.summary.image("smooth_gradients", flowToRgb1(sGrad, 'saturation'))
			# tf.summary.image("seg_gradients", flowToRgb1(segGrad, "saturation"))
			# tf.summary.scalar("mean_photo_grad", tf.reduce_mean(tf.abs(pGrad)))
			# tf.summary.scalar("mean_smooth_grad", tf.reduce_mean(tf.abs(sGrad)))
			# tf.summary.scalar("mean_seg_grad", tf.reduce_mean(tf.abs(segGrad)))
			self.loss = totalLoss
