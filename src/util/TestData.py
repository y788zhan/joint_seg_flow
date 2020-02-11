import tensorflow as tf
from components import *

class TestData:
    """
    No queuing, pass through for processing from files
    """
    def __init__(self,filenames0,filenames1,gtnames0,batchSize,desiredHeight,desiredWidth):
        with tf.variable_scope(None,default_name="ImagePairData"):
            self.im0File = tf.placeholder(tf.string,[])
            self.im1File = tf.placeholder(tf.string,[])
            self.gt0File = tf.placeholder(tf.string,[])

            #read data
            rawData = tf.read_file(self.im0File)
            imData0 = tf.image.decode_png(rawData, channels=3)

            rawData = tf.read_file(self.im1File)
            imData1 = tf.image.decode_png(rawData, channels=3)

            rawData = tf.read_file(self.gt0File)
            gtData0 = tf.image.decode_png(rawData, channels=1)
            #convert to floats, and divide
            mean = [0.407871, 0.457525, 0.481094]
            mean = tf.expand_dims(tf.expand_dims(mean,0),0)
            imData0 = tf.cast(imData0,tf.float32)/255 - mean
            imData1 = tf.cast(imData1,tf.float32)/255 - mean
            gtData0 = tf.cast(gtData0,tf.float32)/255

            #fix image size, be careful here!
            imData0.set_shape([desiredHeight,desiredWidth,3])
            imData1.set_shape([desiredHeight,desiredWidth,3])
            gtData0.set_shape([desiredHeight,desiredWidth,1])

            #place data into batches
            imData0 = tf.expand_dims(imData0,0)
            imData1 = tf.expand_dims(imData1,0)
            gtData0 = tf.expand_dims(gtData0,0)

            self.frame0 = {
                "rgb": imData0,
                "gt": gtData0
            }

            self.frame1 = {
                "rgb": imData1
            }

            self.height = tf.cast(desiredHeight,tf.int32)
            self.width = tf.cast(desiredWidth,tf.int32)

