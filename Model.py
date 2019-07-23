    def setupCNN(self, cnnIn3d):
        "create CNN layers and return output of these layers"
        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 3, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer
        for i in range(numLayers):
            print(tf.shape(pool))
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            self.variable_summaries(conv)
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)

            kernel2 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i + 1], featureVals[i + 1]], stddev=0.1))
            conv2 = tf.nn.conv2d(relu, kernel2, padding='SAME', strides=(1, 1, 1, 1))
            self.variable_summaries(conv2)
            conv_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train)
            relu2 = tf.nn.relu(conv_norm2)

            kernel3 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i + 1], featureVals[i + 1]], stddev=0.1))
            conv3 = tf.nn.conv2d(relu2, kernel3, padding='SAME', strides=(1, 1, 1, 1))
            # self.variable_summaries(conv2)
            conv_norm3 = tf.layers.batch_normalization(conv3, training=self.is_train)
            relu3 = tf.nn.relu(conv_norm3)

            pool = tf.nn.max_pool(relu3, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        return pool
