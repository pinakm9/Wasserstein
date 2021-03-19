import tensorflow as tf
import numpy as np

class W1L1TestFn(tf.keras.Model):
    """
        Description: 
            defines a test function for computing Wasserstein_1 with L1 distance 

        Args:
            num_nodes: number of nodes on each layer
            num_layers: number of layers
            activation: activation function for all layer
            clip_value: absolute value threshold at which to clip each weight 
            dtype: tf float type to be used
    """
    def __init__(self, num_nodes=50, num_layers=4, activation=tf.keras.activations.tanh, clip_value=0.01, dtype=tf.float64):
        super().__init__(dtype=dtype)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.clip_value = clip_value
        self.ls = []
        for layer_id in range(num_layers - 1):
             self.ls.append(tf.keras.layers.Dense(num_nodes, dtype=dtype, name='layer_{}'.format(layer_id)))#,\#kernel_constraint=tf.keras.constraints.MaxNorm(clip_value, axis=1)))
        self.final_layer = tf.keras.layers.Dense(1, dtype=dtype, name='layer_{}'.format(num_layers - 1))#,\#kernel_constraint=tf.keras.constraints.MaxNorm(clip_value, axis=1))


    def call(self, x):
        """
            Description: the required call function for a subclassed tensorflow model

            Args:
                x: input to the neural network

            Returns:
                output(1D) of the neural network  
        """
        for layer_id in range(self.num_layers - 1):
            x = self.activation(self.ls[layer_id](x))
        return self.final_layer(x)

class W1L1Calc(object):
    """
    Description: Wasserstein_1 calculator with L1 distance

    Args:
        sampler_1: sampler for the first distribution 
        sampler_2: sampler for the second distribution
        **test_fn_params: keyword arguments for creating the test function network
    """
    def __init__(self, sampler_1, sampler_2, **test_fn):
        self.sampler_1 = sampler_1
        self.sampler_2 = sampler_2
        self.test_fn = W1L1TestFn(**test_fn)

    def calculate(self, initial_rate=1e-3, epochs=100, batch_size=500):
        """
        Description: Calculates Wasserstein_1 for L1 distance

        Args:
            initial_rate: initial learning rate for ADAM optimizer
            epochs: number of epochs to train the test function
            batch_size: number of samples to be drawn from each distribution during an epoch

        Returns:
            the final approximate value of Wasserstein_1 for L1 distance
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_rate)#,  clipnorm=1.0)
        for epoch in range(epochs):
            s_1 = self.sampler_1(batch_size)
            s_2 = self.sampler_2(batch_size)
            with tf.GradientTape() as tape:
                e_1 = tf.reduce_mean(self.test_fn(s_1))
                e_2 = tf.reduce_mean(self.test_fn(s_2))
                loss = - tf.math.square(e_1 - e_2)
                print('epoch = {}, EMD = {}'.format(epoch + 1, tf.sqrt(-loss)))
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    break
                grads = tape.gradient(loss, self.test_fn.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.test_fn.trainable_weights))
                for l in self.test_fn.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.test_fn.clip_value, self.test_fn.clip_value) for w in weights]
                    l.set_weights(weights)
        return tf.sqrt(-loss)