import tensorflow as tf
import tensorflow_addons as tfa

class GLU(tf.keras.layers.Layer):
    '''
        I imitated Torch's implementation. Halve the number of channels.
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
    '''
    def __init__(self, axis = -1, **kwargs):
        self.axis = axis
        super(GLU, self).__init__(**kwargs)

    def call(self, inputs): 
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return tf.multiply(a, tf.keras.activations.sigmoid(b))


class ConditionalInstanceNormalization(tfa.layers.GroupNormalization):
    def __init__(self, n_class, **kwargs):
        self.n_class = n_class
        kwargs["groups"] = -1
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with a floating-point '
                      f'dtype. Received: dtype={dtype}')
        n_channels = input_shape[-1]

        self.gamma_weight = self.add_weight("gamma", shape=(self.n_class, n_channels), initializer="ones", dtype=self.dtype)
        self.beta_weight = self.add_weight("beta", shape=(self.n_class, n_channels), initializer="zeros", dtype=self.dtype, trainable=True)
        
        self.broadcast_shape = [1] * len(input_shape)
        self.broadcast_shape[0] = -1
        self.broadcast_shape[-1] = n_channels

        self._set_norm_axis(input_shape)

        self._set_number_of_groups_for_instance_norm(input_shape)
        self._create_input_spec(input_shape)
        self.built = True
        super(tfa.layers.GroupNormalization, self).build(input_shape)

    def _set_norm_axis(self, input_shape):
        dim = len(input_shape)
        if dim == 4:
            # NHWC
            self.instance_norm_axes = [1, 2]
        elif dim == 3:
            # NHC
            self.instance_norm_axes = [1]
        else:
            raise ValueError(
                "Input shape: " + str(input_shape) + " is not supported."
                "Enter a 2D input (NHC) or 3D input (NHWC) Tensor."
            )

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_class': self.n_class,
        })
        return config

    def call(self, inputs, condition):
        mean, variance = tf.nn.moments(inputs, self.instance_norm_axes, keepdims=True)

        gamma = tf.matmul(tf.cast(condition, self.compute_dtype), tf.cast(self.gamma_weight, self.compute_dtype))
        gamma = tf.reshape(gamma, self.broadcast_shape)

        beta = tf.matmul(tf.cast(condition, self.compute_dtype), tf.cast(self.beta_weight, self.compute_dtype))
        beta = tf.reshape(beta, self.broadcast_shape)

        normalized_inputs = tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )

        return normalized_inputs

class AdaptiveInstanceNormalization(tfa.layers.GroupNormalization):
    # どう組み込む？
    def __init__(self, **kwargs):
        kwargs["groups"] = -1
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with a floating-point '
                      f'dtype. Received: dtype={dtype}')
        n_channels = input_shape[-1]
        
        self.broadcast_shape = [1] * len(input_shape)
        self.broadcast_shape[0] = -1
        self.broadcast_shape[-1] = n_channels

        self._set_norm_axis(input_shape)

        self._set_number_of_groups_for_instance_norm(input_shape)
        self._create_input_spec(input_shape)
        self.built = True
        super(tfa.layers.GroupNormalization, self).build(input_shape)
        
    def _set_norm_axis(self, input_shape):
        dim = len(input_shape)
        if dim == 4:
            # NHWC
            self.instance_norm_axes = [1, 2]
        elif dim == 3:
            # NHC
            self.instance_norm_axes = [1]
        else:
            raise ValueError(
                "Input shape: " + str(input_shape) + " is not supported."
                "Enter a 2D input (NHC) or 3D input (NHWC) Tensor."
            )

    def call(self, inputs, condition):
        x_mean, x_var = tf.nn.moments(inputs, axes=self.instance_norm_axes, keep_dims=True)
        y_mean, y_var = tf.nn.moments(condition, axes=self.instance_norm_axes, keep_dims=True)
        y_std = tf.sqrt(y_var + self.epsilon)
        
        normalized_inputs = tf.nn.batch_normalization(
            inputs,
            mean=x_mean,
            variance=x_var,
            scale=y_std,
            offset=y_mean,
            variance_epsilon=self.epsilon,
        )

        return normalized_inputs

    
class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, r = 2, **kwargs):
        self.r = r
        super().__init__(**kwargs)

    def call(self, inputs): 
        return tf.nn.depth_to_space(inputs, block_size=self.r)


class GlobalSumPooling2D(tf.keras.layers.Layer):
    def __init__(self, axis=[1, 2], keepdims=True, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)

class InnerProduct(tf.keras.layers.Layer):
    def call(self, a, b):
        return tf.matmul(a, b)

class HightMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis = 1, keepdims=True)
