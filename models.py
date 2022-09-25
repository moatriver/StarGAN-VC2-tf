import tensorflow as tf
import tensorflow_addons as tfa
import modules

def build_generator(mcep_size, t_length, code_size):

    inputs = tf.keras.Input(shape = (mcep_size, t_length, 1), name="melspec")
    codes = tf.keras.Input(shape = (code_size), name="codes")

    # first block
    x = tf.keras.layers.Conv2D(128, kernel_size = (5, 15), strides = (1, 1), padding="same")(inputs)
    x_first = modules.GLU()(x)

    # Downsample (2D)
    x = tf.keras.layers.Conv2D(256, kernel_size = (5, 5), strides = (2, 2), padding="same")(x_first)
    x = tfa.layers.InstanceNormalization()(x)
    x_ds = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(512, kernel_size = (5, 5), strides = (2, 2), padding="same")(x_ds)
    x = tfa.layers.InstanceNormalization()(x)
    x_ds = modules.GLU()(x)


    # 2D -> 1D
    x = tf.keras.layers.Reshape((t_length//4, 2304))(x_ds)
    x = tf.keras.layers.Conv1D(256, kernel_size = 1, strides = 1, padding="same")(x)
    x_1d = tfa.layers.InstanceNormalization()(x)

    # 9 1D blocks
    for _ in range(9):
        x = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")(x_1d)
        x = modules.ConditionalInstanceNormalization(code_size)(x, codes)
        x_1d = modules.GLU()(x)

    # 1D -> 2D
    x = tf.keras.layers.Conv1D(2304, kernel_size = 1, strides = 1, padding="same")(x_1d)
    x_2d = tf.keras.layers.Reshape((9, t_length//4, 256))(x)

    # Upsample (2D)

    x = tf.keras.layers.Conv2D(512, kernel_size = (5, 5), strides = (1, 1), padding="same")(x_2d)
    x = modules.PixelShuffler(r = 2)(x)
    x_2d = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(256, kernel_size = (5, 5), strides = (1, 1), padding="same")(x_2d)
    x = modules.PixelShuffler(r = 2)(x)
    x_2d = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(35, kernel_size = (5, 15), strides = (1, 1), padding="same")(x_2d)
    x = modules.HightMean()(x)
    outputs = tf.keras.layers.Reshape((35, t_length, -1))(x)
    outputs = tf.keras.layers.Activation("linear", dtype="float32")(outputs)

    return tf.keras.Model(inputs=[inputs, codes], outputs=outputs)

class Generator(tf.keras.Model):
    def __init__(self, code_size):
        super().__init__()
        # first block
        self.first_conv = tf.keras.layers.Conv2D(128, kernel_size = (5, 15), strides = (1, 1), padding="same")
        self.first_GLU = modules.GLU()

        # Downsample (2D)
        self.ds_l1_conv = tf.keras.layers.Conv2D(256, kernel_size = (5, 5), strides = (2, 2), padding="same")
        self.ds_l1_IN = tfa.layers.InstanceNormalization()
        self.ds_l1_GLU = modules.GLU()

        self.ds_l2_conv = tf.keras.layers.Conv2D(512, kernel_size = (5, 5), strides = (2, 2), padding="same")
        self.ds_l2_IN = tfa.layers.InstanceNormalization()
        self.ds_l2_GLU = modules.GLU()

        # 2D -> 1D
        self.to1d_conv = tf.keras.layers.Conv1D(256, kernel_size = 1, strides = 1, padding="same")
        self.to1d_IN = tfa.layers.InstanceNormalization()

        # 9 1D blocks
        self.block_1d_1 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_1 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_1 = modules.GLU()

        self.block_1d_2 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_2 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_2 = modules.GLU()

        self.block_1d_3 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_3 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_3 = modules.GLU()

        self.block_1d_4 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_4 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_4 = modules.GLU()

        self.block_1d_5 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_5 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_5 = modules.GLU()

        self.block_1d_6 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_6 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_6 = modules.GLU()

        self.block_1d_7 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_7 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_7 = modules.GLU()

        self.block_1d_8 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_8 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_8 = modules.GLU()

        self.block_1d_9 = tf.keras.layers.Conv1D(512, kernel_size = 5, strides = 1, padding="same")
        self.block_CIN_9 = modules.ConditionalInstanceNormalization(code_size)
        self.block_GLU_9 = modules.GLU()

        # 1D -> 2D
        self.to2d_conv = tf.keras.layers.Conv1D(2304, kernel_size = 1, strides = 1, padding="same")

        # Upsample (2D)
        self.us_l1_conv = tf.keras.layers.Conv2D(512, kernel_size = (5, 5), strides = (1, 1), padding="same")
        self.us_l1_ps = modules.PixelShuffler(r = 2)
        self.us_l1_GLU = modules.GLU()

        self.us_l2_conv = tf.keras.layers.Conv2D(256, kernel_size = (5, 5), strides = (1, 1), padding="same")
        self.us_l2_ps = modules.PixelShuffler(r = 2)
        self.us_l2_GLU = modules.GLU()   

        # last block
        # self.last_conv = tf.keras.layers.Conv2D(36, kernel_size = (5, 15), strides = (1, 1), padding="same")
        # self.last_HM = modules.HightMean()
        self.output_conv = tf.keras.layers.Conv2D(1, kernel_size = 7, strides = (1, 1), padding="same")
        self.last_linear = tf.keras.layers.Activation("linear", dtype="float32")

    
    def call(self, inputs):
        mceps, codes = inputs
        t_length = tf.keras.backend.int_shape(mceps)[2]

        # first block
        x = self.first_conv(mceps)
        x = self.first_GLU(x)

        # Downsample (2D)
        x = self.ds_l1_conv(x)
        x = self.ds_l1_IN(x)
        x = self.ds_l1_GLU(x)

        x = self.ds_l2_conv(x)
        x = self.ds_l2_IN(x)
        x = self.ds_l2_GLU(x)

        # 2D -> 1D
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # NHWC -> NCHW
        x = tf.reshape(x, (-1, 2304, 1, t_length//4))
        x = tf.squeeze(x, [2])
        x = tf.transpose(x, perm=[0, 2, 1]) # NCH -> NHC
        x = self.to1d_conv(x)
        x = self.to1d_IN(x)

        # 9 1D blocks
        x = self.block_1d_1(x)
        x = self.block_CIN_1(x, codes)
        x = self.block_GLU_1(x)

        x = self.block_1d_2(x)
        x = self.block_CIN_2(x, codes)
        x = self.block_GLU_2(x)

        x = self.block_1d_3(x)
        x = self.block_CIN_3(x, codes)
        x = self.block_GLU_3(x)

        x = self.block_1d_4(x)
        x = self.block_CIN_4(x, codes)
        x = self.block_GLU_4(x)

        x = self.block_1d_5(x)
        x = self.block_CIN_5(x, codes)
        x = self.block_GLU_5(x)

        x = self.block_1d_6(x)
        x = self.block_CIN_6(x, codes)
        x = self.block_GLU_6(x)

        x = self.block_1d_7(x)
        x = self.block_CIN_7(x, codes)
        x = self.block_GLU_7(x)

        x = self.block_1d_8(x)
        x = self.block_CIN_8(x, codes)
        x = self.block_GLU_8(x)

        x = self.block_1d_9(x)
        x = self.block_CIN_9(x, codes)
        x = self.block_GLU_9(x)

        # 1D -> 2D
        x = self.to2d_conv(x)
        x = tf.transpose(x, perm=[0, 2, 1]) # NHC -> NCH
        x = tf.reshape(x, (-1, 256, 9, t_length//4)) 
        x = tf.transpose(x, perm=[0, 2, 3, 1]) # NCHW -> NHWC

        # Upsample (2D)
        x = self.us_l1_conv(x)
        x = self.us_l1_ps(x)
        x = self.us_l1_GLU(x)

        x = self.us_l2_conv(x)
        x = self.us_l2_ps(x)
        x = self.us_l2_GLU(x)

        # last block
        # x = self.last_conv(x)
        # x = self.last_HM(x)

        # x = tf.reshape(x, (-1, 35, t_length, 1))
        # x = tf.transpose(x, perm=[0, 3, 2, 1])
        x = self.output_conv(x)

        x = self.last_linear(x)

        return x

def build_discriminator(mcep_size, crop_size, code_size):

    inputs = tf.keras.Input(shape = (mcep_size, crop_size, 1), name="melspec")
    codes = tf.keras.Input(shape = (code_size), name="codes")

    x = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding="same")(inputs)
    x = modules.GLU()(x)

    # Downsample (2D)
    x = tf.keras.layers.Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(1024, kernel_size = (3, 3), strides = (2, 2), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = modules.GLU()(x)

    x = tf.keras.layers.Conv2D(1024, kernel_size = (1, 5), strides = (1, 1), padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = modules.GLU()(x)

    fc = tf.keras.layers.Flatten()(x)
    fc = tf.keras.layers.Dense(1, name="FC")(fc)

    gsp = modules.GlobalSumPooling2D(keepdims=False)(x)
    embed = tf.keras.layers.Dense(512, name="enbed")(codes)
    projection = tf.keras.layers.Dot(axes=1, name="inner_product")([gsp, embed])

    outputs = tf.keras.layers.Add(dtype="float32", name="projection")([fc, projection])

    # outputs = tf.keras.layers.Activation("sigmoid", dtype="float32")(outputs)

    return tf.keras.Model(inputs=[inputs, codes], outputs=outputs)
   
if __name__ == "__main__":
    generator = Generator(8)
    discriminator = build_discriminator(35, 128, 8*2)
    mcep = tf.random.uniform((4, 35, 444, 1))
    code = tf.random.uniform((4, 8))
    gen_mcep = generator((mcep, code))
    print(tf.keras.backend.int_shape(gen_mcep))

    mcep = tf.random.uniform((4, 35, 128, 1))
    code_2 = tf.random.uniform((4, 8))
    env_code = tf.concat((code, code_2), 1)
    y_real = discriminator((mcep, env_code), training=False)
    print(tf.keras.backend.int_shape(y_real))

    print(generator.summary())