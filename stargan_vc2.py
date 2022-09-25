import tensorflow as tf
import numpy as np
import os
import io
import json
import random
import pickle
from tqdm import tqdm

from setup_args import Args
from datasets_loader import DatasetLoader
import models

class StarGAN_VC2():
    def __init__(self, args: Args, resolver = None):
        self.args = args
        if args.set_seed:
            seed = args.seed

            os.environ["PYTONHASHSEED"] = '0'
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            os.environ["TF_DETERMINISTIC_OPS"] = '0'
            os.environ["TF_CUDNN_DETERMINISTIC"] = '0'

        if args.use_tpu:
            self.distribute_strategy = tf.distribute.TPUStrategy(resolver)
            root_dir = "gs://stargan-vc2-data"
        
        else:
            self.distribute_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"]) # only 1 gpu using. if you using 2 gpu : devices=["/gpu:0", "/gpu:1"]
            root_dir = os.path.dirname(__file__)
        
        self.checkpoint_dir = os.path.join(root_dir, "training_checkpoints", self.args.model_name, "ckpt")
        self.log_dir = os.path.join(root_dir, self.args.tensorboard_log_dir, self.args.model_name, self.args.datetime)
        self.savedmodel_dir = os.path.join(root_dir, "saved_models", self.args.model_name, self.args.datetime)

        if args.mixed_precision and not args.use_tpu:
            self.train_step_func = tf.function(self.train_step_mp, jit_compile=True)
            self.test_step_func = tf.function(self.test_step, jit_compile=True)
        else:
            self.train_step_func = tf.function(self.train_step)
            self.test_step_func = tf.function(self.test_step)
        
        with self.distribute_strategy.scope():
            # self.generator = models.build_generator(args.mcep_size, args.dataset_t_length, args.code_size)
            self.generator = models.Generator(args.code_size)
            self.discriminator = models.build_discriminator(args.mcep_size, args.crop_size, args.code_size*2)

            self.bc = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
            # self.mse = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
            self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=args.g_learn_rate, beta_1 = args.momentum)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learn_rate, beta_1 = args.momentum)
            if args.mixed_precision and not args.use_tpu:
                self.g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.g_optimizer)
                self.d_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_optimizer)

            self.train_loss_g = tf.keras.metrics.Mean("training_loss_g", dtype=tf.float32)
            self.train_loss_d = tf.keras.metrics.Mean("training_loss_d", dtype=tf.float32)
            self.test_loss_g = tf.keras.metrics.Mean("test_loss_g", dtype=tf.float32)
            self.test_loss_d = tf.keras.metrics.Mean("test_loss_d", dtype=tf.float32)
            
            self.train_loss_g_cyc = tf.keras.metrics.Mean("training_loss_g_cyc", dtype=tf.float32)
            self.train_loss_g_id = tf.keras.metrics.Mean("training_loss_g_id", dtype=tf.float32)
            self.train_loss_d_real = tf.keras.metrics.Mean("training_loss_d_real", dtype=tf.float32)
            self.train_loss_d_fake = tf.keras.metrics.Mean("training_loss_d_fake", dtype=tf.float32)

            self.checkpoint = tf.train.Checkpoint(g_optimizer=self.g_optimizer, generator=self.generator, d_optimizer=self.d_optimizer, discriminator=self.discriminator)
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        if self.args.restore_bool:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("restored model...")

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

        self.datasets = DatasetLoader()
        self.train_dataset = self.distribute_strategy.experimental_distribute_dataset(self.datasets.get_train_set(args.batch_size).with_options(options))
        self.test_dataset = self.distribute_strategy.experimental_distribute_dataset(self.datasets.get_test_set(args.batch_size).with_options(options))

        # self.per_replica_batch_size = args.batch_size // self.distribute_strategy.num_replicas_in_sync
        # self.crop_shape = (self.per_replica_batch_size, args.mcep_size, args.crop_size, 1)

    def discriminator_loss(self, y_real, y_fake):
        # 1 -> real
        # loss_real = self.bc(tf.ones_like(y_real), y_real)
        # loss_fake = self.bc(tf.zeros_like(y_fake), y_fake)
        ##############################
        # this is important changes #
        #############################
        loss_real = tf.reduce_mean((1.0 - y_real) ** 2, axis=1)
        loss_fake = tf.reduce_mean((y_fake) ** 2, axis=1)

        per_example_loss_real = tf.nn.compute_average_loss(loss_real, global_batch_size=self.args.batch_size)
        per_example_loss_fake = tf.nn.compute_average_loss(loss_fake, global_batch_size=self.args.batch_size)
        return per_example_loss_real, per_example_loss_fake

    def generator_loss(self, y_fake, origin_melspecs, cycle_melspecs, identity_melspecs):
        # loss_adv = self.bc(tf.ones_like(y_fake), y_fake)
        loss_adv = tf.reduce_mean((1.0 - y_fake) ** 2, axis=1)
        per_loss_cyc = self.mse(origin_melspecs, cycle_melspecs)
        per_loss_id = self.mse(origin_melspecs, identity_melspecs)

        per_example_loss_adv = tf.nn.compute_average_loss(loss_adv, global_batch_size=self.args.batch_size)
        total_loss_cyc = tf.nn.compute_average_loss(per_loss_cyc, global_batch_size=self.args.batch_size * self.args.dataset_t_length * self.args.mcep_size)
        total_loss_id = tf.nn.compute_average_loss(per_loss_id, global_batch_size=self.args.batch_size * self.args.dataset_t_length * self.args.mcep_size)
        return per_example_loss_adv, self.args.lambda_cyc * total_loss_cyc, self.args.lambda_id * total_loss_id


    def train_step_mp(self, dataset_inputs):

        def step_fn_old(inputs):
            origin_melspecs, origin_codes, target_codes = inputs
            
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                generate_melspecs = self.generator((origin_melspecs, target_codes), training=True)
                cycle_melspecs = self.generator((generate_melspecs, origin_codes), training=True)
                identity_melspecs = self.generator((origin_melspecs, origin_codes), training=True)

                origin_melspecs_crop = tf.image.random_crop(origin_melspecs, self.crop_shape)
                generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)

                env_code = tf.concat((target_codes, origin_codes), 1)
                y_real = self.discriminator((origin_melspecs_crop, env_code), training=True)
                env_code = tf.concat((origin_codes, target_codes), 1)
                y_fake = self.discriminator((generate_melspecs_crop, env_code), training=True)

                total_adv_loss, total_loss_cyc, total_loss_id = self.generator_loss(y_fake, origin_melspecs, cycle_melspecs, identity_melspecs)
                loss_real, loss_fake = self.discriminator_loss(y_real, y_fake)
                g_loss = total_adv_loss + total_loss_cyc + total_loss_id
                d_loss = loss_real + loss_fake

                g_scaled_loss = self.g_optimizer.get_scaled_loss(g_loss)
                d_scaled_loss = self.d_optimizer.get_scaled_loss(d_loss)

            g_scaled_gradients = g_tape.gradient(g_scaled_loss, self.generator.trainable_variables)
            d_scaled_gradients = d_tape.gradient(d_scaled_loss, self.discriminator.trainable_variables)

            g_gradients = self.g_optimizer.get_unscaled_gradients(g_scaled_gradients)
            d_gradients = self.d_optimizer.get_unscaled_gradients(d_scaled_gradients)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))


            self.train_loss_g.update_state(g_loss * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d.update_state(d_loss * self.distribute_strategy.num_replicas_in_sync)

            self.train_loss_g_cyc.update_state(total_loss_cyc * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_g_id.update_state(total_loss_id * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_real.update_state(loss_real * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_fake.update_state(loss_fake * self.distribute_strategy.num_replicas_in_sync)
        

        def step_fn(inputs):
            origin_melspecs, origin_codes, target_codes = inputs

            with tf.GradientTape() as d_tape:
                generate_melspecs = self.generator((origin_melspecs, target_codes), training=False)

                # origin_melspecs_crop = tf.image.random_crop(origin_melspecs, self.crop_shape)
                # generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)

                env_code = tf.concat((target_codes, origin_codes), 1)
                # y_real = self.discriminator((origin_melspecs_crop, env_code), training=True)
                y_real = self.discriminator((origin_melspecs, env_code), training=True)
                env_code = tf.concat((origin_codes, target_codes), 1)
                # y_fake = self.discriminator((generate_melspecs_crop, env_code), training=True)
                y_fake = self.discriminator((generate_melspecs, env_code), training=True)

                loss_real, loss_fake = self.discriminator_loss(y_real, y_fake)
                d_loss = loss_real + loss_fake

                d_scaled_loss = self.d_optimizer.get_scaled_loss(d_loss)

            d_scaled_gradients = d_tape.gradient(d_scaled_loss, self.discriminator.trainable_variables)
            d_gradients = self.d_optimizer.get_unscaled_gradients(d_scaled_gradients)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))


            with tf.GradientTape() as g_tape:
                generate_melspecs = self.generator((origin_melspecs, target_codes), training=True)
                cycle_melspecs = self.generator((generate_melspecs, origin_codes), training=True)
                identity_melspecs = self.generator((origin_melspecs, origin_codes), training=True)

                # generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)
                env_code = tf.concat((origin_codes, target_codes), 1)
                # y_fake = self.discriminator((generate_melspecs_crop, env_code), training=False)
                y_fake = self.discriminator((generate_melspecs, env_code), training=False)

                total_adv_loss, total_loss_cyc, total_loss_id = self.generator_loss(y_fake, origin_melspecs, cycle_melspecs, identity_melspecs)
                g_loss = total_adv_loss + total_loss_cyc + total_loss_id

                g_scaled_loss = self.g_optimizer.get_scaled_loss(g_loss)

            g_scaled_gradients = g_tape.gradient(g_scaled_loss, self.generator.trainable_variables)
            g_gradients = self.g_optimizer.get_unscaled_gradients(g_scaled_gradients)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))


            # with tf.GradientTape() as g_tape:
            #     generate_melspecs = self.generator((origin_melspecs, target_codes), training=True)
            #     cycle_melspecs = self.generator((generate_melspecs, origin_codes), training=True)
            #     identity_melspecs = self.generator((origin_melspecs, origin_codes), training=True)
            # 
            #     # generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)
            #     env_code = tf.concat((origin_codes, target_codes), 1)
            #     # y_fake = self.discriminator((generate_melspecs_crop, env_code), training=False)
            #     y_fake = self.discriminator((generate_melspecs, env_code), training=False)
            # 
            #     total_adv_loss, total_loss_cyc, total_loss_id = self.generator_loss(y_fake, origin_melspecs, cycle_melspecs, identity_melspecs)
            #     g_loss = total_adv_loss + total_loss_cyc + total_loss_id
            # 
            #     g_scaled_loss = self.g_optimizer.get_scaled_loss(g_loss)
            # 
            # g_scaled_gradients = g_tape.gradient(g_scaled_loss, self.generator.trainable_variables)
            # g_gradients = self.g_optimizer.get_unscaled_gradients(g_scaled_gradients)
            # self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
            

            self.train_loss_g.update_state(g_loss * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d.update_state(d_loss * self.distribute_strategy.num_replicas_in_sync)

            self.train_loss_g_cyc.update_state(total_loss_cyc * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_g_id.update_state(total_loss_id * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_real.update_state(loss_real * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_fake.update_state(loss_fake * self.distribute_strategy.num_replicas_in_sync)

        self.distribute_strategy.run(step_fn, args=(dataset_inputs,))

    def train_step(self, dataset_inputs):
        def step_fn(inputs):
            origin_melspecs, origin_codes, target_codes = inputs
            
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                generate_melspecs = self.generator((origin_melspecs, target_codes), training=True)
                cycle_melspecs = self.generator((generate_melspecs, origin_codes), training=True)
                identity_melspecs = self.generator((origin_melspecs, origin_codes), training=True)

                origin_melspecs_crop = tf.image.random_crop(origin_melspecs, self.crop_shape)
                generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)

                env_code = tf.concat((target_codes, origin_codes), 1)
                y_real = self.discriminator((origin_melspecs_crop, env_code), training=True)
                env_code = tf.concat((origin_codes, target_codes), 1)
                y_fake = self.discriminator((generate_melspecs_crop, env_code), training=True)

                total_adv_loss, total_loss_cyc, total_loss_id = self.generator_loss(y_fake, origin_melspecs, cycle_melspecs, identity_melspecs)
                loss_real, loss_fake = self.discriminator_loss(y_real, y_fake)
                g_loss = total_adv_loss + total_loss_cyc + total_loss_id
                d_loss = loss_real + loss_fake

            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            self.train_loss_g.update_state(g_loss * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d.update_state(d_loss * self.distribute_strategy.num_replicas_in_sync)

            self.train_loss_g_cyc.update_state(total_loss_cyc * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_g_id.update_state(total_loss_id * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_real.update_state(loss_real * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_d_fake.update_state(loss_fake * self.distribute_strategy.num_replicas_in_sync)

        self.distribute_strategy.run(step_fn, args=(dataset_inputs,))

        
    def test_step(self, dataset_inputs):

        def step_fn(inputs):
            origin_melspecs, origin_codes, target_codes = inputs
            # generate
            generate_melspecs = self.generator((origin_melspecs, target_codes), training=False)
            cycle_melspecs = self.generator((generate_melspecs, origin_codes), training=False)
            identity_melspecs = self.generator((origin_melspecs, origin_codes), training=False)

            # origin_melspecs_crop = tf.image.random_crop(origin_melspecs, self.crop_shape)
            # generate_melspecs_crop = tf.image.random_crop(generate_melspecs, self.crop_shape)

            # discriminate
            env_code = tf.concat((target_codes, origin_codes), 1)
            # y_real = self.discriminator((origin_melspecs_crop, env_code), training=False)
            y_real = self.discriminator((origin_melspecs, env_code), training=False)
            env_code = tf.concat((origin_codes, target_codes), 1)
            # y_fake = self.discriminator((generate_melspecs_crop, env_code), training=False)
            y_fake = self.discriminator((generate_melspecs, env_code), training=False)

            total_adv_loss, total_loss_cyc, total_loss_id = self.generator_loss(y_fake, origin_melspecs, cycle_melspecs, identity_melspecs)
            loss_real, loss_fake = self.discriminator_loss(y_real, y_fake)
            g_loss = total_adv_loss + total_loss_cyc + total_loss_id
            d_loss = loss_real + loss_fake

            self.test_loss_d.update_state(d_loss * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_g.update_state(g_loss * self.distribute_strategy.num_replicas_in_sync)
            return y_real, y_fake, origin_melspecs, generate_melspecs, cycle_melspecs, identity_melspecs

        y_real, y_fake, origin_melspecs, generate_melspecs, cycle_melspecs, identity_melspecs = self.distribute_strategy.run(step_fn, args=(dataset_inputs,))
        return y_real, y_fake, origin_melspecs, generate_melspecs, cycle_melspecs, identity_melspecs


    def train(self):
        self.summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)

        with self.summary_writer.as_default():  # 構造をテキストで保存
            # with io.StringIO() as buf:
            #     self.generator.summary(print_fn=lambda x: buf.write(x + "\n"))
            #     text = buf.getvalue()
            # tf.summary.text("generator_summary", text, 0)
            with io.StringIO() as buf:
                self.discriminator.summary(print_fn=lambda x: buf.write(x + "\n"))
                text = buf.getvalue()
            tf.summary.text("discriminator_summary", text, 0)
            tf.summary.text("args_summary", json.dumps(self.args.__dict__), 0)

        with open(os.path.join(".", "datasets", "dataset_size.pkl"), 'rb') as p:
            train_size, test_size = pickle.load(p)  
        train_size, test_size = train_size//self.args.batch_size, test_size//self.args.batch_size



        for iteration in tqdm(range(self.args.start_iteration, self.args.iterations), initial=self.args.start_iteration, total=self.args.iterations):

            for i, train_data in enumerate(tqdm(self.train_dataset, leave=False, total=train_size)):

                if self.args.save_profile: 
                    if iteration == 0 and i == 3:
                        print("profiler start")
                        tf.profiler.experimental.start(self.log_dir)
                    if iteration == 1 and i == 3:
                        print("profiler stop")
                        tf.profiler.experimental.stop()

                if iteration == self.args.id_rate and i == 0:
                    print("change lambda id")
                    self.args.lambda_id = 0.0
                    # corresp for retracting
                    if self.args.mixed_precision and not self.args.use_tpu:
                        self.train_step_func = tf.function(self.train_step_mp, jit_compile=True)
                        self.test_step_func = tf.function(self.test_step, jit_compile=True)
                    else:
                        self.train_step_func = tf.function(self.train_step)
                        self.test_step_func = tf.function(self.test_step)
                self.train_step_func(train_data)

            for i, test_data in enumerate(tqdm(self.test_dataset, leave=False, total=test_size)):     
                y_real, y_fake, origin_melspecs, generate_melspecs, cycle_melspecs, identity_melspecs = self.test_step_func(test_data)


            train_loss_g = self.train_loss_g.result()
            train_loss_d = self.train_loss_d.result()
            test_loss_g = self.test_loss_g.result()
            test_loss_d = self.test_loss_d.result()
            train_loss_g_cyc = self.train_loss_g_cyc.result()
            train_loss_g_id = self.train_loss_g_id.result()
            train_loss_d_real = self.train_loss_d_real.result()
            train_loss_d_fake = self.train_loss_d_fake.result()

            self.train_loss_g.reset_state()
            self.train_loss_d.reset_state()
            self.test_loss_g.reset_state()
            self.test_loss_d.reset_state()
            self.train_loss_g_cyc.reset_state()
            self.train_loss_g_id.reset_state()
            self.train_loss_d_real.reset_state()
            self.train_loss_d_fake.reset_state()

            if (iteration + 1) % self.args.sample_interval == 0 or (iteration + 1) == self.args.iterations:
                # 訓練の進捗を出力する
                print(f"train_loss: g:{train_loss_g}, d:{train_loss_d}, test_loss: g:{test_loss_g}, d:{test_loss_d}")

                # モデルの書き出し
                save_dir = os.path.join(self.savedmodel_dir, f"{(iteration + 1):05d}")
                os.makedirs(save_dir, exist_ok=True)
                self.generator.save(save_dir, include_optimizer = False)
                    

            # TensorBoardにloss類と元/生成melspecを保存
            with self.summary_writer.as_default():
                tf.summary.scalar("train/training_loss_g", train_loss_g, iteration + 1)
                tf.summary.scalar("train/training_loss_d", train_loss_d, iteration + 1)

                tf.summary.scalar("train_summary/cycle_loss", train_loss_g_cyc, iteration + 1)
                tf.summary.scalar("train_summary/identity_loss",train_loss_g_id, iteration + 1)
                tf.summary.scalar("train_summary/discriminator_real", train_loss_d_real, iteration + 1)
                tf.summary.scalar("train_summary/discriminator_fake", train_loss_d_fake, iteration + 1)

                tf.summary.scalar("test/test_loss_g", test_loss_g, iteration + 1)
                tf.summary.scalar("test/test_loss_d", test_loss_d, iteration + 1)
                
                if self.distribute_strategy.num_replicas_in_sync == 1:
                    if iteration == 0:
                        tf.summary.image("origin_melspecs", origin_melspecs[:2], iteration + 1)
                    tf.summary.image("generate_melspecs", generate_melspecs[:2], iteration + 1)
                    tf.summary.image("cycle_melspecs", cycle_melspecs[:2], iteration + 1)
                    tf.summary.image("identity_melspecs", identity_melspecs[:2], iteration + 1)
                    tf.summary.text("test_summary/y_real", tf.strings.as_string(y_real), iteration + 1)
                    tf.summary.text("test_summary/y_fake", tf.strings.as_string(y_fake), iteration + 1)
                    tf.summary.text("test_summary/gen_melspec_min", tf.strings.as_string(tf.math.reduce_min(generate_melspecs[0])), iteration + 1)
                    tf.summary.text("test_summary/gen_melspec_max", tf.strings.as_string(tf.math.reduce_max(generate_melspecs[0])), iteration + 1)
                else:
                    if iteration == 0:
                        tf.summary.image("origin_melspecs", origin_melspecs.values[0][:2], iteration + 1)
                    tf.summary.image("generate_melspecs", generate_melspecs.values[0][:2], iteration + 1)
                    tf.summary.image("cycle_melspecs", cycle_melspecs.values[0][:2], iteration + 1)
                    tf.summary.image("identity_melspecs", identity_melspecs.values[0][:2], iteration + 1)
                    tf.summary.text("test_summary/y_real", tf.strings.as_string(y_real.values[0]), iteration + 1)
                    tf.summary.text("test_summary/y_fake", tf.strings.as_string(y_fake.values[0]), iteration + 1)
                    tf.summary.text("test_summary/gen_melspec_min", tf.strings.as_string(tf.math.reduce_min(generate_melspecs.values[0][0])), iteration + 1)
                    tf.summary.text("test_summary/gen_melspec_max", tf.strings.as_string(tf.math.reduce_max(generate_melspecs.values[0][0])), iteration + 1)

            # 異常終了対策
            self.checkpoint_manager.save()


