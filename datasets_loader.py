import glob
import os
import re
import tensorflow as tf
import tqdm
import numpy as np
import random
import pickle
import gc
from sklearn.preprocessing import MinMaxScaler

from voice_helper import get_conversion_data

from setup_args import Args
ARGS = Args()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class DatasetLoader():
    def __init__(self):
        if ARGS.set_seed:
            random.seed(ARGS.seed)

        if ARGS.use_tpu:
            self.train_dataset_path = os.path.join("gs://stargan-vc2-data", "train")
            self.test_dataset_path = os.path.join("gs://stargan-vc2-data", "test")
        else:
            self.train_dataset_path = os.path.join(".", "datasets", "tf_datasets", "train")
            self.test_dataset_path = os.path.join(".", "datasets", "tf_datasets", "test")

        self.train_shard_num = ARGS.num_workers*2
        self.test_shard_num = ARGS.num_workers*1

        shard_pattern = "shard_{}.records"
        self.shard_read_pattern = shard_pattern.format("*")
        self.shard_write_pattern = shard_pattern.format("{:08d}")


        self.datasets_dtype = {
            'mcep': {"numpy_dtype": np.float32, "tensor_dtype": tf.float32},
            'source': {"numpy_dtype": np.uint8, "tensor_dtype": tf.uint8},
            'target': {"numpy_dtype": np.uint8, "tensor_dtype": tf.uint8},
        }

        if ARGS.remake_datasets or (not ARGS.use_tpu and (not os.path.isdir(self.train_dataset_path) or not os.path.isdir(self.test_dataset_path))):
            self.make_datasets()

        train_dataset_pattern_path = os.path.join(self.train_dataset_path, self.shard_read_pattern)
        test_dataset_pattern_path = os.path.join(self.test_dataset_path, self.shard_read_pattern)

        self.train_shard_files = tf.io.matching_files(train_dataset_pattern_path)
        self.test_shard_files = tf.io.matching_files(test_dataset_pattern_path)
    

    def to_example(self, data):
        mceps_ndarray, source_ndarray, target_ndarray = data
        feature = {
            'mcep': _bytes_feature(tf.io.serialize_tensor(mceps_ndarray)),
            'source': _bytes_feature(tf.io.serialize_tensor(source_ndarray)),
            'target': _bytes_feature(tf.io.serialize_tensor(target_ndarray)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


    def parse_example(self, example_proto):
        feature_description = {
            'mcep': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'source': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        parsed_elem = tf.io.parse_example(example_proto, feature_description)
        for key in feature_description.keys():
            parsed_elem[key] = tf.io.parse_tensor(parsed_elem[key], out_type=self.datasets_dtype[key]["tensor_dtype"])

        return list(parsed_elem.values())


    def load_dataset(self, shard_files, batch_size, use_shuffle):
        shards = tf.data.Dataset.from_tensor_slices(shard_files)
        if use_shuffle:
            shards = shards.shuffle(ARGS.shuffle_buffer_Size)
        dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE).cache()
        dataset = dataset.map(map_func=self.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.map(lambda mceps, source, target: (mceps, tf.cast(source, tf.int64), tf.cast(target, tf.int64)) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if use_shuffle:
            dataset = dataset.shuffle(ARGS.shuffle_buffer_Size)
        return dataset.batch(batch_size, drop_remainder=True)


    def get_train_set(self, batch_size, repeat_num = 1):
        dataset = self.load_dataset(self.train_shard_files, batch_size, use_shuffle=True)
        self.train_dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return self.train_dataset

    def get_test_set(self, batch_size):
        dataset = self.load_dataset(self.test_shard_files, batch_size, use_shuffle=False)
        self.test_dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return self.test_dataset

    def make_datasets(self):
        using_jvs_id = ARGS.using_jvs_id_m + ARGS.using_jvs_id_f

        onehot_dict_path = os.path.join(".", "datasets", "onehot_dict.pkl")
        if os.path.isfile(onehot_dict_path):
            with open(onehot_dict_path, 'rb') as p:
                onehot_vectors = pickle.load(p)  
        else:
            onehot_array = np.identity(len(using_jvs_id), dtype=self.datasets_dtype["source"]["numpy_dtype"])
            onehot_vectors = dict()
            for i, id in enumerate(using_jvs_id):
                onehot_vectors[id] = onehot_array[i]
            with open(onehot_dict_path, 'wb') as p:
                pickle.dump(onehot_vectors, p)

        npz_path = './datasets/tmp_data'
        if os.path.isfile(npz_path + ".npz"):
            npz = np.load(npz_path + ".npz")
            mceps_ndarray, source_ndarray, target_ndarray = npz["arr_0"], npz["arr_1"], npz["arr_2"]
        else:
            mceps_ndarray = np.empty((0, ARGS.mcep_size, ARGS.dataset_t_length, 1), self.datasets_dtype["mcep"]["numpy_dtype"])
            source_ndarray = np.empty((0, len(using_jvs_id)), self.datasets_dtype["source"]["numpy_dtype"])
            target_ndarray = np.empty((0, len(using_jvs_id)), self.datasets_dtype["target"]["numpy_dtype"])

            dist_dict_path = os.path.join(".", "datasets", "dist_dict.pkl")
            if os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'rb') as p:
                    distribution = pickle.load(p)  
            else:
                distribution = dict()

            for jvs_id in tqdm.tqdm(using_jvs_id):
                data_path_list = np.array(glob.glob(os.path.join(".", "datasets", "jvs_datasets", jvs_id, "parallel100", "*", "*.wav")))

                p = np.random.permutation(len(data_path_list))
                data_path_list = data_path_list[p]
                data_path_list = data_path_list[:int(len(data_path_list) * ARGS.preset_datafile_ratio)]

                mceps_list = []
                for data_path in tqdm.tqdm(data_path_list, leave=False):
                    _, mceps, _, _ = get_conversion_data(data_path)
                    mceps_list.append(mceps)

                if os.path.isfile(dist_dict_path):
                    mceps_mean, mceps_std = distribution[jvs_id]
                else:
                    mceps_concatenated = np.concatenate(mceps_list, axis=0)
                    mceps_mean = np.mean(mceps_concatenated, axis=0, keepdims=False)
                    mceps_std = np.std(mceps_concatenated, axis=0, keepdims=False)
                    distribution[jvs_id] = [mceps_mean, mceps_std]

                for mceps in tqdm.tqdm(mceps_list, leave=False):
                    mceps = (mceps - mceps_mean) / mceps_std
                    mceps = mceps.astype(self.datasets_dtype["mcep"]["numpy_dtype"])

                    # pad_size = (ARGS.dataset_t_length - (len(mceps) % ARGS.dataset_hop_size))
                    # mceps = np.pad(mceps, [(pad_size//2, pad_size - (pad_size//2)),(0,0)], 'constant')
                    hop_list = list(range(0, len(mceps)-ARGS.dataset_t_length, ARGS.dataset_hop_size))
                    hop_list.append(len(mceps)-ARGS.dataset_t_length)
                    
                    sorce_jvs_id = re.search("jvs\d{3}", data_path).group()
                    for frame in tqdm.tqdm(hop_list, leave=False):
                        mceps_frame = np.copy(mceps[frame:frame+ARGS.dataset_t_length, :]).T
                        target_jvs_id = random.choice(list(set(using_jvs_id) - set([sorce_jvs_id])))

                        mceps_ndarray = np.append(mceps_ndarray, mceps_frame[np.newaxis, :, :, np.newaxis], axis = 0)
                        source_ndarray = np.append(source_ndarray, onehot_vectors[sorce_jvs_id][np.newaxis, :], axis = 0)
                        target_ndarray = np.append(target_ndarray, onehot_vectors[target_jvs_id][np.newaxis, :], axis = 0)

            if not os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'wb') as p:
                    pickle.dump(distribution, p)
            np.savez(npz_path, mceps_ndarray, source_ndarray, target_ndarray)

        p = np.random.permutation(len(source_ndarray))
        mceps_ndarray = mceps_ndarray[p]
        source_ndarray = source_ndarray[p]
        target_ndarray = target_ndarray[p]

        train_size = int(len(source_ndarray)*ARGS.train_data_ratio)
        train_dataset = tf.data.Dataset.from_tensor_slices((mceps_ndarray[:train_size], source_ndarray[:train_size], target_ndarray[:train_size]))
        test_dataset = tf.data.Dataset.from_tensor_slices((mceps_ndarray[train_size:], source_ndarray[train_size:], target_ndarray[train_size:]))

        ds_size_path = os.path.join(".", "datasets", "dataset_size.pkl")
        with open(ds_size_path, 'wb') as p:
            pickle.dump([train_size, len(source_ndarray)-train_size], p)

        os.makedirs(self.train_dataset_path, exist_ok=True)
        os.makedirs(self.test_dataset_path, exist_ok=True)

        for i in tqdm.tqdm(range(self.train_shard_num)):
            tfrecords_shard_path = os.path.join(self.train_dataset_path, self.shard_write_pattern.format(i))
            shard_data = train_dataset.shard(self.train_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        for i in tqdm.tqdm(range(self.test_shard_num)):
            tfrecords_shard_path = os.path.join(self.test_dataset_path, self.shard_write_pattern.format(i))
            shard_data = test_dataset.shard(self.test_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        del train_dataset
        del test_dataset
        gc.collect()

    def make_datasets_from_vcc2018(self):
        using_id = ["VCC2TF1", "VCC2TF2", "VCC2TM1", "VCC2TM2"]
        onehot_dict_path = os.path.join(".", "datasets", "onehot_dict_vcc2018.pkl")
        if os.path.isfile(onehot_dict_path):
            with open(onehot_dict_path, 'rb') as p:
                onehot_vectors = pickle.load(p)  
        else:
            onehot_array = np.identity(len(using_id), dtype=self.datasets_dtype["source"]["numpy_dtype"])
            onehot_vectors = dict()
            for i, id in enumerate(using_id):
                onehot_vectors[id] = onehot_array[i]
            with open(onehot_dict_path, 'wb') as p:
                pickle.dump(onehot_vectors, p)

        npz_path = './datasets/tmp_data_vcc2018_train'
        if os.path.isfile(npz_path + ".npz"):
            npz = np.load(npz_path + ".npz")
            mceps_ndarray, source_ndarray, target_ndarray = npz["arr_0"], npz["arr_1"], npz["arr_2"]
        else:
            mceps_ndarray = np.empty((0, ARGS.mcep_size, ARGS.dataset_t_length, 1), self.datasets_dtype["mcep"]["numpy_dtype"])
            source_ndarray = np.empty((0, len(using_id)), self.datasets_dtype["source"]["numpy_dtype"])
            target_ndarray = np.empty((0, len(using_id)), self.datasets_dtype["target"]["numpy_dtype"])

            dist_dict_path = os.path.join(".", "datasets", "dist_dict_vcc2018.pkl")
            if os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'rb') as p:
                    distribution = pickle.load(p)  
            else:
                distribution = dict()

            for vcc_id in tqdm.tqdm(using_id):
                data_path_list = glob.glob(os.path.join(".", "datasets", "vcc2018_datasets", "vcc2018_training", vcc_id, "*.wav"))

                mceps_list = []
                for data_path in tqdm.tqdm(data_path_list, leave=False):
                    _, mceps, _, _ = get_conversion_data(data_path)
                    mceps_list.append(mceps)

                if os.path.isfile(dist_dict_path):
                    mceps_mean, mceps_std = distribution[vcc_id]
                else:
                    mceps_concatenated = np.concatenate(mceps_list, axis=0)
                    mceps_mean = np.mean(mceps_concatenated, axis=0, keepdims=False)
                    mceps_std = np.std(mceps_concatenated, axis=0, keepdims=False)
                    distribution[vcc_id] = [mceps_mean, mceps_std]

                for mceps in tqdm.tqdm(mceps_list, leave=False):
                    mceps = (mceps - mceps_mean) / mceps_std
                    mceps = mceps.astype(self.datasets_dtype["mcep"]["numpy_dtype"])

                    # pad_size = (ARGS.dataset_t_length - (len(mceps) % ARGS.dataset_hop_size))
                    # mceps = np.pad(mceps, [(pad_size//2, pad_size - (pad_size//2)),(0,0)], 'constant')
                    hop_list = list(range(0, len(mceps)-ARGS.dataset_t_length, ARGS.dataset_hop_size))
                    # hop_list.append(len(mceps)-ARGS.dataset_t_length)
                    hop_list = [ i + (len(mceps) % ARGS.dataset_hop_size)//2 for i in hop_list]
                    
                    sorce_jvs_id = re.search("VCC2.{3}", data_path).group()
                    for frame in tqdm.tqdm(hop_list, leave=False):
                        mceps_frame = np.copy(mceps[frame:frame+ARGS.dataset_t_length, :]).T
                        target_jvs_id = random.choice(list(set(using_id) - set([sorce_jvs_id])))
                        mceps_ndarray = np.append(mceps_ndarray, mceps_frame[np.newaxis, :, :, np.newaxis], axis = 0)
                        source_ndarray = np.append(source_ndarray, onehot_vectors[sorce_jvs_id][np.newaxis, :], axis = 0)
                        target_ndarray = np.append(target_ndarray, onehot_vectors[target_jvs_id][np.newaxis, :], axis = 0)

            if not os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'wb') as p:
                    pickle.dump(distribution, p)
            np.savez(npz_path, mceps_ndarray, source_ndarray, target_ndarray)

        p = np.random.permutation(len(source_ndarray))
        mceps_ndarray = mceps_ndarray[p]
        source_ndarray = source_ndarray[p]
        target_ndarray = target_ndarray[p]

        train_dataset = tf.data.Dataset.from_tensor_slices((mceps_ndarray, source_ndarray, target_ndarray))
        train_size = len(source_ndarray)
        
        npz_path = './datasets/tmp_data_vcc2018_test'
        if os.path.isfile(npz_path + ".npz"):
            npz = np.load(npz_path + ".npz")
            mceps_ndarray, source_ndarray, target_ndarray = npz["arr_0"], npz["arr_1"], npz["arr_2"]
        else:
            mceps_ndarray = np.empty((0, ARGS.mcep_size, ARGS.dataset_t_length, 1), self.datasets_dtype["mcep"]["numpy_dtype"])
            source_ndarray = np.empty((0, len(using_id)), self.datasets_dtype["source"]["numpy_dtype"])
            target_ndarray = np.empty((0, len(using_id)), self.datasets_dtype["target"]["numpy_dtype"])

            dist_dict_path = os.path.join(".", "datasets", "dist_dict_vcc2018.pkl")
            if os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'rb') as p:
                    distribution = pickle.load(p)  
            else:
                distribution = dict()

            for vcc_id in tqdm.tqdm(using_id):
                data_path_list = glob.glob(os.path.join(".", "datasets", "vcc2018_datasets", "vcc2018_reference", vcc_id, "*.wav"))

                mceps_list = []
                for data_path in tqdm.tqdm(data_path_list, leave=False):
                    _, mceps, _, _ = get_conversion_data(data_path)
                    mceps_list.append(mceps)

                if os.path.isfile(dist_dict_path):
                    mceps_mean, mceps_std = distribution[vcc_id]
                else:
                    mceps_concatenated = np.concatenate(mceps_list, axis=0)
                    mceps_mean = np.mean(mceps_concatenated, axis=0, keepdims=False)
                    mceps_std = np.std(mceps_concatenated, axis=0, keepdims=False)
                    distribution[vcc_id] = [mceps_mean, mceps_std]

                for mceps in tqdm.tqdm(mceps_list, leave=False):
                    mceps = (mceps - mceps_mean) / mceps_std
                    mceps = mceps.astype(self.datasets_dtype["mcep"]["numpy_dtype"])

                    # pad_size = (ARGS.dataset_t_length - (len(mceps) % ARGS.dataset_hop_size))
                    # mceps = np.pad(mceps, [(pad_size//2, pad_size - (pad_size//2)),(0,0)], 'constant')
                    hop_list = list(range(0, len(mceps)-ARGS.dataset_t_length, ARGS.dataset_hop_size))
                    # hop_list.append(len(mceps)-ARGS.dataset_t_length)
                    hop_list = [ i + (len(mceps) % ARGS.dataset_hop_size)//2 for i in hop_list]
                    
                    sorce_jvs_id = re.search("VCC2.{3}", data_path).group()
                    for frame in tqdm.tqdm(hop_list, leave=False):
                        mceps_frame = np.copy(mceps[frame:frame+ARGS.dataset_t_length, :]).T
                        target_jvs_id = random.choice(list(set(using_id) - set([sorce_jvs_id])))

                        mceps_ndarray = np.append(mceps_ndarray, mceps_frame[np.newaxis, :, :, np.newaxis], axis = 0)
                        source_ndarray = np.append(source_ndarray, onehot_vectors[sorce_jvs_id][np.newaxis, :], axis = 0)
                        target_ndarray = np.append(target_ndarray, onehot_vectors[target_jvs_id][np.newaxis, :], axis = 0)

            if not os.path.isfile(dist_dict_path):
                with open(dist_dict_path, 'wb') as p:
                    pickle.dump(distribution, p)
            np.savez(npz_path, mceps_ndarray, source_ndarray, target_ndarray)

        p = np.random.permutation(len(source_ndarray))
        mceps_ndarray = mceps_ndarray[p]
        source_ndarray = source_ndarray[p]
        target_ndarray = target_ndarray[p]

        test_dataset = tf.data.Dataset.from_tensor_slices((mceps_ndarray, source_ndarray, target_ndarray))
        test_size = len(source_ndarray)
        
        ds_size_path = os.path.join(".", "datasets", "dataset_size_vcc2018.pkl")
        with open(ds_size_path, 'wb') as p:
            pickle.dump([train_size, test_size], p)

        os.makedirs(self.train_dataset_path, exist_ok=True)
        os.makedirs(self.test_dataset_path, exist_ok=True)

        for i in tqdm.tqdm(range(self.train_shard_num)):
            tfrecords_shard_path = os.path.join(self.train_dataset_path, self.shard_write_pattern.format(i))
            shard_data = train_dataset.shard(self.train_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        for i in tqdm.tqdm(range(self.test_shard_num)):
            tfrecords_shard_path = os.path.join(self.test_dataset_path, self.shard_write_pattern.format(i))
            shard_data = test_dataset.shard(self.test_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        del train_dataset
        del test_dataset
        gc.collect()
        
if __name__ == "__main__":
    data = DatasetLoader()
    # data.make_datasets_from_vcc2018()
    # train_data = data.get_train_set(4)
    # 
    # print(train_data)
    # for i, datas in enumerate(train_data):
    #     print(i)
    #     mcep, sor, tar = datas
    #     print(mcep.shape)
    #     print(type(mcep), type(sor), type(tar))
    #     if i==3:
    #         exit()