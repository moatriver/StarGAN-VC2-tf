import tensorflow as tf
import os
import pickle
import numpy as np

import pyworld as pw
import pysptk as sptk
import soundfile as sf

import voice_helper
import models
from setup_args import Args
from stargan_vc2 import StarGAN_VC2
ARGS = Args()

def conversion_leaned(model, source_id, target_id):
    with open(os.path.join(".", "datasets", "onehot_dict.pkl"), 'rb') as p:
        onehot_vectors = pickle.load(p) 
    with open(os.path.join(".", "datasets", "dist_dict.pkl"), 'rb') as p:
        distribution = pickle.load(p)  

    source_path = "./datasets/jvs_datasets/" + source_id + "/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
    target_path = "./datasets/jvs_datasets/" + target_id + "/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"

    s_f0, s_mceps, s_ap, s_sr = voice_helper.get_conversion_data(source_path)
    s_mceps = s_mceps.astype(np.float32)
    
    # mceps正規化
    mceps_mean, mceps_std = distribution[source_id]
    s_mceps = (s_mceps - mceps_mean) / mceps_std

    pad_size = (4 - len(s_mceps) % 4)
    s_mceps = np.pad(s_mceps, [(0, pad_size),(0,0)], 'constant')

    s_mceps_ndarray = s_mceps.T[np.newaxis, :, :, np.newaxis]
    target_ndarray = onehot_vectors[target_id][np.newaxis, :]

    t_mceps_tensor = model((s_mceps_ndarray, target_ndarray))
    t_mceps_ndarray = t_mceps_tensor.numpy()
    t_mceps = np.squeeze(t_mceps_ndarray).T.copy(order='C')

    # mceps逆正規化
    mceps_mean, mceps_std = distribution[target_id]
    t_mceps = (t_mceps * mceps_std) + mceps_mean

    convert_voice = voice_helper.synthesize_convert_voice(s_f0, t_mceps[:-pad_size], s_ap, s_sr, target_path, write_path = f"./conversion_data/{source_id}to{target_id}_convert.wav")

    return convert_voice


def conversion_nonlearn(model, source_path, target_id):
    with open(os.path.join(".", "datasets", "onehot_dict.pkl"), 'rb') as p:
        onehot_vectors = pickle.load(p) 
    with open(os.path.join(".", "datasets", "dist_dict.pkl"), 'rb') as p:
        distribution = pickle.load(p)  

    target_path = "./datasets/jvs_datasets/" + target_id + "/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"

    s_f0, s_mceps, s_ap, s_sr = voice_helper.get_conversion_data(source_path)
    s_mceps = s_mceps.astype(np.float32)
    
    # mceps正規化
    mceps_mean = np.mean(s_mceps, axis=0, keepdims=False)
    mceps_std = np.std(s_mceps, axis=0, keepdims=False)
    s_mceps = (s_mceps - mceps_mean) / mceps_std

    pad_size = (4 - len(s_mceps) % 4)
    s_mceps = np.pad(s_mceps, [(0, pad_size),(0,0)], 'constant')

    s_mceps_ndarray = s_mceps.T[np.newaxis, :, :, np.newaxis]
    target_ndarray = onehot_vectors[target_id][np.newaxis, :]

    t_mceps_tensor = model((s_mceps_ndarray, target_ndarray))
    t_mceps_ndarray = t_mceps_tensor.numpy()
    t_mceps = np.squeeze(t_mceps_ndarray).T.copy(order='C')

    # mceps逆正規化
    mceps_mean, mceps_std = distribution[target_id]
    t_mceps = (t_mceps * mceps_std) + mceps_mean

    convert_voice = voice_helper.synthesize_convert_voice(s_f0, t_mceps[:-pad_size], s_ap, s_sr, target_path, write_path = f"./conversion_data/ORGto{target_id}_convert.wav")

    return convert_voice

if __name__ == "__main__":  
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    # set generator weights
    load_generator = tf.keras.models.load_model("./saved_models/stargan_vc2/20221121-093359/15000", compile = False)
    generator = models.Generator(ARGS.code_size)
    generator((tf.random.uniform((1, 35, 444, 1)), tf.random.uniform((1, 4)))) # build
    generator.set_weights(load_generator.get_weights()) 
    
    # m = ["jvs009", "jvs020"]
    # f = ["jvs002", "jvs010"]

    conversion_leaned(generator, "jvs009", "jvs009")
    print("9to9 done!")

    conversion_leaned(generator, "jvs009", "jvs020")
    print("9to20 done!")

    conversion_leaned(generator, "jvs009", "jvs002")
    print("9to2 done!")

    conversion_leaned(generator, "jvs009", "jvs010")
    print("9to10 done!")

    conversion_nonlearn(generator, "/workspace/StarGAN-VC2-tf/datasets/test_conersion.wav", "jvs009")
    conversion_nonlearn(generator, "/workspace/StarGAN-VC2-tf/datasets/test_conersion.wav", "jvs020")
    conversion_nonlearn(generator, "/workspace/StarGAN-VC2-tf/datasets/test_conersion.wav", "jvs002")
    conversion_nonlearn(generator, "/workspace/StarGAN-VC2-tf/datasets/test_conersion.wav", "jvs010")
    print("test conversion done!")