import numpy as np
import pyworld as pw
import pysptk as sptk
import soundfile as sf
import librosa

from setup_args import Args
ARGS = Args()

def get_conversion_data(file_path):
    x, sr = sf.read(file_path, always_2d=True)
    x = x[:, 0]
    target_sr = ARGS.target_sr
    x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    # frame_period : Period between consecutive frames in milliseconds, default:5.0
    # fft_size : Length of Fast Fourier Transform (in number of samples) The resulting dimension of `ap` adn `sp` will be `fft_size` // 2 + 1, default:1024
    f0, sp, ap = pw.wav2world(x, sr, fft_size=ARGS.fft_size, frame_period=ARGS.frame_period)

    alpha = 1/2.9 * np.log10(sr/1000) # https://qiita.com/mitsu-h/items/ba51b3a66a690bd1a502
    mceps = sptk.sp2mc(sp, order=ARGS.mcep_size-1, alpha=alpha)

    return f0, mceps, ap, sr

def synthesize_convert_voice(f0, converted_mceps, ap, sr, target_file_path, write_path = None):

    x_t, sr_t = sf.read(target_file_path) # sr:24000
    target_sr = ARGS.target_sr
    x_t = librosa.resample(x_t, orig_sr=sr_t, target_sr=target_sr)
    sr_t = target_sr

    _f0_t, t_t = pw.dio(x_t, sr_t)    # raw pitch extractor
    f0_t = pw.stonemask(x_t, _f0_t, t_t, sr_t)  # pitch refinement

    np.putmask(f0, f0 < 0.5, None)
    np.putmask(f0_t, f0_t < 0.5, None)

    mu = np.nanmean(np.log(f0))
    var = np.nanvar(np.log(f0))
    sigma = np.sqrt(var)

    mu_t = np.nanmean(np.log(f0_t))
    var_t = np.nanvar(np.log(f0_t))
    sigma_t = np.sqrt(var_t)

    # f0 transformation
    f0_conv = np.e ** ((np.log(f0) - mu)/sigma * sigma_t + mu_t)
    np.putmask(f0_conv, np.isnan(f0_conv), 0)

    alpha = 1/2.9 * np.log10(sr/1000) # https://qiita.com/mitsu-h/items/ba51b3a66a690bd1a502
    converted_sp = sptk.mc2sp(converted_mceps, alpha=alpha, fftlen=ARGS.fft_size)

    convert_voice = pw.synthesize(f0_conv, converted_sp, ap, sr, ARGS.frame_period)

    if write_path != None:
        sf.write(write_path, convert_voice, sr)

    return convert_voice
