from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
from tqdm import tqdm
from clean import has_sound, delete_silent_files, split_wavs


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**/*.wav'.format(args.src), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    predictions = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        print(y_pred)
        print(len(y_pred))

        y_mean = np.mean(y_pred, axis=0)
        print(y_mean)
        y_median = np.median(y_pred, axis=0)
        print(y_median)
        y_pred2 = np.argmax(y_mean)
        predicted_class_global = classes[y_pred2]
        print(f'Predicted class of {wav_fn}: {predicted_class_global}')

        print('-'*50)
        
        for i in range(len(y_pred)):
            max_index = np.argmax(y_pred[i])
            predicted_class = classes[max_index]
            predictions.append(predicted_class)
       
            print(f'Predicted class of {wav_fn}: {predicted_class}')
        
   


    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--src', type=str, default='predict222',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--src_root', type=str, default='predict',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dst_root', type=str, default='predict222',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    split_wavs(args)
    predictions = make_prediction(args)

