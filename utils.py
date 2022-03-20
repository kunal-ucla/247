import tensorflow as tf
from scipy import signal
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import shuffle
import os
import pickle

def fgsm_attack(model, image, label, eps, plot=False):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(image)
    pred = model(image)
    loss = tf.keras.losses.MSE(label, pred)
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    adversary = (image + (signedGrad * eps)).numpy()

    if plot:
        original_pred = model(image).numpy()
        adversary_pred = model(adversary).numpy()
        plot_spectrum(image[0,:,:,0], converted=True, title=f'Original with prediction {np.argmax(original_pred)}')
        plot_spectrum(adversary[0,:,:,0], converted=True, title=f'Adversary with prediction {np.argmax(adversary_pred)}')

    return adversary

def gnaa_attack(model, image, label, eps, plot=False):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(image)
    pred = model(image)
    loss = tf.keras.losses.MSE(label, pred)
    gradient = tape.gradient(loss, image)
    normdGrad = gradient / np.linalg.norm(gradient)
    adversary = (image + (normdGrad * eps)).numpy()

    if plot:
        original_pred = model(image).numpy()
        adversary_pred = model(adversary).numpy()
        plot_spectrum(image[0,:,:,0], converted=True, title=f'Original with prediction {np.argmax(original_pred)}')
        plot_spectrum(adversary[0,:,:,0], converted=True, title=f'Adversary with prediction {np.argmax(adversary_pred)}')

    return adversary

def stft(X, fs=250, nperseg=16, noverlap=None, stride=8, nfft=512, padded=True):
    if not noverlap: noverlap = nperseg - stride
    f, _, Zxx = signal.stft(X, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded, boundary=None)
    Z = np.abs(Zxx)/(fs/2)
    return f, Z

def extract(X, fs=250, nperseg=16, noverlap=None, stride=8, nfft=512, padded=False, needed_dim=62):
    pad_len = stride * (needed_dim-1) - len(X) + nperseg
    X = np.pad(X, (0,pad_len))
    
    if not noverlap: noverlap = nperseg - stride
    f, _, Zxx = signal.stft(X, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded, boundary=None)
    Z = np.abs(Zxx)/(fs/2)

    band1 = np.squeeze(Z[np.where((f>=4) & (f<=15)),:])
    band2 = np.squeeze(Z[np.where((f>=19) & (f<=30)),:])
    band2 = cv2.resize(band2, dsize=(np.shape(band1)[1],np.shape(band1)[0]), interpolation=cv2.INTER_CUBIC)

    combined = np.vstack([band1, band2])
    return combined

def extract_all(X, fs=250, nperseg=16, noverlap=None, stride=8, nfft=512, padded=False, needed_dim=62):
    electrodes = np.shape(X)[0]
    curr = np.empty((0,needed_dim))

    for electrode in range(electrodes):
        out = extract(X[electrode,:], fs, nperseg, noverlap, stride, nfft, padded, needed_dim)
        curr = np.vstack([curr,out])
    
    return curr

def plot_spectrum(X, sample_idx=0, anchor_length=128, num_electrodes=3, trim_time=500, converted=False, title=''):
    plt.figure()
    if converted: image = X
    else: image = extract_all(X[sample_idx,:num_electrodes,:trim_time], nperseg=anchor_length)
    plt.title(title+f'\nSpectral image generated for {trim_time} samples, {num_electrodes} electrodes with anchor length {anchor_length}')
    plt.imshow(image, interpolation='nearest', aspect='auto')
    plt.show()

def convert_spectrum(X, y, anchors=[16,32,64,128,256], out_dim=(44, 62)):
    N, E, _ = np.shape(X)
    X_out = np.empty((0, E*out_dim[0], out_dim[1]))
    y_out = np.empty((0,))
    for n in tqdm(range(N)):
        for anchor in anchors:
            x = extract_all(X[n,:,:], nperseg=anchor)
            X_out = np.insert(X_out, len(X_out), x, axis=0)
            y_out = np.insert(y_out, len(y_out), y[n])
    return X_out, y_out

def filter_data(data, fs, order, lowcut, highcut):
    filtered_data = np.zeros_like(data)
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    for n in np.arange(data.shape[0]):
        single_instance = data[n, :, :]
        for channel in np.arange(single_instance.shape[0]):
            X = single_instance[channel, :]
            b, a = signal.butter(order, [low, high], btype='band')
            y = signal.lfilter(b, a, X)
            filtered_data[n, channel, :] = y
    return filtered_data

def smooth_data(data, ws):
    kern = signal.hanning(ws)[None, None, :]
    kern /= kern.sum()
    return signal.convolve(data, kern, mode='same')

def train_val_split(X, y, val_ratio=4):
    N = np.shape(X)[0]
    ind_valid = np.random.choice(N, N//val_ratio, replace=False)
    ind_train = np.array(list(set(range(N)).difference(set(ind_valid))))
    X_train, X_val = X[ind_train], X[ind_valid]
    y_train, y_val = y[ind_train], y[ind_valid]
    return X_train, y_train, X_val, y_val

def shuffle_data(X, y):
    N = np.shape(X)[0]
    X_shuffle = np.copy(X)
    y_shuffle = np.copy(y)
    ind_list = [i for i in range(N)]
    shuffle(ind_list)
    X_shuffle = X_shuffle[ind_list]
    y_shuffle = y_shuffle[ind_list]
    return X_shuffle, y_shuffle

def datagen(X, y=None, mode='skipnet', window=2):
    if mode == 'skipnet':
        X = X[:, :, :500]
        X, y = convert_spectrum(X, y)
        X = np.expand_dims(X, axis=3)
        y = to_categorical(y, 4)
    elif mode == 'strip':
        X = X[:, :, :window]
    elif mode == 'max':
        X = np.max(X.reshape(X.shape[0], X.shape[1], -1, window), axis=3)
    elif mode == 'avg':
        X = np.mean(X.reshape(X.shape[0], X.shape[1], -1, window), axis=3)
    elif mode == 'noise':
        X = X + np.random.normal(0, 0.5, X.shape)
    elif mode == 'subsample':
        X = X[:, :, window::2]
    elif mode == 'reshape':
        X = np.expand_dims(X, axis=3)
        y = to_categorical(y, 4)
    elif mode == 'swapaxes':
        X = np.swapaxes(X, 1,3)
        X = np.swapaxes(X, 1,2)
    else:
        print('Invalid mode')
    return X, y

def eeg_to_spectrum(X_train, y_train, X_val, y_val, X_test, y_test):
    if os.path.isfile('skipnet_data.pickle'):
      with open('skipnet_data.pickle', 'rb') as f:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(f)
    else:
      X_train, y_train = utils.datagen(X_train, y_train)
      X_val, y_val = utils.datagen(X_val, y_val)
      X_test, y_test = utils.datagen(X_test, y_test)
      with open('skipnet_data.pickle', 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
    X_train = np.vstack((X_og, X_val[:1600], X_test[:1400]))
    y_train = np.vstack((y_og, y_val[:1600], y_test[:1400]))
    
    return (X_train, y_train, X_val, y_val, X_test, y_test)


def augment(X, y, strip_len=500, max_len=2, avg_len=2, sub_len=2):
    X, _ = datagen(X, y, mode='strip', window=strip_len)

    X_max, _ = datagen(X, y, mode='max', window=max_len)

    X_avg, _ = datagen(X, y, mode='avg', window=avg_len)
    X_avg, _ = datagen(X_avg, y, mode='noise')

    X_sub_1, _ = datagen(X, y, mode='subsample', window=0)
    X_sub_1, _ = datagen(X_sub_1, y, mode='noise')
    
    X_sub_2, _ = datagen(X, y, mode='subsample', window=1)
    X_sub_2, _ = datagen(X_sub_2, y, mode='noise')

    X_total = np.vstack((X_max, X_avg, X_sub_1, X_sub_2))
    y_total = np.hstack((y, y, y, y))
    return X_total, y_total

def augment_reshape(X, y, strip_len=500, max_len=2, avg_len=2, sub_len=2):
    X, y = augment(X, y, strip_len, max_len, avg_len, sub_len)
    X, y = datagen(X, y, mode='reshape')
    X, _ = datagen(X, y, mode='swapaxes')
    return X, y

def plot_history(ret, title1='Accuracy Plot', title2='Loss Plot'):
    plt.plot(ret.history['accuracy'])
    plt.plot(ret.history['val_accuracy'])
    plt.title(title1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plotting loss trajectory
    plt.plot(ret.history['loss'],'o')
    plt.plot(ret.history['val_loss'],'o')
    plt.title(title2)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def max_vote(model, X, y, ret_corr=False, n_augmented=5):
    y_pred = model.predict(X)
    N = np.shape(X)[0]
    y_out = np.zeros(N//n_augmented)
    count = 0
    corr_idx = []
    for i in range(N//n_augmented):
        idx = i*n_augmented
        pred = np.zeros(n_augmented)
        for j in range(n_augmented): pred[np.argmax(y_pred[idx+j])] += 1
        y_out[i] = np.argmax(pred)
        if y_out[i] == np.argmax(y[idx]):
            count+=1
            corr_idx += [idx, idx+1, idx+2, idx+3, idx+4]
    accuracy = count*100/(N//n_augmented)
    print(f'Accuracy = {accuracy}%')
    if ret_corr:
        return np.array(corr_idx)

def max_vote_cnn(model, X, y, n_augmented=4):
    y_pred = model.predict(X)
    N = np.shape(X)[0]
    count = 0
    for i in range(N//n_augmented):
        pred = np.zeros(n_augmented)
        for j in range(n_augmented):
          pred[np.argmax(y_pred[i + (N//n_augmented)*j])] += 1
        if np.argmax(pred) == np.argmax(y[i]): count+=1
    accuracy = count/(N//n_augmented)
    return accuracy

def acc_per_vote(model, X, y, n_augmented=5):
    y_pred = model.predict(X)
    N = np.shape(X)[0]
    count = [0]*n_augmented
    for i in range(N//n_augmented):
        idx = i*n_augmented
        for j in range(n_augmented):
          if np.argmax(y_pred[idx+j]) == np.argmax(y[idx+j]): count[j]+=1
    acc = [0]*n_augmented
    for j in range(n_augmented): acc[j] = count[j]*100/(N//n_augmented)
    return acc

def generate_samples(X, y, model, eps=0.001):
    X_new = np.empty((0, np.shape(X)[1], np.shape(X)[2], 1))
    y_new = np.empty((0, np.shape(y)[1]))
    N = np.shape(X)[0]
    for i in tqdm(range(N)):
        adversary = fgsm_attack(model, X[i:i+1], y[i], eps)
        X_new = np.insert(X_new, len(X_new), adversary, axis=0)
        y_new = np.insert(y_new, len(y_new), y[i], axis=0)
    return X_new, y_new
