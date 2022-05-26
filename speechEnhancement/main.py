import os

import librosa
import soundfile
import numpy as np
import csv
import time
from scipy import signal
import pickle
import pickle as cPickle
import h5py
from sklearn import preprocessing
import main as main
import CNN as CNN

speechDir = r"C:\Users\pavan\PycharmProjects\speechEnhancement\audio\speech"
noiseDir = r"C:\Users\pavan\PycharmProjects\speechEnhancement\audio\noise"


def folder(f):
    if not os.path.exists(f):
        os.makedirs(f)


def readAudio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def writeAudio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def createMixtures():
    magnification = 3
    fs = 16000

    speechNames = [na for na in os.listdir(speechDir) if na.lower().endswith(".wav")]
    noiseNames = [na for na in os.listdir(noiseDir) if na.lower().endswith(".wav")]

    rs = np.random.RandomState(0)
    csv = os.path.join("workspace1", "mixture", "%s.csv" % dataType)
    main.folder(os.path.dirname(csv))

    cnt = 0

    f = open(csv, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))
    for speechNa in speechNames:
        speechPath = os.path.join(speechDir, speechNa)
        (speechAudio, _) = readAudio(speechPath)

        if dataType == 'train':
            selectedNoise = rs.choice(noiseNames, size=magnification, replace=False)
        elif dataType == 'test':
            selectedNoise = noiseNames
        else:
            raise Exception ("dataType has to be train or test")

        for noiseNa in selectedNoise:
            noisePath = os.path.join(noiseDir, noiseNa)
            (noiseAudio, _) = readAudio(noisePath)

            if cnt % 100 == 0:
                print(cnt)

            cnt += 1
            f.write("%s\t%s\t%d\t%d\n" % (speechNa, noiseNa, 0 , len(noiseAudio)))
        #f.close()
        print(csv)
        print("Create %s mixture csv finished!" % dataType)

def extractFeatures():
    snr = 0
    fs = 16000

    csvPath = os.path.join("workspace1", "mixture", "%s.csv" % dataType)
    with open(csvPath, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        readerList = list(reader)

    t = time.time()
    cnt = 0
    for i in range(1,len(readerList)):
        [speechNa, noiseNa, noiseOnset, noiseOffset] = readerList[i]
        noiseOnset = int(noiseOnset)
        noiseOffset = int(noiseOffset)

        speechPath = os.path.join(speechDir, speechNa)
        (speechAudio, _) = readAudio(speechPath, target_fs=fs)
        noisePath = os.path.join(noiseDir, noiseNa)
        (noiseAudio, _) = readAudio(noisePath, target_fs=fs)

        snRmsRatio = np.sqrt(np.mean(np.abs(speechAudio) ** 2, axis=0, keepdims=False))/np.sqrt(np.mean(np.abs(noiseAudio) ** 1, axis=0, keepdims=False))
        targetRatio = 10. ** (float(snr)/20.)
        scalar = snRmsRatio/targetRatio
        speechAudio *= scalar

        mixedAudio = speechAudio + noiseAudio
        alpha = 1. / np.max(np.abs(mixedAudio))
        mixedAudio *= alpha
        speechAudio *= alpha
        noiseAudio *= alpha

        outNA = os.path.join("%s.%s" % (os.path.splitext(speechNa)[0], os.path.splitext(noiseNa)[0]))
        audioPath = os.path.join("workspace1", "mixed_audios", "spectrogram",
                dataType, "%ddb" % int(snr), "%s.wav" % outNA)

        folder(os.path.dirname(audioPath))
        writeAudio(audioPath, mixedAudio, fs)

        mX = spectrogram(mixedAudio, mode='complex')
        speech_x = spectrogram(speechAudio, mode='magnitude')
        noise_x = spectrogram(noiseAudio, mode='magnitude')

        features = os.path.join("workspace1", "features", "spectrogram", dataType, "%ddb" % int(snr), "%s.txt" % outNA)
        folder(os.path.dirname(features))
        data = [mX, speech_x, noise_x, alpha, outNA]
        cPickle.dump(data, open(features, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        if cnt % 100 == 0:
            print(cnt)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t))

def spectrogram (audio, mode):
    window = 512
    overlap = 256
    hammingWindow = np.hamming(window)
    [f, t, x] = signal.spectral.spectrogram(audio,
                                            window=hammingWindow,
                                            nperseg=window,
                                            noverlap=overlap,
                                            detrend=False,
                                            return_onesided=True,
                                            mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

def packing():
    snr = 0
    concat = 7
    hop = 3

    x = []
    y = []
    cnt = 0
    t = time.time()

    featureDir = os.path.join("workspace1", "features", "spectrogram", dataType, "%ddb" % snr)
    names = os.listdir(featureDir)
    for na in names:
        featurePath = os.path.join(featureDir, na)
        data = cPickle.load(open(featurePath, 'rb'))
        [complexX, speechX, noiseX, alpha, na] = data
        mixedX = np.abs(complexX)

        padding = (concat - 1) / 2
        mixedPad = [mixedX[0:1]] * int(padding) + [mixedX] + [mixedX[-1:]] * int(padding)
        mixedX = np.concatenate(mixedPad, axis=0)
        speechPad = speechX[0:1] * int(padding) + [speechX] + [speechX[-1:]] * int(padding)
        speechX = np.concatenate(speechPad, axis=0)

        mixed3D = mat3d(mixedX, num=concat, hop=hop)
        x.append(mixed3D)
        speech3D = mat3d(speechX, num=concat, hop=hop)
        y.append(speech3D[:, (concat - 1) // 2, :])

        if cnt % 100 == 0:
            print(cnt)
        cnt += 1

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    x = np.log(x + 1e-08).astype(np.float32)
    y = np.log(y + 1e-08).astype(np.float32)

    path = os.path.join("workspace1", "packed_features", "spectrogram", dataType, "%ddb" % snr, "data.h5")
    folder(os.path.dirname(path))
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('x', data=x)
        hf.create_dataset('y', data=y)

    print("Write out to %s" % path)
    print("Pack features finished! %s s" % (time.time() - t,))


def mat3d(x, num, hop):
    lenX, nIn = x.shape
    if(lenX < num):
        x = np.concatenate((x, np.zeros((num - lenX, nIn))))

    lenX =len(x)
    i = 0
    x3d =[]
    while(i + num <= lenX):
        x3d.append(x[i : i +num])
        i += hop
    return np.array(x3d)

def scaler():
    snr = 0
    t = time.time()
    filePath = os.path.join("workspace1", "packed_features", "spectrogram", dataType, "%ddb" % snr, "data.h5")
    with h5py.File(filePath, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)

    (segs, concat, freq) = x.shape
    x2d = x.reshape((segs * concat, freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.scale_)

    path = os.path.join("workspace1", "packed_features", "spectrogram", dataType, "%ddb" % snr, "scaler.p")
    folder(os.path.dirname(path))
    pickle.dump(scaler, open(path, 'wb'))
    print("Save scaler to %s" % path)
    print("Compute scaler finished! %s s" % (time.time() - t))






if __name__ == '__main__':
    x = int(input("What mode would you like? "))
    if x == 1:
        dataType = input("Train or Test? ")
        createMixtures()
    elif x == 2:
        dataType = input("Train or Test? ")
        extractFeatures()
    elif x == 3:
        dataType = input("Train or Test? ")
        packing()
    elif x == 4:
        dataType = input("Train or Test? ")
        scaler()
    elif x == 5:
        CNN.CNN()
    elif x == 6:
        CNN.enhance()
    elif x == 7:
        dataType = input("Train or Test? ")
        createMixtures()
        extractFeatures()
        packing()
        scaler()
    else:
        raise Exception("Enter a correct method")
