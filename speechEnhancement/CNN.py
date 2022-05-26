import pickle

import h5py
import numpy as np
import os
import pickle as cPickle
import time
import main as main
from DataGenerator import DataGenerator
from speech import wave
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

snr = 0

def evalualte(model, gen, x, y):
    predList, yList = [], []

    for(batchX, batchY) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batchX)
        predList.append(pred)
        yList.append(batchY)

    predList = np.concatenate(predList, axis=0)
    yList = np.concatenate(yList, axis=0)
    loss = np.mean(np.abs(predList - yList))
    return loss

def hdf5(path):
    """Load hdf5 data.
    """
    with h5py.File(path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)
        y = np.array(y)
    return x, y

def CNN():
    lr = 1e-4
    batchSize = 500
    t = time.time()

    trainPath = os.path.join("workspace1", "packed_features", "spectrogram", "train", "%ddb" % snr, "data.h5")
    testPath = os.path.join("workspace1", "packed_features", "spectrogram", "test", "%ddb" % snr, "data.h5")
    (trainX, trainY) = hdf5(trainPath)
    (testX, testY) = hdf5(testPath)
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    print("Load data time: %s s" % (time.time() - t))
    print("%d iterations / epoch" % int(trainX.shape[0] / batchSize))

    if True:
        t = time.time()
        sPath = os.path.join("workspace1", "packed_features", "spectrogram", "train", "%ddb" % snr, "scaler.p")
        scaler = pickle.load(open(sPath, 'rb'))

        (segments, concat, frequency) = trainX.shape
        trainX2d = trainX.reshape(segments * concat, frequency)
        trainX2d = scaler.transform(trainX2d)
        trainX = trainX2d.reshape(segments, concat, frequency)
        trainY = scaler.transform(trainY)
        (segments, concat, frequency) = testX.shape
        testX2d = testX.reshape(segments * concat, frequency)
        testX2d = scaler.transform(testX2d)
        testX = testX2d.reshape(segments, concat, frequency)
        testY = scaler.transform(trainY)
        print("Scale data time: %s s" % (time.time() - t))

    (_, concat, frequency) = trainX.shape
    hidden = 2048
    model = Sequential()
    model.add(Flatten(input_shape=(concat, frequency)))
    model.add(Dense(hidden, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(frequency, activation='linear'))
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=lr))

    tr_gen = DataGenerator(batch_size=batchSize, type='train')
    eval_te_gen = DataGenerator(batch_size=batchSize, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batchSize, type='test', te_max_iter=100)


    modelDirectory = os.path.join("workspace1", "models", "%ddb" % snr)
    main.folder(modelDirectory)
    statsDirectory = os.path.join("workspace1", "training_stats", "%ddb" % snr)
    main.folder(statsDirectory)

    iterations = 0
    print(trainX.shape)
    trainLoss = evalualte(model, eval_tr_gen, trainX, trainY)
    testLoss = evalualte(model, eval_tr_gen, testX, testY)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iterations, trainLoss, testLoss))

    stats = {'iterations': iterations,
             'training loss': trainLoss,
             'testing loss': testLoss}
    path = os.path.join(statsDirectory, "%diters.p" % iterations)
    cPickle.dump(stats, open(path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    t = time.time()
    for(batchX, batchY) in tr_gen.generate(xs=[trainX], ys=[trainY]):
        loss = model.train_on_batch(batchX, batchY)
        iterations += 1

        if iterations % 250 == 0:
            trainLoss = evalualte(model, eval_tr_gen, trainX, trainY)
            testLoss = evalualte(model, eval_te_gen, testX, testY)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iterations, trainLoss, testLoss))
            stats = {'iterations': iterations,
                     'training loss': trainLoss,
                     'testing loss': testLoss}
        if iterations % 1000 == 0:
            path = os.path.join(statsDirectory, "%diters.p" % iterations)
            cPickle.dump(stats, open(path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        if iterations % 2500 == 0:
            modelPath = os.path.join(modelDirectory, "md_%diters.h5" % iterations)
            model.save(modelPath)
            print("Saved model to %s" % modelPath)

        if iterations == 10001:
            break
    print("Training time: %s s" % (time.time() - t))

def enhance():
    iterations = 10000
    concats = 7
    sampleRate = 16000
    window = 512
    overlap = 256
    scale = True

    modelPath = os.path.join("workspace1", "models", "%ddb" % snr, "md_%diters.h5" % iterations)
    print(modelPath)
    model = load_model(modelPath)
    scalerPath = os.path.join("workspace1", "packed_features", "spectrogram", "train", "%ddb" % snr, "scaler.p")
    scaler = pickle.load(open(scalerPath, 'rb'))
    directory = os.path.join("workspace1", "features", "spectrogram", "test", "%ddb" % snr)
    names = os.listdir(directory)

    for(cnt, na) in enumerate(names):
        path = os.path.join(directory, na)
        data = cPickle.load(open(path, 'rb'))
        [compleX, speechX, noiseX, alpha, na] = data
        mixedX = np.abs(compleX)
        padding = (concats - 1)/2
        pad = [mixedX[0:1]] * int(padding) + [mixedX] + [mixedX[-1:]] * int(padding)
        mixedX = np.concatenate(pad, axis=0)
        mixedX = np.log(mixedX + 1e-08)
        speechX = np.log(speechX + 1e-08)

        if scale:
            mixedX = scaler.transform(mixedX)
            speechX = scaler.transform(mixedX)

        mixedX3d = main.mat3d(mixedX, num=concats, hop=1)
        prediction = model.predict(mixedX3d)
        print(cnt, na)

        if scale:
            mixedX = mixedX * scaler.scale_[None, :] + scaler.mean_[None, :]
            speechX = speechX * scaler.scale_[None, :] + scaler.mean_[None, :]
            prediction = prediction * scaler.scale_[None, :] + scaler.mean_[None, :]

        prediction = np.exp(prediction)
        signal = wave(prediction, compleX, overlap, np.hamming)
        signal *= np.sqrt((np.hamming(window)**2).sum())

        path = os.path.join("workspace1", "enhanced", "test", "%ddb" % snr, str(iterations), "%s.wav" % na)
        main.folder(os.path.dirname(path))
        main.writeAudio(path, signal, sampleRate)

