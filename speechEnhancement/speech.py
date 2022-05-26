import numpy as np
import numpy
import decimal

def wave (pdX, gtX, overlap, function, length=None):
    x = pdX * np.exp(1j * np.angle(gtX))
    x = np.concatenate((x, np.fliplr(np.conj(x[:, 1:-1]))), axis=1)
    frame = np.real(np.fft.ifft(x))
    (frames, windows) = frame.shape

    frameLength = int(decimal.Decimal(windows).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    frameSteps = int(decimal.Decimal(windows-overlap).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    assert numpy.shape(frame)[1] == frameLength

    indices = numpy.tile(numpy.arange(0, frameLength), (frames, 1)) + numpy.tile(numpy.arange(0, frames*frameSteps, frameSteps), (frameLength,1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padding = (frames-1)*frameSteps + frameLength

    signalLength = 0
    if signalLength <= 0:
        signalLength = padding
    signal = numpy.zeros(padding,)
    windowCorrection = numpy.zeros(padding,)
    window = function(frameLength)

    for i in range(0, frames):
        windowCorrection[indices[i,:]] = windowCorrection[indices[i,:]] + window + 1e-15
        signal[indices[i,:]] = signal[indices[i,:]] + frame[i,:]
    signal = (signal/windowCorrection)[0:signalLength]
    if length:
        signal = pad_or_trunc(signal, length)

    return signal

def pad_or_trunc(s, wav_len):
    if len(s) >= wav_len:
        s = s[0 : wav_len]
    else:
        s = np.concatenate((s, np.zeros(wav_len - len(s))))
    return s