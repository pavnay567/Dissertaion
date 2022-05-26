import json
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = "audio"
data = "data.json"

labels = {
    "Type": [],
    "stft": []
}

frameSize = 2048
hopSize = 512

for i, (path, folder, file) in enumerate(os.walk(dataset)):
    semantic_label = ""
    if path is not dataset:
        semantic_label = path.split("\\")[-1]

        print("\nProcessing: {}".format(semantic_label))

    for f in file:
        print(f)
        file_path = os.path.join(path, f)
        signal, sr = librosa.load(file_path)
        S = abs(librosa.stft(signal, n_fft=frameSize, hop_length=hopSize))

        labels["stft"].append(S.tolist())
        labels["Type"].append(semantic_label)

        with open(data, "w") as fp:
            json.dump(labels, fp, indent=4)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Spectrogram of: ' + f)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()
