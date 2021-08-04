import os

import fftToTriangleV7 as tri

file = "LedZeppelin/DazedAndConfusedm.wav"
waveform, sr = librosa.load(fichier, sr=None)
all_ifft = ifft_par_bande(waveform, RAIES)
analyse_simple(0, 2500, 5000, file, 1, all_ifft, sr, AS, "coco")
  
