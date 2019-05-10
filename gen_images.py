#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from scipy import signal
from librosa import load

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import json


def decibel_filter(spectrogram, db_cutoff=-100.0):
    remove_zeros = np.copy(spectrogram)
    remove_zeros[remove_zeros == 0.0] = np.nan
    inDb = 10.0 * np.log10(remove_zeros)
    inDb[inDb <= db_cutoff] = db_cutoff
    return np.nan_to_num(10.0 ** (inDb / 10.0))


def chunk_preprocess(chunk):
    results = [None] * chunk.shape[0]
    for idx, (_, row) in enumerate(chunk.iterrows()):
        results[idx] = (row["Index"], preprocess(row["Filename"]))
    return results


def preprocess(filename):

    print(f'\nfilename = {filename}\n')

    # The path for p.stem
    p = Path(filename)
 #   print(f"filename: {filename}")
 #   print(f"filename.name = {filename.stem}")
 #   print(f"filename.suffix = {filename.suffix}")
 

    # Generate frequencies and times
    samples, sample_rate = load(
        f"{base}/{p.parent}/{p.stem}.wav", mono=False, sr=44100, res_type="kaiser_fast" #22050
    )
    freq, time, spec = signal.spectrogram(
        samples,
        sample_rate,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=512,
        scaling="spectrum",
    )

    # Filters
    spec = decibel_filter(spec)
    spec = np.log10(spec)
    spec_mean = np.mean(spec)
    spec_std = np.std(spec)
    spec = (spec - spec_mean) / spec_std

    # Lowpass filter
    lowpass = 500 0 #hz
    highest_index = np.abs(freq - lowpass).argmin()
    spec = spec[0:highest_index, :]
    freq = freq[0:highest_index, :]

    # Save spectrogram of the wav file
    scaler = MinMaxScaler(feature_range=(0, 255))
    spec = scaler.fit_transform(spec)
    image = Image.fromarray(np.flip(spec, axis=0))
    image = image.convert("RGB")
    image = image.resize((5152,96)) #10304, 256  notched specto = 5164 x 101, squished to be divisible by 32
    image.save(f"{base}/JPEGImages/{p.stem}.jpg")   

    # Read the corresponding CSV
    inner = pd.read_csv(f"{base}/{p.parent}/{p.stem}.csv")
    
    labels = set()

    with open(f"{base}/labels/{p.stem}.txt", "w") as f:
        for idx, row in inner.iterrows():
#            print(f'row = {row}')

            #image constants
            xmin = np.min(time)
            xmax = np.max(time)
            ymin = np.min(freq)
            ymax = np.max(freq)
            img_width = xmax - xmin
            img_height = ymax - ymin

            #print(f"img_width = {img_width}, img_height = {img_height}")

            #box size
            x_los = json.loads(row["Start Times"])
            x_his = json.loads(row["End Times"])
            y_los = json.loads(row["Low Freqs"])
            y_his = json.loads(row["High Freqs"])
            #print(f'x_los = {x_los}, x_his = {x_his}, y_los = {y_los}, y_his = {y_his}')
            #print(type(x_los))
            #print(type(x_his))

            for x_lo, x_hi, y_lo, y_hi in zip(x_los, x_his, y_los, y_his):
                #print(f'x_lo = {x_lo}, x_hi = {x_hi}, y_lo = {y_lo}, y_hi = {y_hi}')
                
                #find label
                label = 0 #names.iloc[:,0].values.tolist().index(row["Bird"]) # first column of label = class number
                labels.add(row[0])#"bamboo-rat"]) #row["Bird"])
                # print(row["bamboo-rat"])
               
                 #FFT overlap compresses time axis slightly (60s ~> 59.9s) - make sure x min and max are still correct
                if x_lo < xmin:
                    x_lo = xmin
                if x_hi > img_width:
                    x_hi = img_width
                if x_lo > img_width: #super edge case
                   #print("super edge case!")
                   x_lo = x_hi -.01
                if y_lo < ymin:
                    y_lo = ymin
                if y_hi > ymax:
                    y_hi = ymax 
               
                #print(f"x = [{x_lo}, {x_hi}], img_width = {img_width}, y = [{y_lo}, {y_hi}]")
                #find percentages of box dimensions with respect to image:
                x_lo_p = x_lo/img_width
                x_hi_p = x_hi/img_height
                y_lo_p = y_lo/img_height
                y_hi_p = y_hi/img_height
                #print(f"x_p = [{x_lo_p}, {x_hi_p}], y_p = [{y_lo_p}, {y_hi_p}]")
                #print(f"x_lo_p = {x_lo_p}. x_lo = {x_lo}, img_width = {img_width}")
                box_width_p = (x_hi - x_lo)/img_width
                box_height_p = (y_hi - y_lo)/img_height

                #print(f"box_width_p = {box_width_p}, box_height_p = {box_height_p}\n")

                assert x_lo_p + box_width_p <= 1.0 and y_lo_p + box_height_p <= 1.0
                f.write(
                    f"{label} {x_lo_p + 0.5 * box_width_p} {y_lo_p + 0.5 * box_height_p} {box_width_p} {box_height_p}\n"
                )
#    return f"{base}/JPEGImages/{p.stem}.jpg", f"{list(labels)}"
#    return f"{base}/fieldData_jpgs/{p.stem}.jpg", f"{list(labels)}"
    return f"{base}/JPEGImages/{p.stem}.jpg", f"{list(labels)}"


#######################################################################################################
lofreq_px = 0
hifreq_px = 100
img_width_32 = 0
img_height_32 = 0

thisdir = "easy-rats-5s" #"sparse-rats-5s"
base = f"/media/rats/{thisdir}"   # "/media/PNRE/practice"  # "/media/PNRE"  # "/media/PNRE/noisy/short/real"
names = pd.read_csv(f"{base}/{thisdir}.names", header=None)      #  "pnre.names", header=None)

#alldata = pd.read_csv(f"{base}/{thisdir}_scapes.csv")
#print(alldata.columns)

# Fill DF "df" with all scape names and indices
df = pd.read_csv(f"{base}/{thisdir}_wavfiles.txt", header=None)
df.columns= ["Filename"]
df["Index"] = df.index.values

# Create empty DF "results" with indices matching "df"
results = pd.DataFrame(index=df.index.values, columns=["X", "y"], dtype=str)

# Parallelizee - send one filename at a time (chunk) to preproceser, which sends it to be processed)
nprocs = cpu_count()
chunks = np.array_split(df[["Filename", "Index"]], nprocs)
executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(chunk_preprocess, chunk) for chunk in chunks]

for fut in as_completed(futs):
    res = fut.result()
    for idx, (X, y) in res:
        results.loc[idx, "X"] = X
        results.loc[idx, "y"] = y


#this file now kind of already exists from scape gen
results.to_csv(f"{base}/all_files.csv", index=None)
