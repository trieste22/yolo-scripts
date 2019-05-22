import pandas as pd

from pathlib import Path
import numpy as np

from scipy import signal
from librosa import load

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed


def decibel_filter(spectrogram, db_cutoff=-100.0):
    remove_zeros = np.copy(spectrogram)
    remove_zeros[remove_zeros == 0.0] = np.nan
    inDb = 10.0 * np.log10(remove_zeros)
    inDb[inDb <= db_cutoff] = db_cutoff
    return np.nan_to_num(10.0 ** (inDb / 10.0))


def chunk_preprocess(chunk):
    results = [None] * chunk.shape[0]
    for idx, (_, row) in enumerate(chunk.iterrows()):
        results[idx] = (row["Index"], preprocess(row["Filename"],width_px, height_px))
    return results


def preprocess(filename, width_px, height_px):

    print(f'\nfilename = {filename}\n')

    # The path for p.stem
    p = Path(filename) 

    # Generate frequencies and times
    samples, sample_rate = load(
        f"{out_dir}/{p.parent}/{p.stem}.wav", mono=False, sr=44100, res_type="kaiser_fast" #22050
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
    lowpass = max_freq #hz
    highest_index = np.abs(freq - lowpass).argmin()
    spec = spec[0:highest_index, :]
    freq = freq[0:highest_index]

    # Save spectrogram of the wav file
    scaler = MinMaxScaler(feature_range=(0, 255))
    spec = scaler.fit_transform(spec)
    image = Image.fromarray(np.flip(spec, axis=0))
    image = image.convert("RGB")
    image = image.resize((width_px, height_px)) #10304, 256  notched specto = 5164 x 101, squished to be divisible by 32
    image.save(f"{out_dir}/JPEGImages/{p.stem}.jpg")   

    return f"{out_dir}/JPEGImages/{p.stem}.jpg"


base_name = "rats_EATO_WOTH" # the name prefixing every scape. Should match output folder name - UPDATE 
source_dir = "/Users/kitzeslab/Desktop/yolo-scripts/scape-gen"
out_dir = f"{source_dir}/{base_name}"
max_freq = 8000 # for lowpass filter in image gen - UPDATE
width_px = 5152 # define image resize - UPDATE 
height_px = 96


# Fill DF "df" with all scape names and indices
df = pd.read_csv(f"{out_dir}/wavfiles.txt", header=None)
df.columns= ["Filename"]
df["Index"] = df.index.values

# Create empty DF "results" with indices matching "df"
results = pd.DataFrame(index=df.index.values, columns=["X", "y"], dtype=str)

# Parallelize - send one filename at a time (chunk) to preproceser, which sends it to be processed)
nprocs = cpu_count()
chunks = np.array_split(df[["Filename", "Index"]], nprocs)
executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(chunk_preprocess, chunk) for chunk in chunks]
