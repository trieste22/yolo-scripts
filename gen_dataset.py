#!/usr/bin/env python3
""" gen_dataset.py


Usage:
    gen_dataset.py -c <config.csv> -i <input_dir> [-o <output_dir>] [-f <format>] [-d <duration>] [-n <number>] [-hv]

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version and exit
    -c --config <config.csv>            Specify the configuration
    -i --input_dir <input_dir>          Specify the input directory
    -o --output_dir <output_dir>        Specify the output directory [default: scapes]
    -f --format <format>                Specify the output format [default: tensorflow]
    -d --duration <duration>            Specify the output duration in seconds [default: 5]
    -n --number <number>                Specify the number of scapes to generate [default: 10000]
"""
from docopt import docopt
import pandas as pd
import pathlib
import numpy as np

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import random

import scaper

from scipy import signal
from librosa import load
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


##################################################################################################### DATA-GEN FUNCTIONS

def confirm_integer(key):
	try:
		return int(args[key])
	except ValueError:
		exit(f"{key} must be an integer")



    # fill out each row with background, call times
def place_calls(row): #row: n classes
	bg_path = bg_clips.sample(1).values[0]
	bg_label = str(bg_path.parent).split("/")[-1]
	bg_file = bg_path.name

	call_count = sum(row[1][:-1])
	scape_id = row[1][2]

	empty_data = {
		"Filename": [None]*call_count,
		"Label": [None]*call_count,
		"Start": [0.]*call_count,
		"Duration": [0.]*call_count,
		"LowFreq": [0.]*call_count,
		"HighFreq": [0.]*call_count,
		"Index": [0]*call_count,
	 }

	df = pd.DataFrame(empty_data)

	curr_idx = 0 # call position in empty_data
	labels_with_counts = zip(row[1].index[:-1], row[1].values[:-1]) # ie [('rats-singlebarks', 1), ('farinosas', 1)]

	for label, count in labels_with_counts:
		activity = cfg[cfg["Label"]==label]["Activity"].values[0]

		for i in range(count):
			call = fg_clips[fg_clips["Label"] == label].sample(1)
			df.iloc[curr_idx,0] = call.iloc[0]["Filename"] 
			df.iloc[curr_idx,1] = call.iloc[0]["Label"]
			df.iloc[curr_idx,2] = round(random.uniform(0, args["--duration"]),3)
			df.iloc[curr_idx,3] = call.iloc[0]["Duration"]
			df.iloc[curr_idx,4] = call.iloc[0]["LowFreq"]
			df.iloc[curr_idx,5] = call.iloc[0]["HighFreq"]
			df.iloc[curr_idx,6] = scape_id
	
			curr_idx += 1

	return df
		

##################################################################################################### AUDIO-GEN FUNCTIONS

# build audio file, build image, output records
def build_scape(scape_def):

	# print(scape_def)
	fname = f"{args['--output_dir']}/{'_'.join(np.unique(scape_def['Label'].values))}_{scape_def['Index'].iloc[0]}"
	audiofile = f"{fname}.wav"
	jamsfile = f"{fname}.jams"

	foreground_folder = f"{base}/foreground"
	background_folder = f"{base}/background"
	scape_dur = args["--duration"]
	scape_count = args["--number"]

	sc = scaper.Scaper(scape_dur, foreground_folder, background_folder)
	sc.ref_db = -52 #TODO

	sc.add_background(
		label = ("const", bg_label),
		source_file = ("choose", []),
        source_time = ("uniform", 0, 60-scape_dur) # background files are 1min long for rats as of 2019/05/08.
	)

	for _, call in scape_def.iterrows():
		
		sc.add_event(
			label=('const', call["Label"]),
			source_file=('const', f"{foreground_folder}/{call['Label']}/{call['Filename']}"),
			source_time=('const', 0),
			event_time=('const', call["Start"]),
			event_duration=('const', call["Duration"]),
			snr=('uniform', 1, 10), #TODO: check for new datasets!
			pitch_shift=None,
			time_stretch=None
		)

	sc.generate(
		audiofile, jamsfile,
		allow_repeated_label=True,
		allow_repeated_source=True,
		reverb=0,
		disable_sox_warnings=True,
		no_audio=False
	)

	gen_image(fname)



def run_place_calls(chunk):
	results = [None]*chunk.shape[0]
	for idx, row in enumerate(chunk.iterrows()):
		results[idx] = build_scape(place_calls(row))
	return results



##################################################################################################### IMAGE-GEN FUNCTIONS

def decibel_filter(spectrogram, db_cutoff=-100.0):
    remove_zeros = np.copy(spectrogram)
    remove_zeros[remove_zeros == 0.0] = np.nan
    inDb = 10.0 * np.log10(remove_zeros)
    inDb[inDb <= db_cutoff] = db_cutoff
    return np.nan_to_num(10.0 ** (inDb / 10.0))



def gen_image(fname):

    print(fname)
    # print(pathlib.Path(filename).stem)
    output_dir = f"{pathlib.Path(args['--output_dir']).parent}/JPEGImages" 
    filename = pathlib.Path(fname).stem

    # Generate frequencies and times
    samples, sample_rate = load(
        f"{fname}.wav", mono=False, sr=44100, res_type="kaiser_fast" #22050
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
    lowpass = 10000 #hz
    highest_index = np.abs(freq - lowpass).argmin()
    spec = spec[0:highest_index, :]
    freq = freq[0:highest_index]

    # Save spectrogram of the wav file
    scaler = MinMaxScaler(feature_range=(0, 255))
    spec = scaler.fit_transform(spec)
    image = Image.fromarray(np.flip(spec, axis=0))
    image = image.convert("RGB")
    # image = image.resize((width_px, height_px))
    image.save(f"{output_dir}/{filename}.jpg")   

    return f"{output_dir}/{filename}.jpg"




##################################################################################################### SPECS

args = docopt(__doc__,version="gen_dataset version 0.0.1")
args["--number"] = confirm_integer("--number")
args["--duration"] = confirm_integer("--duration")

base = pathlib.Path(args["--input_dir"])

# read config file
cfg = pd.read_csv(args["--config"])
fg_files = cfg[cfg["Type"] == "foreground"]
bg_files = cfg[cfg["Type"] == "background"]
bg_label = bg_files["Label"].iloc[0]


# build list of foreground and background clips to draw from
merger = [None]*fg_files.shape[0]
for idx, label in enumerate(fg_files["Label"]):
	merger[idx] = pd.read_csv(f"{base}/foreground_csvs/{label}.csv")
	merger[idx]["Label"] = [label]*merger[idx].shape[0]
fg_clips = pd.concat(merger) # contains every foreground clip for desired labels (with boxing/label info)
merger = [None]*bg_files.shape[0]
for idx, label in enumerate(bg_files["Label"]):
	merger[idx] = pd.Series(pathlib.Path(f"{base}/background/{label}").glob("*.wav"))
bg_clips = pd.concat(merger)


clip_density = np.zeros((fg_files.shape[0], args["--number"]), dtype=np.int) # to hold label count
split_clips = np.array_split(clip_density, fg_files.shape[0], axis=1) # initially populate 1/(#classes) present for each class
for i in range(len(split_clips)):
	split_clips[i][i][:] = 1
	split_clips[i] = split_clips[i].transpose()

# build probability overlap list (assuming 2% rats, 30% parrots, 68% none)
probability_labels = [*fg_files["Label"].values, "no-overlap"]
overlap_probs = [.02,.30,.68]
gen_random_overlap = lambda: np.random.choice(probability_labels,args["--number"],p=overlap_probs)
add_extra = lambda: np.random.choice([0,1,2],1,p=[.90, .08, .02])

# insert overlaps
for x in range(len(split_clips)):
	for y, r in zip(range(len(split_clips[x])), gen_random_overlap()):
		if r != "no-overlap":
			label_idx, = np.where(fg_files["Label"].values == r)
			split_clips[x][y][label_idx[0]] += 1 + add_extra()
call_df = pd.DataFrame(np.vstack(split_clips), columns=fg_files["Label"])
call_df["Index"] = call_df.index



##################################################################################################### RUN (PARALLELIZE)

nprocs = cpu_count()
executor = ProcessPoolExecutor(nprocs)
chunks = np.array_split(call_df, nprocs)

futs = [executor.submit(run_place_calls, chunk) for chunk in chunks]
for fut in as_completed(futs):
	_res = fut.result()
