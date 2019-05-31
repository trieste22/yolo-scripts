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

args = docopt(__doc__,version="gen_dataset version 0.0.1")
try:
	args["--number"] = int(args["--number"])
except ValueError:
	exit("Number must be an integer")

base = pathlib.Path(args["--input_dir"])

# read config file
cfg = pd.read_csv(args["--config"])
fg_files = cfg[cfg["Type"] == "foreground"]
bg_files = cfg[cfg["Type"] == "background"]


# build list of foreground and background clips to draw from
merger = [None]*fg_files.shape[0]
for idx, label in enumerate(fg_files["Label"]):
	merger[idx] = pd.read_csv(f"{base}/foreground_csvs/{label}.csv")
	merger[idx]["label"] = [label]*merger[idx].shape[0]
fg_clips = pd.concat(merger) # contains every foreground clip for desired labels

merger = [None]*bg_files.shape[0]
for idx, label in enumerate(bg_files["Label"]):
	merger[idx] = pd.Series(pathlib.Path(f"{base}/background/{label}").glob("*.wav"))
bg_clips = pd.concat(merger)


# build giant dataframe with specs for each of (n = count) scapes
# interpret Activity to build scapes

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


# NEXT: build scape-defining dataframe (one scape/row):
# - combine splits, choose start times for each call, background?
# split into chunks, send chunks for processing in parallel
# (in parallel: create scapes, images, labels in correct format)