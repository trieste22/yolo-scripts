import pandas as pd
import random
import scaper

from pathlib import Path
import numpy as np

from scipy import signal
from librosa import load

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed



def gen_scapes_labels(sounds_toUse, source_dir, scape_count, base_name, background_label, scape_dur, foreground_dir):
    for i in range(len(sounds_toUse)):
        # read csv of each sound_toUse:
        sounds_list[i] = pd.read_csv(f"{source_dir}/audio/foreground_csvs/{sounds_toUse[i][0]}.csv", header=None)
        sounds_list[i].columns = ["Src_file", "Duration", "Low_freq", "High_freq"]
        # print(sounds_list[i], "\n")

    # print(f"sounds_list[0] = {sounds_list[0]}\n")

    for scape in range(scape_count):
        # set up scape info
        scape_name = f"{base_name}_scape{scape}"
        print(f"scape{scape}:")
        label = ""
        row = f"{source_dir}/{base_name}/JPEGImages/{scape_name}.jpg,\"["

        # set up scaper
        sc = scaper.Scaper(scape_dur, f"{source_dir}/audio/foreground", f"{source_dir}/audio/background")
        sc.ref_db = -52 #TODO

        audiofile = f"{out_dir}/scapes/{scape_name}.wav"
        jamsfile = f"{out_dir}/jams/{scape_name}.jams"

        sc.add_background(label = ("const", background_label),
                        source_file = ("choose", []),
                        source_time = ("uniform", 0, 60-scape_dur)) # background source files are 60 seconds long # TODO

        for i in range(len(sounds_toUse)): # for each type of call
            df = sounds_list[i]
            j = random.randint(sounds_toUse[i][1], sounds_toUse[i][2]) # choose how many calls of this type in the file
            
            if j >0: # for all_files.csv - if there is an instance of this class, add it to the row
                if row.endswith("["):
                    row = f"{row}\'{sounds_toUse[i][3]}\'"
                else:
                    row = f"{row}, \'{sounds_toUse[i][3]}\'"
                
            for k in range(j):
                foreground_label = sounds_toUse[i][0]

                choice = df.sample() # choose a random sound from the list of this call type
                src = choice["Src_file"].iloc[0]

                t = round(random.uniform(0,scape_dur-.25), scape_dur) #start time in file
                dur = round(choice["Duration"].iloc[0], 3)
                end = t + dur
                if(end > scape_dur):
                    end = scape_dur
                    dur = end - t
                lo_freq = choice["Low_freq"].iloc[0]
                hi_freq = choice["High_freq"].iloc[0]

                if (hi_freq > max_freq): # just in case
                    hi_freq = max_freq

                YOLO_class = i
                xCenter_percent = round((end + t)/(2*scape_dur),6)
                yCenter_percent = round((hi_freq + lo_freq)/(2*max_freq),6)

                width_percent = round((end-t)/scape_dur, 6)
                height_percent = round((hi_freq - lo_freq)/max_freq, 6)

                if (label == ""):
                    label = f"{YOLO_class} {xCenter_percent} {yCenter_percent} {width_percent} {height_percent}"    

                else:
                    label = f"{label}\n{YOLO_class} {xCenter_percent} {yCenter_percent} {width_percent} {height_percent}"

                sc.add_event(label=("const", sounds_toUse[i][0]),
                        source_file = ("const", f"{foreground_dir}/{foreground_label}/{src}"),
                        source_time = ("const", 0),
                        event_time = ("const",t),
                        event_duration = ("const", dur), # might get warnings
                        snr = ("uniform", 5, 10), #-10, 6), #TODO: this always needs tested in case something is different about the foreground files
                        pitch_shift = None,
                        time_stretch = None )

        # save labels to .txt file
        with open(f"{out_dir}/labels/{scape_name}.txt", "w") as text_file:
            text_file.write(label)
#         print(label)

        # add junk sounds, if any
        if (junk != None):
            junk_count = random.randint(junk[1], junk[2])    

            for j in range(junk_count):
                sc.add_event(label=("const", junk[0]),
                        source_file = ("choose", []),
                        source_time = ("const", 0),
                        event_time = ("const",t),
                        event_duration = ("const", dur), # might get warnings
                        snr = ("uniform", 10, 20), #-10, 6), #TODO: this always needs tested in case something is different about the foreground files
                        pitch_shift = None,
                        time_stretch = None )

        sc.generate(audiofile,jamsfile,
                       allow_repeated_label=True,
                       allow_repeated_source=True,
                       reverb=0,
                       disable_sox_warnings=True,
                       no_audio=False)
        row = f"{row}]\""
        all_files[scape+1] = row
            

#### SETUP ####

base_name = "rats_EATO_WOTH" # the name prefixing every scape. Should match output folder name - UPDATE 
scape_count = 5 # UPDATE
scape_dur = 5 # UPDATE

# UPDATE: update source_dir, confirm directory structure inside source_dir matches expected (audio/foreground, audio/background, audio/foreground_csvs)
#       note: there should be a matching .csv file in audio/foreground_csvs for each foreground folder, containing clip length & freq information to build boxes from
#       TODO: store/read background file length information instead of assuming 1-minute files

source_dir = "/Users/kitzeslab/Desktop/yolo-scripts/scape-gen"
out_dir = f"{source_dir}/{base_name}"
background_dir = f"{source_dir}/audio/background"
background_label = "norats-nofarinosas"
foreground_dir = f"{source_dir}/audio/foreground"
junk = None #("junk-easy", 0, 2) # junk = None if none

max_freq = 8000 # for lowpass filter in image gen - UPDATE
width_px = 5152 # define image resize - UPDATE 
height_px = 96

# prep empty variables to fill
all_files = [None]*(scape_count+1)
all_files[0] = "X,y" # to output to all_files.csv
count = 0 # current label number, as reference index for place in labels

# UPDATE: fill sounds_toUse with desired scape properties
# sounds_toUse = list of tuples containing which sound categories to include in this dataset, and how many to include
#    TODO: add probability option?
sounds_toUse = [("rats-fewer-singlebarks", 1, 2, "bamboo-rat"), ("EATO", 0, 1, "EATO"), ("WOTH", 0, 2, "WOTH")] # foldername in foreground, min/scape, max/scape, class name
sounds_list = [None]*len(sounds_toUse)


#### RUNNING ####

gen_scapes_labels(sounds_toUse, source_dir, scape_count, base_name, background_label, scape_dur, foreground_dir)
np.savetxt(f"{out_dir}/all_files.csv", all_files, fmt='%s')