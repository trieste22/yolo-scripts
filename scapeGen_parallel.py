import scaper
import numpy as np
import os
import random
from datetime import datetime
import jams
import pandas as pd

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

# Build data for a single scapered file.
# Choose start times ensuring separation between calls (rat barks don't overlap)
# output dictionary of data for reference to build scapes and annotation csvs from

def build_scapeData(barks_df, curr_scape, max_calls_per_file, scape_dur, basename): #call_list without probs yet, n = max calls/file
    
    num_barks = random.randint(1,max_calls_per_file) #choose how many barks in this file
    
    sources = [None]* num_barks
    starts = [None]*num_barks
    ends = [None]* num_barks
    low_fs = [None]* num_barks
    high_fs = [None]* num_barks
    
    for i in range(num_barks):
        t = round(random.uniform(0,scape_dur-.1), 3) #start time in file
        b = barks_df.sample() #grab a random bark source file

        sources[i] = b['Source File'].iloc[0]
        starts[i] = t
        ends[i] = round(t + b['Length'].iloc[0],3)
        low_fs[i] = b['Low Freq'].iloc[0]
        high_fs[i] = b['High Freq'].iloc[0]
    
        if ends[i] > scape_dur: #make sure end time is correct on edge case
            ends[i] = scape_dur
    
#     scapedeets = pd.Series([sources, starts, ends, low_fs, high_fs])
    thisscape = {'Source Files' : sources,
                'Scape Name' : f'{basename}_scape{curr_scape}',
                'Start Times' : starts,
                'End Times' : ends,
                'Low Freqs' : low_fs,
                'High Freqs' : high_fs
                }
    return(thisscape)


# build a scape and corresponding csv as described by the dictionary thisscape in the following format:
# thisscape = dict with boxing and file info for vocalizations to include. curr = which scape. outfile = path/to/filename

def build_scape(thisscape, outdir, scape_dur, sourcedir, bg_label, fg_label, junk_label):
    # print(thisscape)
    
    sc = scaper.Scaper(scape_dur, f"{sourcedir}/foreground", f"{sourcedir}/background")
    sc.ref_db = -52 #TODO

    fname = thisscape['Scape Name']
    audiofile = f"{outdir}/{fname}.wav"
    jamsfile = f"{outdir}/{fname}.jams"
    
    # print(f"fname: {fname}, audiofile: {audiofile}, jamsfile: {jamsfile}")
    
    sc.add_background(label = ("const", bg_label),
                    source_file = ("choose", []),
                    source_time = ("uniform", 0, 60-scape_dur)) # background files are 1min long for rats as of 2019/05/08.

    for i in range(len(thisscape['Start Times'])): #add each planned vocalization
        sc.add_event(label=('const', fg_label),
                    source_file = ('const', f'{sourcedir}/foreground/{fg_label}/' + thisscape['Source Files'][i]),
                    source_time = ('const', 0),
                    event_time = ('const', thisscape['Start Times'][i]),
                    event_duration = ('const', thisscape['End Times'][i] - thisscape['Start Times'][i]),
                    snr = ('uniform', 14, 20), #-10, 6), #TODO: this always needs tested in case something is different about the foreground files
                    pitch_shift = None,
                    time_stretch = None )
        
#    num_junk = random.randint(0,2) #2 for 5s rats, 10 for easyjunk pnre, 5 for shortjunk pnre
#    for j in range(num_junk):
#        sc.add_event(label=('const', junk_label),
#                    source_file = ('choose', []),
#                    source_time = ('const', 0),
#                    event_time = ('uniform', 0, scape_dur-.5),
#                    event_duration = ('const', 5), # TODO: get length so don't have to deal with the warnings.
 #                   snr = ('uniform',-5,2),
 #                   pitch_shift=('normal', -.5,.5),
 #                   time_stretch=('uniform',.5,2))

                
    sc.generate(audiofile,jamsfile,
                   allow_repeated_label=True,
                   allow_repeated_source=True,
                   reverb=0,
                   disable_sox_warnings=True,
                   no_audio=False)
    
    df = pd.DataFrame(thisscape)
    df = df.transpose()
    df.to_csv(f'{outdir}/{fname}.csv')

    print(f"Scape {fname} generated.")
                




################################################################################################################
# The below variables are all that need changed before running.
# Barry:
#   2) Run with '-W ignore' if you don't want a scaper error every time it has to cut down a piece of
#       audio when it runs past the end of the scape.
#   3) After the code completes, my terminal goes weird and I have to kill the tab and reopen
#       This started happening when I implemented the parallel code a while back, I don't know why.

#sourcedir is the directory where the script is run from,
#   and where file 'singlebarkspecs.csv' and directories 'foreground' and 'background' are located
sourcedir = "/media/PNRE/noisy/short/rats/makin-scapes"#sparse-rats-scapering"
outdir = "easy-rats-5s-same_scapes"
scape_dur = 5
scapecount = 1  # how many scapes to make
bg_label = "norats-nofarinosas" # name of the subdirectory within 'background' with desired backgrounds
fg_label = "single-barks" # name of the subdirectory within 'foreground' with desired vocalizations
junk_label = "junk" # name of the subdirectory within 'foreground' with junk sounds to add
basename = "easy-rats-5s-same" # scapes will be names {basename}_scape{index}.wav (with corresponding .csv and .jams)
max_calls_per_file = 3 # insert between one and three rat barks into each scape

if not os.path.exists(f'{sourcedir}/{outdir}'):
    print(f'making {outdir}')
    os.makedirs(f'{sourcedir}/{outdir}')

start = datetime.now()

barks_df = pd.read_csv(f'{sourcedir}/singlebarkspecs.csv', header=None)
barks_df.columns = ['Source File', 'Length', 'Low Freq', 'High Freq']

allthescapes = [None]*scapecount


for i in range(scapecount):
    thisscape_dict = build_scapeData(barks_df, i, max_calls_per_file, scape_dur, basename)
    allthescapes[i] = thisscape_dict
    
scapes = pd.DataFrame(allthescapes) # probably unnecessary but I didn't feel like changing the next few lines so I just converted it again
scapes.columns = ['End Times', 'High Freqs', 'Low Freqs', 'Scape Name', 'Source Files', 'Start Times']
    
nprocs = cpu_count()
executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(build_scape, row, outdir, scape_dur, sourcedir, bg_label, fg_label, junk_label) for idx, row in scapes.iterrows()]
for fut in as_completed(futs):
    _ = fut.result()
    
# build_scapes(outdir, scape_dur, scapecount, sourcedir, bg_label, junk_label, basename)
print("Done! Completed in " + str(datetime.now()-start) + " seconds.\n")
