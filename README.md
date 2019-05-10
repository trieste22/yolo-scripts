# yolo-scripts
## Trieste Devlin, 2019

This repo contains scripts used to generate, prepare, and annotate audio/visual data for training YOLO models using Darknet. Below is a working guide for use.

# Spawn a new machine

To spawn a new machine, log in via `portal.azure.com`. You will see the dashboard, which shows all the machines you've made. From the menu on the left, select Virtual Machines > Add 

Set the VM up with the following settings:

# Basics
* Subscription: Use the subscription that relates to what you're doing, Deep Learning or more standard OpenSoundcsape analysis
* Resource group: Where the money to run this VM is pulled from.
* Virtual machine name: name it helpfully
* Region: select the closest geographic region (e.g. East US). No infrastructure redundancy is required.
* Image: For now, use Ubuntu 18.04 image, but eventually we will create our own image. To use that, select Browse all images.
* Size: select based on your needs RE: RAM, etc.
  * for YOLO ML I've been using Standard NC6 - 6vcpus, 56 GB memory
* Authentication type: use an SSH key.
* Username: choose the username you want
* SSH Public key: paste your public key
  * Create an SSH key on the computer from which you want to SSH:
    * Run ```ssh-keygen``` and make a key especially for Azure, ie: `~/.ssh/azure` (might need absolute path, ie /home/trieste/.ssh/azure)
    * Then copy and paste public key into the VM slot: `cat ~/.ssh/azure.pub`
* Public inbound ports: none!

# Disks
* OS disk size: this is where the operating system is installed. If you need to install a lot of packages on the machine, get a bigger disk.
* OS disk type: standard seems fine
* Data disks: 
  * If you need an additional TB of data, create a new data disk - note that these disks will cost money for storage even if the VM is shut down & we're not doing any processing on them
  * If you want to process previously known data, attach existing disk

# Networking
* Virtual Network: select our private network, Opensoundscape--it will act as a local network for our machines
From the Dashboard select the VM name > St VM

# Diagnostics
* Turn boot diagnostics off

# SSH into machine from your current machine
Create a new `~/.ssh/config` with the following contents

```
Host <the hostname you want to use>
     HostName <Public IP Address listed on Azure>
     User <the username you picked>
     IdentityFile /Users/<your name on robin>/ssh/azure
```

* Find ip address `ifconfig` and use `en0`

* Add port: Source submenu > IP Addresses > Add inbound security rule

```
Source IP addresses: <your IP address>/32
Source port ranges: *
Destination: any
Destination port ranges: *
Protocol TCP
Priority 100
Name: Ports_Robin
```

* Now ssh to the hostname you wrote in the config above: `ssh <hostname>`.


# Prepare machine

* make a dev account to download:..
  * CUDA 10.0: https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
  * (run 'lscpu' to print info about CPU architecture)
  * cuDNN for CUDA10.0: https://developer.nvidia.com/rdp/cudnn-download (I'm trying the cuDNN Library for Linux, not sure if it's the right one)
* on the VM:

```
sudo apt update
sudo apt upgrade
sudo apt install make gcc
```
* reboot the VM (online portal)

```
lsblk
```
* This will display the disks and partitions. If there's no sdc, you need to add a disk (online portal)
  * says Barry: " I would add a Standard SSD of size 1023. It gives you a lot of flexibility and the IOPs are fine for our use case (Input/Output Per second).'
* Mount disk, upload data to the mounted data disk, run CUDA install:

```
#on VM:
sudo mkfs.ext4 /dev/sdc1
mkdir /media #or whatever
mount /dev/sdc1 /media/

#on LOCAL:
scp -i ~/.ssh/azure ~/Downloads/cuda_10.0.130_410.48_linux.run username@(VM's public ip address):~/media
scp -i ~/.ssh/azure ~/Downloads/cudnn-10.0-linux-x64-v7.5.0.56.tgz username@(VM's public ip address):~/media

#on VM:
sudo bash cuda_10.0.130_410.48_linux.run

#(type fff to scroll faster through the User Licence Agreement)
#answer yes/no/quit as per defaults
```

* add /usr/local/cuda-10.0/bin to path, as indicated after CUDA install
* TODO: LD_LIBRARY_PATH includes /usr/local/cuda-10.0/lib64, or, add /usr/local/cuda-10.0/lib64 to /etc/ld.so.conf and run ldconfig as root

```
#on VM:
#tar -zcvf archive_name.tar.gz directory_to_compress.
tar xvf cudnn-10.0-linux-x64-v7.5.0.56.tgz  # untar cuDNN
copy the `include/cudnn.h` to `/usr/local/cuda-10.0/include` # These will make Darknet faster as well.
copy the `lib64/libcu*` to `/usr/local/cuda-10.0/lib64`
```

* After installing CUDA/Driver and cuDNN reboot the machine a second time.
* Then `nvidia-smi` should print `410.x` and a bunch of other stuff.

* Install required python libraries:
 * apt install python3-opencv (pip3 install opencv-python is no good! You'll get a seg fault running gen_anchors)

* compress and transfer data:

```
# if necessary: for FILENAME in *; do mv $FILENAME Unix_$FILENAME; done
#on local (this will take a while, depending)
tar -zcvf pnre_scapes.tar.gz path/to/PNRE_scapes
scp -i ~/.ssh/azure pnre_scapes.tar.gz trieste@<VM public ip>:~/media/
```

# Darknet directory structure:

* root = /media/PNRE (mounted to sdc1 - the data will stay there, but you have to mount every time you turn on the machine)
  * root/scapes - wav and associated csv files
  * root/JPEGImages (start empty, filled by gen_scapes)
  * root/labels (start empty, filled by gen_scapes)
  * root/backup (starts empty, filled with weights from training runs)
  * root/darknet - the git clone
  * pnre.data pnre.names train.txt validate.txt wavfiles.txt - in root (where training is run)


# What to run to prep for YOLO (ordered, unless otherwise noted):

* NOTE: on VM, run everything with python3 / pip3 etc so don't end up going halvsies with python2
* NOTE: I'm running everything from /media/PNRE
* Create scapes (mine are in /media/PNRE/scapes #todo: parallelize this. Should be straightforward.
* mkdir labels JPEGImages backups
* create data descriptor files:
  * ls scapes/\*.wav > wavfiles.txt #list of all wav files in one place
  * pnre.names: one class name / line
  * pnre.data: contains info about where things are:
```
classes = 22
train = /media/PNRE/train.txt
valid = /media/PNRE/validate.txt
names = /media/PNRE/pnre.names
backup = /media/PNRE/backup
```
* run gen_scapes.py
  * update paths & check filenames @ bottom of file and in the f.write() functions
  * labels/ and JPEGImages/ will be populated, and all_files.csv will be created containing the filename and associated classes
  * labels are in the format < object-class > < x > < y > < width > < height >, with the class represented as a number, and all measurements in terms of percent of the entire image. x and y are the center coordinates.
  * since the FFT squishes the audio a tiny bit due to overlap, so there are some edge cases where the box as defined by scaper overlaps with the end of the image, and some super edge cases where the /start/ of the box is actually after the end of the image. My solution for now (as of 4/18/19) was to fudge it. So there will be some teensy tiny boxes at the end of the image. Prob better to delete those labels outright moving forward. **#TODO.**
 
```
 sudo apt install imagemagick
 identify JPEGImages/scape0.jpg #print out image size (to see how much they're squished by)
 add to ~/.bashrc: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH  #"look for this nonstandard directory in the right place"
```
 
 * generate anchors:
   * this will generate 9 pairs of numbers, print them, and save them in a file in the designated folder.
   * I ran this ~10 times and chose the set with the highest average IOU (all mine were about 85%) to put in the config
   * I had quite a few failures upon running this script. **#TODO: figure out why.**
 ```
 python3 gen_anchors.py -filelist /path/to/train.txt -output_dir /path/to/anchors -num_clusters 9
 ```
  * generate train/test sets:
    * gen_trainTest.py (after gen_scapes is done, because it's creating the all_files.csv which contains the file (X. It's capitalized, guys) and clas list (y, lowercase, obv) associations. gen_trainTest just grabs these out. In the future we might want to do a stratifiedShuffleSplit instead of a smiple train_test_split, to get a verified good equal balance of classes in train and test. TTS should be ok for now with 10,000 files though. **#TODO.**
  * correct config file for this model:
    * follow instructions from: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects ) 
  * Start training!
    * running in a tmux panel = protection from getting cut off from VM
    * Don't forget log output so you can plot loss!!
  ```
  tmux
  ./darknet/darknet detector train pnre.data cfg/pnre.cfg [optional: backup/logged.weights] | tee training.log
  ```
    * some NaNs ok (I'm getting them for every single Region 106, but loss is consistently going down so gonig with it)
  * apply model to test image:
    * path to alphabet to print labels on prediction output is an incorrect absolute path:
    * adjust darknet/src/image.c line 247: change to correct path -> ‘make’ from darknet root
    * VM might need _sudo apt-get install build-essential g++_

  ```
  #testing on a single file. Output prediction with labeled boxes saved to predictions.jpg
  ./darknet/darknet detector test pnre.data cfg/pnre.cfg backup/pnre_last.weights JPEGImages/scape123.jpg (-thresh .44)
  
  #testing on whole validation (test) set using VG's fork:https://github.com/vincentgong7/VG_AlexeyAB_darknet
  #still needs re-make'd with correct path in image.c for correct boxing
  ./darknet/darknet detector batch pnre.data pnre.cfg backup/pnre_last.weights batch fieldData_jpgs/ fieldData_predicts_betterTrained/ > betterTrained_results.txt
  ```
  
  # Data Preparation & lab-side prep for training:
  * For the model to learn to pick out species of interest, you will need to create many fake, well-annotated soundscapes to feed into it as training data. Some of these will be used for training and some for testing. The model needs to know exactly when every species of interest occurs in the file, and what frequency that vocalization occurs at in every train and test file. Once the model is trained, you will be able to feed it new un-annotated data from the field, and it will spit out annotations for that data.
  * We've been creating ~8-10,000 files for minute-long scapes, and up to 25,000 files for 15-second scapes. These are split randomly into "Train" and "Test" groups later.
  * A scape consists of a background file with some number of foreground sounds superimposed on top. The foreground sounds should include both vocalization types that the model needs to learn to identify and 'junk' noises that it should learn to ignore (like microphone pops, wind sounds, human speech, common species that aren't of interest, or confusion species that sounds similar to the one you're actually looking for).
  
  ## To create scapes:
  * Choose a set of background files. The more, the better. I've found that the most believable scapes are made using actual background clips from the place I'm emulatating. The scape generation code runs a lot faster if you are able to give it background clips with a good amount of noise already in them, that way you have to add fewer 'junk' sounds. Make sure the background clips don't have any vocalizations of species that you're training the model to find. Cut each background track to be greater than or equal to the length of your final scapes. I recommend 15 seconds for YOLO. Place background files in a folder called 'background'.
  * From field data, cut specific foreground calls of the species (singular or plural) you're looking to train on and save them each in their own well-named file. Each foreground clip should be cut so that there's not any dead space and noise reduced as well as you can so that it only contains a very clean vocalization and nothing else (I use Audacity to edit audio, and Export to .wav).
  * Record the filename (ie "EATO_song1_WV_summer.wav"), class name ("Eastern towhee Song"), duration in seconds (ie 3.141), and low and high frequency of the vocalization in the file in Hz (1000, 15500).
  * Save all of the details about each file in a big CSV (or Excel or Google Sheets) file called "all_calls.csv"
  * It should look like the example attached. **#TODO** - attach example
    * Note that you can (and should) have more than one foreground file per species and call type. In that case, you'd have multiple rows, each with a different filename but the same class name
    * "Class", in this case, refers to a category of vocalization that you'd like the model to ID in the annotated output. It's probably safer to make multiple classes for disimmilar vocalizations by the same species (ie, you might have three classes for Eastern Towhees to be able to correctly pinpoint its two kinds of song and one kind of call). For each class type, try to prepare at least five foreground files as a template to build into the scapes. **#TODO** - number
  * 
    
    
1) create scapes
2) create wavfiles.txt
3) create the .names & .data files, and empty folders JPEGImages, labels, backup
4) gen_images (previously gen_scapes)
  * to fill JPGEGImages & labels
  * TODO: fix what gets output to all_files.csv for next step..
  * confirm visually with a few files that the JPGs and labels look correct
5) gen_trainTest
  * (based on all_files.csv)
6) gen_anchors ~ 10 times, choose best percentage output to copy to .cfg
  * make sure to change image size to match actual - line 15-16!!
  * ctl-c out if you get nans
7) set up .cfg:
  * This round (sparse-rats-5s), I updated width and height as well. This might be wrong!!

 
