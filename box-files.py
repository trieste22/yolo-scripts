import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Plot all labeled boxes specified in a single labelfile in copy of the original jpgimg created in output_dir
def plotBoxes(jpgimg, labelfile, output_dir, count):
    #print(f"{jpgimg}, {labelfile}, to {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    head, filename = os.path.split(jpgimg)
    name= os.path.splitext(filename)[0]    
    
    im = np.array(Image.open(jpgimg), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    imy = im.shape[0]
    imx = im.shape[1]

    label = pd.read_csv(labelfile, sep=' ', header=None)
    label.columns = [['Class', 'x_center', 'y_center', 'width', 'height']]

    # print(f"imx: {imx}, imy: {imy}")
    # print(label)
    # print(label['x_center'].iloc[0])

    x_center_percents = label['x_center'].values
    y_center_percents = 1 - label['y_center'].values
    width_percents = label['width'].values
    height_percents = label['height'].values
    
    for i in range(len(x_center_percents)):
#         label_box_percents = [float(label['x_center'].iloc[i]), float(label['y_center'].iloc[i]), float(label['width'].iloc[i]), float(label['height'].iloc[i])]
        # print(label_box_percents)

        y_topleft = (y_center_percents[i] - height_percents[i]/2)*imy
        x_topleft = (x_center_percents[i] - width_percents[i]/2)*imx
        
        #print(f"(xtopleft, ytopleft) = ({x_topleft}, {y_topleft})")

        # print(f"x_topleft = {x_topleft}, y_topleft = {y_topleft}")

        # Create a Rectangle patch - practice
        #box = [234, 280, 349, 411] #xlo, xhi, ylo, yhi (in px, from top left)
        # rect = patches.Rectangle((box[0],box[2],),(box[1]-box[0]),(box[3]-box[2]),linewidth=1,edgecolor='r',facecolor='none')
        # Rectangle((x,y) = top left, width, height, angle=0.0 = rotation, **kwargs)

        label_rect = patches.Rectangle((x_topleft, y_topleft), width_percents[i]*imx, height_percents[i]*imy, linewidth = .5, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

        # Add the patch to the Axes
        ax.add_patch(label_rect)


#    plt.show()
    fig.savefig(f"{output_dir}/{name}_boxed.jpg", dpi=450)
    if(count > 19):
        plt.close('all')
        count = 0
        print("closing figs!\n")
    return (count + 1)



# Visualize all labels for contents of src_dir
#   For every .jpg image in src_dir that has a corresponding .txt label file,
#   Create a copy of that jpg with boxed labels added, and save it in output_dir
def box_directory_contents(src_dir, output_dir, label_dir):
    count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = os.listdir(path=src_dir)
    
    for file in files:
        head, filename = os.path.split(file)
#        print(file)
        
        parts= os.path.splitext(file)
        name = parts[0]
        ext = parts[1]
#         print(name, ext)
        if not name.startswith(".") and (ext == ".jpg"):
            labelfile = f"{label_dir}/{name}.txt"
            if os.path.isfile(labelfile):
                count = plotBoxes(f"{src_dir}/{file}", labelfile, output_dir, count)            

                
#box_directory_contents("sampleData", "boxed")


# input directory: python box-files.py jpgname labelname output_dir
if len(sys.argv) == 4:
    print("input directory\n")
    box_directory_contents(sys.argv[1], sys.argv[2], sys.argv[3])
    
# input single file: python box-files.py src_dir output_dir    
if len(sys.argv) == 5:
    print("input file\n")
    plotBoxes(sys.argv[1], sys.argv[2], sys.argv[3], 0)
    
plt.close('all')