#!/usr/bin/python3

import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv("all_files.csv") #file created by gen_specs.py containing files with list of contained classes
X = df["X"].values
y = df["y"].apply(lambda x: ast.literal_eval(x))

binarize = MultiLabelBinarizer()
y = binarize.fit_transform(y) #fit and transform

#sss = StratifiedShuffleSplit(test_size=.3)
#train, test = sss.split(X,y)
X_train, X_validate, y_train, y_test = train_test_split(X,y, test_size=.1)

with open("train.txt", "w") as f:
    for line in X_train:
        f.write(f"{line}\n")
with open("validate.txt", "w") as f:
    for line in X_validate:
        f.write(f"{line}\n")


