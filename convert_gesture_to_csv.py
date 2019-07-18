#converting gesture to csv format
import os
os.chdir('C:\\Users\\user')
from scipy.misc import imread
import numpy as np
import pandas as pd
import os
root = './gestures4' # or ‘./test’ depending on for which the CSV is being created

# go through each directory in the root folder given above
for directory, subdirectories, files in os.walk(root):
# go through each file in that directory
    for file in files:
    # read the image file and extract its pixels
        print(file)
        im = imread(os.path.join(directory,file))
        value = im.flatten()
# I renamed the folders containing digits to the contained digit itself. For example, digit_0 folder was renamed to 0.
# so taking the 13th value of the folder gave the digit (i.e. "./gestures4/8" ==> 13th value is 8), which was inserted into the first column of the dataset.
        value = np.hstack((directory[12:],value))
        df = pd.DataFrame(value).T
        df = df.sample(frac=1) # shuffle the dataset
        with open('train_foo16.csv', 'a') as dataset:
            df.to_csv(dataset, header=False, index=False)
