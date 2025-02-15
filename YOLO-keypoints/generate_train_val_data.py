import numpy as np
from math import ceil
import os
import shutil

# extract data filenames
data_path = './out'
data_fnames = os.listdir(data_path)

# split the data filenames into train and validation sets
def train_val_split_(data:list, val_size:float=None, random_seed:float=None) -> dict:
    """
    Splits the data into training and validation sets acc to val_size

    :return: dictionary of train and validation sets containing file names
    """

    num_data = len(data)
    num_val = int(ceil(num_data * val_size))
    
    np.random.seed(random_seed)
    val = np.random.choice(data, size=num_val, replace=False)
    train = np.setdiff1d(data, val)

    return {'train': train,
            'val': val}
train_val_data = train_val_split_(data_fnames, val_size=0.20, random_seed=21)

# organize the images and labels into folders (following YOLO format)
def organize_data(data:dict, label_src:str, 
                  image_src:str, data_dst:str) -> None:
    """ Organize the data according to YOLO requirements """
    subfolders = ['images', 'labels']
    subsubfolders = ['train', 'val']

    # create folders if not existing
    if not os.path.exists(data_dst):
        os.makedirs(data_dst)
        for sf in subfolders:
            for ssf in subsubfolders:
                os.makedirs(data_dst + f'/{sf}/{ssf}')
    
    else:
        # remove all existing files
        for sf in subfolders:
            for ssf in subsubfolders:
                for filename in os.listdir(data_dst +
                                           f'/{sf}/{ssf}'):
                    file_path = os.path.join(data_dst + f'/{sf}/{ssf}', 
                                             filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        print('Old files in the directory are now deleted.')

        # organize labels
        for key in data.keys():
            for filename in data[key]:
                src_path = os.path.join(label_src, filename)
                dst_path = os.path.join(data_dst+f'/labels/{key}', filename)
                shutil.copy(src_path, dst_path)
        
        # organize images
        for key in data.keys():
            for filename in data[key]:
                filename = str(filename)[:-4] + '.png'

                src_path = os.path.join(image_src, filename)
                dst_path = os.path.join(data_dst+f'/images/{key}', filename)

                shutil.copy2(src_path, dst_path)
        print('New files are uploaded in the directory.')

label_src = './out'
image_src = './Cattle side and back view images/side view'
data_dst = './data'
organize_data(train_val_data, label_src, image_src, data_dst)