import torch
import numpy as np
import pandas as pd

npy_file=np.load('image_feat.npy')
npy2_file=np.load('video_feat.npy')
user=pd.read_csv('u_id_mapping.csv')