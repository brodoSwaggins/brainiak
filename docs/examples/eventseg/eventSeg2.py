import warnings
import sys
import os
import glob
from functools import reduce
import numpy as np
from brainiak.eventseg.event import EventSegment
from scipy.stats import norm
import matplotlib
bcke = matplotlib.get_backend()
from matplotlib import pyplot as plt
import matplotlib.patches as patches
try:
    matplotlib.use("Qt5Agg")
    plt.figure()
    plt.close()
    print("Running interactive backend", matplotlib.get_backend(),".")
except:
    matplotlib.use(bcke)
    print("Can't run interactive backend, run",matplotlib.get_backend(), "instead")
smallsize=14; mediumsize=16; largesize=18
plt.rc('xtick', labelsize=smallsize); plt.rc('ytick', labelsize=smallsize); plt.rc('legend', fontsize=mediumsize)
plt.rc('figure', titlesize=largesize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)
import nilearn as nl
from nilearn import plotting, image, datasets
smallsize=14; mediumsize=16; largesize=18
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
print(f"Datasets are stored in: {datasets.get_data_dirs()!r}")
# plt.rc('xtick', labelsize=smallsize); plt.rc('ytick', labelsize=smallsize); plt.rc('legend', fontsize=mediumsize)
# plt.rc('figure', titlesize=largesize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)
from pathlib import Path
output_dir = Path.cwd() / "images"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
#%% Example use
file = r'C:\Users\izika\OneDrive\Documents\ComDePri\Memory\fMRI data Project\RecallFiles_published\recall_files\sherlock_recall_s1.nii'
print(image.load_img(file).shape)
first_TR = image.index_img(file, 0)
print(first_TR.shape)
plotting.plot_stat_map(first_TR, threshold=1)#, output_file=output_dir / "first_TR.png")
plotting.plot_img(image.smooth_img(first_TR, fwhm=3), threshold=1)
# first_TR.to_filename(output_dir / "first_TR.nii.gz")
#%% Harvard Atlas
atlas = datasets.fetch_atlas_allen_2011()
# The first label correspond to the background
print(f"The atlas contains {len(atlas.labels) - 1} non-overlapping regions")
plotting.plot_img(atlas.maps, title="Harvard-Oxford atlas", colorbar=True)
#%%
masker = NiftiLabelsMasker(labels_img=atlas.maps, labels=atlas.labels, standardize=True)
masker.fit(first_TR)
masker.mask_img_
#%% ********************************************************************************************************************
#***********************************************************************************************************************
# if not os.path.exists(data_path'Sherlock_AG_movie.npy'):
#     !wget https://ndownloader.figshare.com/files/22927253 -O Sherlock_AG_movie.npy
# if not os.path.exists('Sherlock_AG_recall.npy'):
#     !wget https://ndownloader.figshare.com/files/22927256 -O Sherlock_AG_recall.npy
#%%
# Sherlock dataset
data_path = r'/home/itzik/Downloads/Sherlock'
movie = np.load(data_path+'/Sherlock_AG_movie.npy')
recall = np.load(data_path+'/Sherlock_AG_recall.npy')
movie_group = np.mean(movie, axis=0)
print("(subj x TRs x Voxels) = ", movie.shape, recall.shape, movie_group.shape)
#%%

