import sys
from functools import reduce
import numpy as np
import matplotlib
bcke = matplotlib.get_backend()
from matplotlib import pyplot as plt
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
import pandas as pd
import nilearn as nl
from nilearn import plotting, image, datasets, masking
from nilearn.maskers import NiftiLabelsMasker
from brainiak.eventseg.event import EventSegment
print(f"Datasets are stored in: {datasets.get_data_dirs()!r}")
from pathlib import Path; output_dir = Path.cwd() / "images"
output_dir.mkdir(exist_ok=True, parents=True); print(f"Output will be saved to: {output_dir}")
#%%
def getBoolMask(atlas, labels, map=None):
    label_index = [atlas.labels.index(l) for l in labels]
    if map is None:
        map = atlas.maps
    bool_mask = reduce(lambda x, y: x + y, [(map.get_fdata() == i) for i in label_index])
    mask_img = nl.image.new_img_like(map, bool_mask)
    return mask_img, bool_mask
#%% Example use - single subject
if sys.platform == 'linux':
    file = r'/home/itzik/Desktop/EventBoundaries/recall_files/sherlock_recall_s1.nii'
    blck = True
else:
    file = r'C:\Users\izika\OneDrive\Documents\ComDePri\Memory\fMRI data Project\RecallFiles_published\recall_files\sherlock_recall_s1.nii'
    blck = False
all_TR = image.load_img(file)
print(all_TR.shape)
first_TR = image.index_img(file, 0)
print(first_TR.shape)
plotting.plot_stat_map(first_TR, threshold=1)#, output_file=output_dir / "first_TR.png")
plotting.plot_img(image.smooth_img(first_TR, fwhm=3), threshold=1); plt.show(block=blck)
# first_TR.to_filename(output_dir / "first_TR.nii.gz")
#%% Extract hippocampus using Harvard-Oxford atlas fitted to 3 mm MNI152 template
atlas_HarvOx = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-2mm")
mm3_maps = image.resample_to_img(atlas_HarvOx.maps, all_TR, interpolation="nearest")
plotting.plot_img(atlas_HarvOx.maps, title="Harvard-Oxford atlas", colorbar=True); plt.show(block=blck)
#%%
label = ['Right Hippocampus', 'Left Hippocampus'] ; label_index = [atlas_HarvOx.labels.index(l) for l in label]
mask_img, bool_mask = getBoolMask(atlas_HarvOx, label, mm3_maps)
# bool_mask = reduce(lambda x, y: x + y, [(mm3_maps.get_fdata() == i) for i in label_index])
# mask_img = nl.image.new_img_like(mm3_maps, bool_mask)
print(label, "mask // Shape:", bool_mask.shape, ", # voxels: ", np.sum(bool_mask))
#%%
hippocampi_HarvOX = masking.apply_mask([all_TR], mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)
hippocampi_HarvOX.shape # TRs x voxels
plotting.plot_roi(mask_img, title='hippocampi, Harvard-Oxford ({} voxels)'.format(np.sum(bool_mask)), display_mode='tiled', draw_cross=False); plt.show(block=blck)
#%% Extract hippocampus using Juelich (based on Hahamy)
atlas_juelich = datasets.fetch_atlas_juelich("maxprob-thr0-2mm")
mm3_maps = image.resample_to_img(atlas_juelich.maps, all_TR, interpolation="nearest")
#%%
label_index = [atlas_juelich.labels.index(l) for l in atlas_juelich.labels if 'hippocampus' in l.lower()]
bool_mask = reduce(lambda x, y: x + y, [(mm3_maps.get_fdata() == i) for i in label_index])
mask_img = nl.image.new_img_like(mm3_maps, bool_mask)
print(label, "mask // Shape:", bool_mask.shape, ", # voxels: ", np.sum(bool_mask))
hippocampi_juelich = masking.apply_mask([all_TR], mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)
hippocampi_juelich.shape # TRs x voxels
plotting.plot_roi(mask_img, title='hippocampi, Juelich hist. ({} voxels)'.format(np.sum(bool_mask)), display_mode='tiled', draw_cross=False); plt.show(block=blck)
#%% Extract cortical surface areas
fsaverage = datasets.fetch_surf_fsaverage()
atlas_destrieux = datasets.fetch_atlas_surf_destrieux()
all_TR_surfR = nl.surface.vol_to_surf(all_TR, fsaverage.pial_right)
all_TR_surfL = nl.surface.vol_to_surf(all_TR, fsaverage.pial_left)
#%% Plotting the cortical parcellation
plotting.plot_surf_roi(
    fsaverage["infl_left"],
    roi_map=atlas_destrieux["map_left"],
    hemi="left",
    view="medial",
    bg_map=fsaverage["sulc_left"],
    bg_on_data=True,
    darkness=0.5,
)
plt.show(block=blck)
#%%
imgR = nl.image.new_img_like(all_TR_surfR, atlas_destrieux["map_right"])
imgL = nl.image.new_img_like(all_TR_surfL, atlas_destrieux["map_left"])
# maskerL = NiftiLabelsMasker(labels_img=imgL, standardize=True)
# maskerR = NiftiLabelsMasker(labels_img=atlas_destrieux["map_right"], standardize=True)
#%% cortical_L = maskerL.fit_transform([all_TR])
side = 'left'
for i in range(1, 5, 1):
    plotting.plot_surf_roi(
        fsaverage["pial_" + side],
        roi_map=all_TR_surfL[:,i] if side == 'left' else all_TR_surfR[:,i],
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_" + side],
        bg_on_data=True,
        title=f"Destrieux {side} {i}",
    )
    plt.show(block=False)
    plt.pause(5)
    plt.close()
#%% Focus on G_oc-temp_med-Lingual
side = 'left'
label =  b'G_temp_sup-Lateral' # b'S_temporal_transverse' # Heschl's gyri - primary auditory cortex (Brodmann areas 41 and 42)
label_index = [atlas_destrieux['labels'].index(label)]
bool_mask = reduce(lambda x, y: x + y, [(atlas_destrieux["map_"+side] == i) for i in label_index])
#%% plot on surf
# make all_TR_surfL zero everywhere outside the mask
plot_region = np.zeros_like(all_TR_surfR)
plot_region[bool_mask,:] = all_TR_surfR[bool_mask,:] if side == 'right' else all_TR_surfL[bool_mask,:]
plotting.plot_surf_roi(
    fsaverage["pial_" + side],
    roi_map= plot_region[:,14],
    hemi=side,
    view="lateral",
    bg_map=fsaverage["sulc_"+side],
    bg_on_data=True,
    title=f"Destrieux {side} {label}",
)
plt.show(block=blck)
############################################################################################################
############################################################################################################
#%% Linear regression
from nilearn.glm.first_level import make_first_level_design_matrix
t_r = 1.5
slice_time_ref = 0.5
T = all_TR_surfL.shape[-1]; time_x = (np.arange(T) + 0.5) * t_r

#%%
from nilearn.datasets import fetch_localizer_first_level
t_r = 2.4
slice_time_ref = 0.5
data = fetch_localizer_first_level()
fmri_img = data.epi_img
events_file = data.events
events = pd.read_table(events_file)
texture = nl.surface.vol_to_surf(fmri_img, fsaverage.pial_right)
T = texture.shape[-1]; time_x = (np.arange(T) + 0.5) * t_r
#%%
#%% plot fmri image
design_matrix = make_first_level_design_matrix(time_x,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )
