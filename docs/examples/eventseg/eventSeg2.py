import wesanderson as wa
import sys
from functools import reduce
import numpy as np
from MDL_tools  import *
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
plotting.plot_roi(mask_img, title='hippocampi, Harvard-Oxford ({} voxels)'.format(np.sum(bool_mask)), display_mode='tiled', draw_cross=False, cmap = 'viridis'); plt.show(block=blck)
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
plotting.plot_roi(mask_img, title='hippocampi, Juelich hist. ({} voxels)'.format(np.sum(bool_mask)), display_mode='tiled', draw_cross=False, cmap='viridis'); plt.show(block=blck)
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
    plt.pause(3)
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
    cmap='plasma',
    threshold=0.0001,
)

plt.show(block=blck)
############################################################################################################
############################################################################################################
#%% Linear regression
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
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
#%% plot fmri image
design_matrix = make_first_level_design_matrix(time_x,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )
labels, estimates = run_glm(texture.T, design_matrix.values)
#%%
contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
basic_contrasts['audio'] = (
    basic_contrasts['audio_left_hand_button_press']
    + basic_contrasts['audio_right_hand_button_press']
    + basic_contrasts['audio_computation']
    + basic_contrasts['sentence_listening'])

# one contrast adding all conditions involving instructions reading
basic_contrasts['visual'] = (
    basic_contrasts['visual_left_hand_button_press']
    + basic_contrasts['visual_right_hand_button_press']
    + basic_contrasts['visual_computation']
    + basic_contrasts['sentence_reading'])

# one contrast adding all conditions involving computation
basic_contrasts['computation'] = (basic_contrasts['visual_computation']
                                  + basic_contrasts['audio_computation'])

# one contrast adding all conditions involving sentences
basic_contrasts['sentences'] = (basic_contrasts['sentence_listening']
                                + basic_contrasts['sentence_reading'])
#%%
contrasts = ['audio', 'visual', 'computation', 'sentences', 'sentence_reading', 'sentence_listening', 'audio_computation']
for index, contrast_id in enumerate(contrasts):
    contrast_val = basic_contrasts[contrast_id]
    print(f"  Contrast {index + 1:1} out of {len(contrasts)}: "
          f"{contrast_id}, right hemisphere")
    # compute contrast-related statistics
    contrast = nl.glm.contrasts.compute_contrast(labels, estimates, contrast_val,
                                stat_type='t')
    # we present the Z-transform of the t map
    z_score = contrast.z_score()
    # we plot it on the surface, on the inflated fsaverage mesh,
    # together with a suitable background to give an impression
    # of the cortex folding.
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, z_score, hemi='right',
        title=contrast_id, colorbar=True,
         bg_map=fsaverage.sulc_right, threshold=3.,
    )
    plt.show(block=False)
    plt.pause(10)
    plt.close()
############################################################################################################
##### Event segmentation of narratives #####################################################################
#%%  import fMRI data. project to cortical surface

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
fsaverage = datasets.fetch_surf_fsaverage()
atlas_destrieux = datasets.fetch_atlas_surf_destrieux()
all_TR_surfR = nl.surface.vol_to_surf(all_TR, fsaverage.pial_right)
all_TR_surfL = nl.surface.vol_to_surf(all_TR, fsaverage.pial_left)
#%% import word embeddings
#%% Load Kumar et al. data ("Pieman" embeddings, reduced to 50 PCs )
# dataPath = 'C:\\Users\\izika\OneDrive\Documents\ComDePri\Memory\\fMRI data Project\Kumar23Data\\'
task = 'pieman'
dataPath = r'/home/itzik/Desktop/EventBoundaries/Data from Kumar23/'
# embeddingsData = pd.read_csv(dataPath + r'extract-embeddings-data/results/tunnel/tunnelgpt2-xl-c_1024-layer_0_pca50d.csv')
embeddingsData = pd.read_csv(dataPath + r'extract-embeddings-data/results/pieman/piemangpt2-xl-c_1024-layer_0_pca50d.csv')

# First 4 columns a are metadata (word, onset, offset, speaker), the rest are the embeddings. Separate them
metadata = embeddingsData.iloc[:,:5]
embeds = embeddingsData.iloc[:,5:] # time x dim
embeds.columns = range(0,50)

#%% Run MDL with multiple b values and record where events occurred
# For each word in embeds, save the number of conditions for which it appeared as an event boundary
# =============================================================================
#%% open previously saved numEB npy file
EBdata = np.load(r'/home/itzik/PycharmProjects/EventBoundaries_deploy/numEB_monkey_narrative_.npz')
EB_all = EBdata['EBs']; bvals = EBdata['bvals']; # tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']
#%% Run over multiple values of parameters b and tau
Y = embeds.values.T
event_rep = 'const' ; sig = np.std(Y, axis=-1)
bvals = np.arange(90, 131, 5) # such that we capture the 3rd quartile of button press ratio from the data
tvals = np.arange(25,530,100)
#%% Run aposteriori MDL
EB_all = np.zeros((len(bvals), Y.shape[-1]))
stime = time.time()
for i, b in enumerate(bvals):
    print( "b=",b,".........")
    sstime = time.time()
    # EBs, MDL, seg_points, seg_offsets = MDL_tau(Y, tau, b, sig, rounded=True, v=False)
    EB, MDL = EB_split(Y, b=b, rep='const', sig=sig)
    for k in EB:
        EB_all[i, k] += 1
    print("                   ====>time: %f score: %f" % (time.time()-sstime, MDL))
print("total time: %f" % (time.time()-stime))

#%% Print story with EB in Capital letters
for i, w in enumerate(metadata['0']):
    if i in EB :
        print(w.upper(), end=' || ')
    else:
        print(w, end=' ')
    if i % 20 == 19:
        print('\n')
print()
#%% Running with tau:
# EB_all = np.zeros((len(bvals), Y.shape[-1]))
# segPts_all = np.zeros((len(bvals), len(tvals), Y.shape[-1]))
# stime = time.time()
# for i, b in enumerate(bvals):
#     for j, tau in enumerate(tvals):
#         print( "b=",b, "tau=",tau, ".........")
#         sstime = time.time()
#         # EBs, MDL, seg_points, seg_offsets = MDL_tau(Y, tau, b, sig, rounded=True, v=False)
#         EBs, MDL, seg_points = MDL_tau_narrative(Y, tau, b, sig, rounded=False, updateSig=False, v=False)
#         EB = EBs[-1]
#         for k in EB:
#             EB_all[i, j, k] += 1
#         for k in seg_points:
#             segPts_all[i, j, k] += 1
#         print("                   ====>time: %f score: %f" % (time.time()-sstime, MDL[-1]))
# print("total time: %f" % (time.time()-stime))
#%% save params and results
np.savez('mriEB_'+task, EBs=EB_all, bvals=bvals)
#%% Plot EB hierarchy
fig = plt.figure()
title_str = task
waxis = np.arange(0, len(EB_all.T))
cc = wa.color_palettes['Darjeeling Limited'][0]
for i,b in enumerate(bvals):
    bndrs = np.where(EB_all[i])
    db = (bvals[1]-bvals[0])/2
    plt.vlines(bndrs, b-db, b+db, colors=cc[i % len(cc)], linewidth=1)
    gran = int(np.sum(EB_all[i]))
    plt.text(0, b, f'{gran}', color=cc[i % len(cc)], fontsize=8)
plt.ylabel('b value'); plt.xlabel('Word index'); plt.yticks(bvals)
plt.title('Event boundaries hierarchy over b values, '+title_str)
plt.show(block=True)

# =============================================================================
# =============================================================================
#%%
#%% load Narrative DS partcipants tsv
pathDS = r'/home/itzik/Desktop/EventBoundaries/Narratives_DSs'
participants = pd.read_csv(pathDS + r'/participants.tsv', sep='\t')

# extract rows where 'task' field contains 'pieman', exclude 'piemanpni'
task_participants = participants[participants.task.str.contains(task) & ~participants.task.str.contains('piemanpni')]
#%%
pathDS = r'smb://132.64.186.144/hartlabnas/personal_folders/isaac.ash/OptCodingEB/narrative_DS/ds002345-download'
excluded = ['sub-001', 'sub-013', 'sub-014', 'sub-021', 'sub-022', 'sub-038', 'sub-056', 'sub-068', 'sub-069']
folders = [f for f in task_participants.participant_id.values if f not in excluded]
paths = [pathDS + '/' + f +'/func/' + f + '_task-pieman_bold.nii.gz' for f in folders]
files = [nl.image.load_img(p) for p in paths]
## todo ............. acces the preprocessed data
#%% Run HMM on the data. First, use average to decide best granularity per region


