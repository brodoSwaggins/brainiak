figsToPDF = []
import pickle
import wesanderson as wa
import sys
from functools import reduce
import numpy as np
from MDL_tools  import *
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
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
plt.rc('figure', titlesize=smallsize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)
import pandas as pd
import nilearn as nl
from nilearn import plotting, image, datasets, masking
from nilearn.maskers import NiftiLabelsMasker
# from brainiak.eventseg.event import EventSegment
import brainiak.eventseg.event
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

def within_across_corr(D, nEvents, w=5, nPerm=1000, verbose=0, rsd = 0):
    """
    Compute within vs across boundary correlations for a given number of events
    Parameters
    ----------
    D : np array (nSubj x nVertices x nTR)
    nEvents : for HMM
    w : time window for time autocorrelation
    nPerm : number of permutations for statistical analysis
    verbose : 0 - no print, 1 - print log, 2 - print log and output violin plot
    Returns
    -------
    within_across : np array (nSubj x nPerm+1)
    """
    nSubj, _,  nTR = D.shape
    within_across = np.zeros((nSubj, nPerm+1))
    print(".......... Computing time correlation for HMM with {} events.............".format(nEvents))
    for left_out in range(nSubj):
        # Fit to all but one subject
        ev = brainiak.eventseg.event.EventSegment(nEvents)
        ev.fit(D[np.arange(nSubj) != left_out,:,:].mean(0).T)
        events = np.argmax(ev.segments_[0], axis=1)

        # TEST: Compute correlations separated by w in time for the held-out subject
        corrs = np.zeros(nTR-w)
        for t in range(nTR-w):
            corrs[t] = pearsonr(D[left_out,:,t],D[left_out,:,t+w])[0]
        _, event_lengths = np.unique(events, return_counts=True)

        # Compute mean
        # within vs across boundary correlations, for real and permuted bounds
        np.random.seed(rsd)
        for p in range(nPerm+1): # p=0 is the real events (no permuataion)
            within = corrs[events[:-w] == events[w:]].mean()
            across = corrs[events[:-w] != events[w:]].mean()
            within_across[left_out, p] = within - across
            # This makes the next itertion run over a permuted version of the event lengths
            perm_lengths = np.random.permutation(event_lengths)
            events = np.zeros(nTR, dtype=int)
            events[np.cumsum(perm_lengths[:-1])] = 1
            events = np.cumsum(events)
        if verbose >0:
            print('Subj ' + str(left_out+1) + ': WvsA = ' + str(within_across[left_out,0]))
    if verbose > 1:
        plt.figure(figsize=(1.5, 5))
        plt.violinplot(within_across[:, 1:].mean(0), showextrema=True)  # permuted
        plt.scatter(1, within_across[:, 0].mean(0))  # real
        plt.gca().xaxis.set_visible(False)
        plt.ylabel('Within vs across boundary correlation')
        plt.title('{} {} :\nHeld-out subject HMM with {} events ({} perms)'.format(side, label, nEvents, nPerm))
        plt.show(block=blck)
    return within_across

#%% Example use - single subject
if sys.platform == 'linux':
    file = r'/home/itzik/Desktop/EventBoundaries/recall_files/sherlock_recall_s1.nii'
    blck = False
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
plotting.plot_img(atlas_HarvOx.maps, title="Harvard-Oxford atlas", colorbar=True); plt.show(block=False)
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
    fsaverage["pial_left"],
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
label =   b'G_pariet_inf-Angular'#b'G_temp_sup-Lateral' # b'S_temporal_transverse' # Heschl's gyri - primary auditory cortex (Brodmann areas 41 and 42)
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
    blck = False
    dataPath = r'/home/itzik/Desktop/EventBoundaries/Data from Kumar23/'
else:
    file = r'C:\Users\izika\OneDrive\Documents\ComDePri\Memory\fMRI data Project\RecallFiles_published\recall_files\sherlock_recall_s1.nii'
    dataPath = r'C:\\Users\\izika\OneDrive\Documents\ComDePri\Memory\\fMRI data Project\Kumar23Data\\'
    blck = False
all_TR = image.load_img(file)
print(all_TR.shape)
first_TR = image.index_img(file, 0)
print(first_TR.shape)
#%%
fsaverage = datasets.fetch_surf_fsaverage()
atlas_destrieux = datasets.fetch_atlas_surf_destrieux()
all_TR_surfR = nl.surface.vol_to_surf(all_TR, fsaverage.pial_right)
all_TR_surfL = nl.surface.vol_to_surf(all_TR, fsaverage.pial_left)
#%% Load MilkyWay preprocessed data
task = 'milkyway'
if sys.platform == 'linux':
    pathDS = r'/home/itzik/Desktop/EventBoundaries/milkyway_vodka/Milkyway/niftis_preprocessed'
else:
    pathDS = r'C:\Users\izika\OneDrive\Documents\Hebrew U\Modeling of cognition\brainiak\docs\examples\eventseg\4\milkyway_vodka\Milkyway\niftis_preprocessed'
file_names = [pathDS + '/' + f for f in os.listdir(pathDS) if f.endswith('.nii') and not f.startswith('.')]
files = [nl.image.load_img(f) for f in file_names]
#%% Let's look at the first subject
ff = file_names[1]
masker = nl.maskers.NiftiMasker(standardize=True)
ffz = masker.fit_transform(ff)
ffz = masker.inverse_transform(ffz)
#%%
all_TR = ffz
print(all_TR.shape)
first_TR = image.index_img(ffz, 0)
print(first_TR.shape)
plotting.plot_stat_map(first_TR, threshold=1)#, output_file=output_dir / "first_TR.png")
plt.show(block = blck)
#%% Z score all data
masker = nl.maskers.MultiNiftiMasker(standardize=True)
BOLD = masker.fit_transform(file_names)
BOLD = masker.inverse_transform(BOLD)
#%% First subject the same?
plotting.plot_stat_map(image.index_img(BOLD[1], 0), threshold=1)#, output_file=output_dir / "first_TR.png")
plt.show(block = blck)
#%% the story actually started at TR=15, and ended at TR=283. This is true for all the subjects,
# except Subj18 and Subj27 who started at TR=11 and ended at TR=279
#%% Extract the story relevant TRS only
BOLD_sliced = [image.index_img(BOLD[b], slice(15, 283)) for b in range(len(BOLD))]
BOLD_sliced[17] = image.index_img(BOLD[17], slice(11, 279))
# BOLD_sliced[27] = image.index_img(BOLD[27], slice(11, 279))

#%% Project to cortical surface
side = 'left'
fsaverage = datasets.fetch_surf_fsaverage()
atlas_destrieux = datasets.fetch_atlas_surf_destrieux()
surfL = [nl.surface.vol_to_surf(s, fsaverage.pial_left) for s in BOLD_sliced]
surfR = [nl.surface.vol_to_surf(s, fsaverage.pial_right) for s in BOLD_sliced]
#%% plot on surf
for i in range(1, 4, 1):
    tr = i*10
    plotting.plot_surf_roi(
        fsaverage["pial_" + side],
        roi_map=surfL[0][:,tr],
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_" + side],
        bg_on_data=True,
        title=f"TR {tr}",
        #cmap='plasma',
        threshold=0.0001,
    )
    plt.show(block=blck)
    plt.pause(3)
    plt.close()

#%%  Fit Cortical with held-out subjects, focus on one region
label =   b'G_pariet_inf-Angular'#b'G_temp_sup-Lateral' # b'S_temporal_transverse' # Heschl's gyri - primary auditory cortex (Brodmann areas 41 and 42)
# label = b'S_temporal_transverse'
label = b'G_temp_sup-Lateral'
label_index = [atlas_destrieux['labels'].index(label)]
regionInd = np.where(atlas_destrieux["map_"+side] == label_index)[0]
#show region on surface
plot_region = np.zeros_like(surfR[0])
plot_region[regionInd,:] = surfR[0][regionInd,:]
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
#%% Find the best number of HMM segments for the region
all_surf = np.array(surfL) if side == 'left' else np.array(surfR)
all_region = all_surf[:,regionInd,:]
segments_vals = np.arange(2, 15, 1)
score = [] ; nPerm = 1000 ; w = 5 ; nSubj = len(files)
within_across_all = np.zeros((len(segments_vals),nSubj, nPerm+1))
for i,nSegments in enumerate(segments_vals):
    within_across_all[i] = within_across_corr(all_region, nSegments, w, nPerm, verbose=0)
    score.append(within_across_all[i,:,0].mean())
    print(f"Number of HMM segments: {nSegments}, HMM score: {score[-1]}")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt.plot(segments_vals, score, marker='o', color='black'); plt.title('num of events comparison, {}'.format(label))
plt.xlabel('Number of events'); plt.ylabel('mean(within-across) correlation'); ax.set_xticks(segments_vals)
plt.axhline(np.max(score), color='black', linestyle='--', linewidth=0.5)
plt.show(block = blck)
#%% For the best number of events, violin plot of within-across correlation
best_ind = np.argmax(score)
plt.figure(figsize=(1.5,5))
plt.violinplot(within_across_all[best_ind,:,1:].mean(0), showextrema=True) # permuted
plt.scatter(1, within_across_all[best_ind,:,0].mean(0), label= 'Real events') # real
plt.gca().xaxis.set_visible(False)
plt.ylabel('Within vs across boundary correlation'); plt.legend()
plt.title('{} {} :\nHeld-out subject HMM with {} events ({} perms)'.format(side, label, segments_vals[best_ind], nPerm))
plt.show(block = blck)
#%% Loop over all cortical regions
segments_vals = np.arange(2, 50, 1)
bestHMMPerRegion= {}
allHMMruns_within_acrr= {}
nPerm = 1000 ; w = 5 ; nSubj = len(files)
for r in range(1, 17, 1):
    label = atlas_destrieux['labels'][r]
    regionInd = np.where(atlas_destrieux["map_"+side] == r)[0]
    all_region = all_surf[:,regionInd,:]
    within_across_all = np.zeros((len(segments_vals),nSubj, nPerm+1))
    for i,nSegments in enumerate(segments_vals):
        within_across_all[i] = within_across_corr(all_region, nSegments, w, nPerm, verbose=0)
        score = within_across_all[i,:,0].mean()
        print(f"Region {r}: {label}, number of segments: {nSegments}, HMM score: {score}")
        if r not in bestHMMPerRegion or score > bestHMMPerRegion[r]['score']:
            "+++++++++++++++++++++updating best++++++++++++++++++++++++++++"
            bestHMMPerRegion[r] = {
                'name': label,
                'nSegments': nSegments,
                'score': score
            }
    allHMMruns_within_acrr[r] = within_across_all
#%% SAve the results
# np.savez('HMMscorePerRegion_left_w5', HMMscorePerRegion=HMMscorePerRegion)
np.savez('HMMperRegion_'+side+'_w'+w, bestHMMPerRegion=bestHMMPerRegion, allHMMruns_within_acrr=allHMMruns_within_acrr)
#%% ++++++++++++++++++++++++++++++ old format
bestNumEvents= {}
for r in range(len(HMMscorePerRegion)):
    score = np.max(HMMscorePerRegion[r][:,:,0].mean(-1))
    best = num_events[np.argmax(HMMscorePerRegion[r][:,:,0].mean(-1))]
    bestNumEvents[r+1] = {
        'name': atlas_destrieux['labels'][r + 1],
        'numEvents': best,
        'score': score
    }
#%%
np.savez('bestNumEvents_left_w5', bestNumEvents=bestNumEvents)
#%%
bestNumEvents = np.load('bestNumEvents_left_w5.npz', allow_pickle=True)['bestNumEvents'].item()
#%% +++++++++++++++++++++++++++++++++++++ end old format
plot_nSegs = np.zeros_like(surfL[0][:,0])
for r in bestHMMPerRegion:
    regionInd = np.where(atlas_destrieux["map_"+side] == r)[0]
    #show region on surface
    plot_nSegs[regionInd] = bestHMMPerRegion[r]['numEvents'] # ['nSegments']
    print(r, ": ", bestHMMPerRegion[r]['name'], bestHMMPerRegion[r]['numEvents'])
#%%
plotting.plot_surf_roi(
        fsaverage["infl_" + side],
        roi_map= plot_nSegs,
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_"+side],
        bg_on_data=True, darkness=0.25,
        title=f"Optimal HMM granularity, {side} hemi, '{task}'", title_font_size=14,
        colorbar = True, cmap = 'viridis', threshold=2
)
plt.show(block=blck)
#%%
plotting.plot_surf(
        fsaverage["infl_" + side],
        plot_nSegs,
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_"+side],
        bg_on_data=True,
        title=f"Destrieux {side}",
        cmap = 'viridis', cbar_tick_format="auto"
)
plt.show(block=blck)
#%% find max and min of bestNumEvents
maxNumSegs = np.max([bestHMMPerRegion[r]['nSegments'] for r in bestHMMPerRegion])
minNumSegs = np.min([bestHMMPerRegion[r]['nSegments'] for r in bestHMMPerRegion])
print(f"Max number of events: {maxNumSegs}, Min number of events: {minNumSegs}")

#%% import word embeddings pickle
dataPath = '/home/itzik/PycharmProjects/brainiak/docs/examples/eventseg/results/milkyway'
task = 'milkyway'
hiddenLayer_data = pd.read_pickle(dataPath+'/milkywaygpt2-xl-c_1024.pkl')
#%% PCA
k = 50
embeddings = np.array([np.array(a[0]) for a in hiddenLayer_data]) # [np.array(a) for a in hiddenLayer_data]
pca = PCA(n_components=k, whiten=False, random_state=42)
Y = pca.fit_transform(embeddings)
# Rescale
Y = Y / np.sqrt(pca.explained_variance_)  # same as whiten
Y = Y.T
# reducedX = reducedX / pca.singular_values_
#%% save embeddings
np.savez('embeddingsPCA_'+task, embeddings=Y)
#%%
Y = np.load('embeddingsPCA_milkyway.npz')['embeddings'].T
#%% Read pickle of embeds
# pfile = open('/home/itzik/PycharmProjects/EventBoundaries/results/milkyway/milkywaygpt2-xl-c_1024.pkl', 'rb')
# embeds_data = pickle.loads(pfile)


#%% Run MDL with multiple b values and record where events occurred
# For each word in embeds, save the number of conditions for which it appeared as an event boundary
# =============================================================================
#%% open previously saved numEB npy files and combine
# EBdata1 = np.load(r'fullMDL_EB_milkyway.npz')
# EB_all1 = EBdata1['EBs']; bvals1 = EBdata1['bvals']; # tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']
# EBdata2 = np.load(r'fullMDL_addendum_EB_milkyway.npz')
# EB_all2 = EBdata2['EBs']; bvals2 = EBdata2['bvals']; # tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']
#%% Combine the two EBs ordered by the corresponding bvals
# bvals = np.unique(np.concatenate((bvals1, bvals4)))
# EB_all = np.zeros((len(bvals), 1315))
# for i, b in enumerate(bvals):
#     if b in bvals1:
#         ind1 = np.where(bvals1 == b)[0][0]
#         if b in bvals4:
#             print(b, "found in both")
#             ind2 = np.where(bvals4 == b)[0][0]
#             assert np.all(EB_all1[ind1,:]==EB_all4[ind2,:])
#         EB_all[i,:] = EB_all1[ind1,:]
#     elif b in bvals4:
#         ind2 = np.where(bvals4 == b)[0][0]
#         EB_all[i,:] = EB_all4[ind2,:]
#     else:
#         print("Error: b not found in either EBs")

#%% Run over multiple values of parameters b and tau
event_rep = 'const' ; sig = np.std(Y, axis=-1)
bvals =  np.arange(101,500,1) # np.concatenate((np.arange(100,410,10),np.arange(425,525,25)))
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
np.savez('fullMDL_EB_v2_'+task, EBs=EB_all, bvals=bvals)
#%%
#%% open previously saved numEB npy file
# EBdata = np.load(r'/home/itzik/PycharmProjects/EventBoundaries_deploy/numEB_monkey_narrative_.npz')
path= r'/home/itzik/PycharmProjects/brainiak/docs/examples/eventseg/'
EBdata = np.load(path+'fullMDL_EB_101to500_v3_'+task+'.npz')
EB_all = EBdata['EBs']; bvals = EBdata['bvals'] #; tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']

#%% Plot EB hierarchy
fig = plt.figure()
title_str = task
waxis = np.arange(0, len(EB_all.T))
numEvents_b = {}
cc = wa.color_palettes['Darjeeling Limited'][0]
for i,b in enumerate(bvals):
    bndrs = np.where(EB_all[i])
    db = (bvals[1]-bvals[0])/2
    plt.vlines(bndrs, b-db, b+db, colors=cc[i % len(cc)], linewidth=1)
    gran = int(np.sum(EB_all[i]))
    if gran not in numEvents_b:
        numEvents_b[gran] = []
    numEvents_b[gran].append(b)
    plt.text(0, b, f'{gran}', color=cc[i % len(cc)], fontsize=8)
plt.ylabel('b value'); plt.xlabel('Word index')
plt.yticks(bvals, fontsize=8)
plt.title('Event boundaries hierarchy over b values, '+title_str)
plt.show(block=blck)
#%% show support size for each b value
fig = plt.figure()
supp = [len(numEvents_b[bb]) for bb in numEvents_b]
plt.bar(list(numEvents_b.keys()), supp)
plt.xticks(list(numEvents_b.keys())); plt.xlabel('Number of events'); plt.ylabel('Number of b values')
#%% For each region, find the b value closest to its num events
b_per_region = {}
inaccurate_b = []
for r in bestHMMPerRegion:
    numEvents = bestHMMPerRegion[r]['nSegments']-1 # -1 to get num boundaries
    while numEvents not in numEvents_b:
        inaccurate_b.append(r)
        print(f"Number of events {numEvents} for region {bestHMMPerRegion[r]['name']} not found in EB hierarchy")
        numEvents += 1
        print(f"Adding {numEvents} instead")
        continue
    b_per_region[r] = numEvents_b[numEvents]
##########################################################################################
#%% #######################Time correlation vs. the HMM results###########################
##########################################################################################
# the story actually started at TR=15, and ended at TR=283. For all the subjects.
# except Subj18 and Subj27 who started at TR=11 and ended at TR=279
label = b'S_temporal_transverse'
label_index = [atlas_destrieux['labels'].index(label)]
regionInd = np.where(atlas_destrieux["map_"+side] == label_index)[0]
all_surf = np.array(surfL) if side == 'left' else np.array(surfR)
all_region = all_surf[:,regionInd,:]
nEvents = bestHMMPerRegion[75]['numSegments'] ; spMerge = False
ev = brainiak.eventseg.event.EventSegment(nEvents, split_merge=spMerge)
ev.fit(all_region.mean(0).T)
#%%
segments = np.argmax(ev.segments_[0], axis=1)
HMM_ebs = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0] # todo how is this 40 when nSeg=39??
segmentsVar = np.var(ev.segments_[0], axis=-1)
modal_diffs = []
for i in range(len(ev.segments_[0])):
    probabilities = ev.segments_[0][i]
    sorted_probabilities = np.sort(probabilities)[::-1]  # Sort in descending order
    modal_diffs.append(sorted_probabilities[0] - sorted_probabilities[1])
#%% Animation of probability distribution
# Function to update the plot for each frame
length = 0 ; running_diff = []
def pop_all(l):
    r, l[:] = l[:], []
    return r
def init_func():
    ax.clear()
    ax.set_ylim(0, 1)
    ax.set_title('Time Point: 0')
    ax.set_xlabel('States')
    ax.set_ylabel('Probability')
    return bars
def update(frame):
    ax.clear()
    probabilities = ev.segments_[0][frame]
    sorted_probabilities = np.sort(probabilities)[::-1]  # Sort in descending order
    diff  = sorted_probabilities[0] - sorted_probabilities[1]
    running_mean = 0 if not running_diff else np.mean(running_diff)
    # print(frame, running_mean, diff)
    # print("===", sorted_probabilities[:5], "======" )
    running_diff.append(diff)
    if frame in HMM_ebs:
        cc = 'blue'
        pop_all(running_diff)
    else:
        cc = 'grey'
    ax.bar(range(nEvents), sorted_probabilities, color=cc)
    ax.set_ylim(0, 1)  # Ensure the y-axis stays consistent
    # ax.set_yscale('log')
    # ax.axhline(running_mean, color='red', linestyle='--', label=f'mean diff={running_mean:.3f}')
    ax.axhline(diff, color='black', linestyle='--', label=f'diff={diff:.3f}')
    ax.set_title(f'Time Point: {frame+1}')
    ax.set_xlabel('States'); plt.legend()
    ax.set_ylabel('Probability')
fig, ax = plt.subplots()
bars = ax.bar(range(nEvents), np.zeros(nEvents))

# Create the animation
anim =FuncAnimation(fig, update, frames=100, init_func= init_func, repeat=False, interval=250)

# Save or display the animation
plt.show()
#%% Plot the measure as probability of event boundaries
measure = np.array(modal_diffs); title = 'Modal difference'
# measure = segmentsVar ; title = 'segment variance'
fig = plt.figure(); plt.title(title+' of HMM boundaries')
plt.hist(measure[HMM_ebs], bins=10) ; plt.xlabel(title); plt.ylabel('EB count')
plt.hist(measure[~HMM_ebs], bins=10, alpha=0.5)
plt.show()
# figsToPDF.append(plt.gcf())
#%%
b = b_per_region[75][0]
b_ind = np.where(bvals == b)[0][0]
model_ebs = np.where(EB_all[b_ind])[0]
dt= Y.shape[-1]/all_surf.shape[-1] # TR duration in words [words per TR]
HMM_ebs_inText = np.array([int(bb*dt) for bb in HMM_ebs])
print("MODEL==========>", model_ebs)
print("HMM==========>", HMM_ebs_inText)

model_1hot = np.zeros(Y.shape[-1]); model_1hot[model_ebs] = 1
hmm_1hot = np.zeros(Y.shape[-1]); hmm_1hot[HMM_ebs_inText] = 1
#%% compute correlation between two 1-hot smoothed boundary vectors
def boundaryCorrelation(vec1, vec2, smoothing_sig = None):
    if smoothing_sig: # apply Gaussian smoothing
        smooth1 = gaussian_filter1d(vec1, sigma=smoothing_sig, mode='constant', cval=0)
        smooth2 = gaussian_filter1d(vec2, sigma=smoothing_sig, mode='constant', cval=0)
        cor = pearsonr(smooth1, smooth2)
    else:
        # cor = np.sum([np.min(np.abs(modelBoundaries - h)) for h in hmmBoundaries])/len(hmmBoundaries)
        print("sigma missing")
    return cor
#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.plot(gaussian_filter1d(model_1hot, sigma=10, mode='constant', cval=0), label='Model', color='red')
plt.plot(gaussian_filter1d(hmm_1hot, sigma=10, mode='constant', cval=0), label='HMM', color='blue')
plt.legend(); plt.show()
#%% Plot both boundaries
fig = plt.figure()
waxis = np.arange(0, len(all_surf[0].T))
plt.vlines(model_ebs, 1.5, 2.5, linewidth=3, alpha=0.7, color='grey', label='Model')
# plt.vlines(hmmBoundaries, 1.7, 2.4, linewidth=1, color=wa.color_palettes['Darjeeling Limited'][spMerge][1], label='HMM')
plt.scatter(HMM_ebs_inText, 2*np.ones_like(HMM_ebs_inText), marker='|', linewidths=3, s=500, \
                color=wa.color_palettes['Darjeeling Limited'][spMerge][1], label='HMM'+'+merge'*spMerge)
plt.legend(); plt.xlim(left=-3); plt.show()
figsToPDF.append(plt.gcf())
#%% plot model boundaries against some measure of likelihhod to transition
fig = plt.figure(figsize=(12, 5))
measure = np.array(modal_diffs); title = 'modal difference'
# measure = segmentsVar; title = 'segment variance'
model_ebs_inTR = np.rint(model_ebs/dt).astype(int)
plt.plot(measure, label=title, color='grey'); plt.xlabel('Time point'); plt.ylabel(title)
# plt.scatter(model_ebs_inTR, measure[model_ebs_inTR], color='red', label='Model boundaries')
plt.scatter(HMM_ebs, measure[HMM_ebs], color='blue', marker="*", label='HMM boundaries')
# for bb in HMM_ebs:
#     plt.plot([bb-1, bb, bb+1], [measure[bb-1],measure[bb],measure[bb+1]], color='blue', linewidth=2)
plt.legend(); plt.show()
figsToPDF.append(plt.gcf())
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
## need: ............. acces the preprocessed data
#%%
fsaverage = datasets.fetch_surf_fsaverage()
atlas_destrieux = datasets.fetch_atlas_surf_destrieux()
all_TR_surfR = nl.surface.vol_to_surf(all_TR, fsaverage.pial_right)
all_TR_surfL = nl.surface.vol_to_surf(all_TR, fsaverage.pial_left)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Run HMM on the data. First, use average to decide best granularity per region
side = 'left'
# downsample the TR data into 76 cortical regions
regions = np.zeros((len(atlas_destrieux['labels']),all_TR.shape[-1]))
all_TR_avg = np.zeros_like(all_TR_surfR)
for i, label in enumerate(atlas_destrieux['labels']):
    if label == b'Unknown':
        continue
    bool_mask = reduce(lambda x, y: x + y, [(atlas_destrieux["map_"+side] == i)])
    print(f"Region {label} has {np.sum(bool_mask)} voxels")
    regions[i,:] = np.mean(all_TR_surfL[bool_mask,:], axis=0) if side == 'left' else np.mean(all_TR_surfR[bool_mask,:], axis=0)
    all_TR_avg[bool_mask,:] = regions[i,:]
regions = regions[1:,:]
#%% plot region avergaes on cortical surface
for i in range(1, 4, 1):
    plotting.plot_surf_roi(
        fsaverage["pial_" + side],
        roi_map=all_TR_surfL[:,i*10],
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_" + side],
        bg_on_data=True,
        title=f"TR {i}",
        cmap='plasma',
        threshold=0.0001,
    )
    plt.show(block=False)
    plt.pause(3)
    plt.close()
#%%
label =   b'G_pariet_inf-Angular'#b'G_temp_sup-Lateral' # b'S_temporal_transverse' # Heschl's gyri - primary auditory cortex (Brodmann areas 41 and 42)
label_index = [atlas_destrieux['labels'].index(label)]
regionInd = np.where(atlas_destrieux["map_"+side] == label_index)[0]
# exRegion = regions[label_index,:].reshape(1,-1)
exRegion = all_TR_surfL[regionInd,:] if side == 'left' else all_TR_surfR[regionInd,:]

#%%
numEvents = np.arange(30, 100, 10)
segs = []
for n in numEvents:
    ev = brainiak.eventseg.event.EventSegment(n)
    ev.fit(exRegion.T)
    segs.append(ev)
    print(f"Number of regions: {n}, log likelihood: {ev.ll_[-1]}")
# ev = brainiak.eventseg.event.EventSegment(70)
# ev.fit(regions.T)
#%%
savefig(figsToPDF, os.getcwd(), savename='modal_diff', tight=False, prefix="figures")

