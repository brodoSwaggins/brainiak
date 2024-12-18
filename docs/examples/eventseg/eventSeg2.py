import string
import wesanderson as wa
import sys
from functools import reduce
from MDL_tools  import *
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import pandas as pd
import nilearn as nl
from nilearn import plotting, image, datasets, masking
from nilearn.maskers import NiftiLabelsMasker
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
from matplotlib.animation import FuncAnimation
smallsize=14; mediumsize=16; largesize=18
plt.rc('xtick', labelsize=smallsize); plt.rc('ytick', labelsize=smallsize); plt.rc('legend', fontsize=mediumsize)
plt.rc('figure', titlesize=smallsize); plt.rc('axes', labelsize=mediumsize); plt.rc('axes', titlesize=mediumsize)
# from brainiak.eventseg.event import EventSegment
import brainiak.eventseg.event
print(f"Datasets are stored in: {datasets.get_data_dirs()!r}")
from pathlib import Path; output_dir = Path.cwd() / "images"
output_dir.mkdir(exist_ok=True, parents=True); print(f"Output will be saved to: {output_dir}")
figsToPDF = []
#%% Function declarations
def getBoolMask(atlas, labels, map=None):
    label_index = [atlas.labels.index(l) for l in labels]
    if map is None:
        map = atlas.maps
    bool_mask = reduce(lambda x, y: x + y, [(map.get_fdata() == i) for i in label_index])
    mask_img = nl.image.new_img_like(map, bool_mask)
    return mask_img, bool_mask
def ISC(D):
    """
    Inter-subject correlation of each voxel in the data - for each subj, the correlation of the time series of each voxel with
    the average of all subjects, then average over all subjects.
    Parameters
    ----------
    D - np array (nSubj x nVertices x nTR) - fMRI data

    Returns
    -------
    ISC - np array(nVertices) - ISC of each voxel
    """
    nSubj, nVox, nTR = D.shape
    ISC = np.zeros((nVox))
    for left_out in range(nSubj):
        D_avg = D[np.arange(nSubj) != left_out].mean(0)
        for v in range(nVox):
            ISC[v] += pearsonr(D[left_out, v, :], D_avg[v, :])[0]
    ISC /= nSubj
    return ISC
def within_across_corr(D, nEvents, w=5, nPerm=1000, verbose=0, rsd = 0, MDL_b = -1, title_prefix=''):
    """
    Compute within vs across boundary correlations for a given number of events OR a given MDL_b
    Parameters
    ----------
    D : np array (nSubj x nVertices x nTR)
    nEvents : for HMM [ignored if @MDL_b > 0]
    w : time window for time autocorrelation
    nPerm : number of permutations for statistical analysis
    verbose : 0 - no print, 1 - print log, 2 - print log and output violin plot
    rsd : random seed
    MDL_b : if > 0, use MDL with b=MDL_b to estimate events. Otherwise, use HMM with @nEvents
    title_prefix : string to add to the title of the violin plot (for verbose=2)
    Returns
    -------
    within_across : np array (nSubj x nPerm+1)
    """
    nSubj, _,  nTR = D.shape
    within_across = np.zeros((nSubj, nPerm+1))
    if MDL_b <= 0:
        print(".......... Computing time correlation for HMM with {} events.............".format(nEvents))
    else:
        print(":::::::::: Computing time correlation for MDL with b={} :::::::::::".format(MDL_b))
        sigD = np.var(D) ; ev = None
    for left_out in range(nSubj):
        # Fit to all but one subject
        if MDL_b > 0:
            boundaries, _ = EB_split(D[np.arange(nSubj) != left_out,:,:].mean(0), b=MDL_b, rep='const', sig=sigD)
            event_lengths = np.diff(boundaries, prepend=0, append=nTR)
            events = np.zeros(nTR, dtype=int)
            events[np.cumsum(event_lengths[:-1])] = 1
            events = np.cumsum(events)
        else:
            ev = brainiak.eventseg.event.EventSegment(nEvents)
            ev.fit(D[np.arange(nSubj) != left_out,:,:].mean(0).T)
            events = np.argmax(ev.segments_[0], axis=1)
            _, event_lengths = np.unique(events, return_counts=True)
        # TEST: Compute correlations separated by w in time for the held-out subject
        corrs = np.zeros(nTR-w)
        for t in range(nTR-w):
            corrs[t] = pearsonr(D[left_out,:,t],D[left_out,:,t+w])[0] #todo over the entire range instead of one time point?

        # Compute mean within vs across boundary correlations, for real and permuted bounds
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
        plt.title(title_prefix+'\nHeld-out subject HMM with {} events ({} perms)'.format(label, nEvents, nPerm))
        plt.show(block=blck)
    return within_across, ev
def BOLDscore(modelEB, D, w=5, nPerm=1000, verbose=0, rsd = 0):
    """
    Given model EB and BOLD data, checks within-across boundary correlations on each subject
    for the model EB as well for permutations of the EB.
    Parameters
    ----------
    modelEB - np array, timings of the model boundaries
    D - np array (nSubj x nVortex x nTR), BOLD data
    w  - time window for time autocorrelation
    nPerm - number of permutations for statistical analysis
    verbose - 0 - no print, 1 - print log, 2 - print log and output violin plot
    rsd - random seed

    Returns
    -------
    within_across - np array (nSubj x nPerm+1)
    """
    nSubj, _, nTR = D.shape
    within_across = np.zeros((nSubj, nPerm+1))
    event_lengths = np.diff(modelEB, prepend=0, append=nTR)
    events = np.zeros(nTR, dtype=int)
    events[np.cumsum(event_lengths[:-1])] = 1
    events = np.cumsum(events)
    for left_out in range(nSubj):
        # TEST: Compute correlations separated by w in time for the held-out subject
        corrs = np.zeros(nTR-w)
        for t in range(nTR-w):
            corrs[t] = pearsonr(D[left_out,:,t],D[left_out,:,t+w])[0]
        # Compute mean within vs across boundary correlations, for real and permuted bounds
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
        plt.title('subject predicition score for model with {} events ({} perms)'.format(len(modelEB), nPerm))
        plt.show(block=blck)
    return within_across
def longest_common_substring(s1: str, s2: str):
    if len(s1) < 1 and len(s2) < 1:
        return "", 0, 0
    # Initialize a matrix to store the lengths of longest common suffixes
    n1, n2 = len(s1), len(s2)
    # Matrix to store the lengths of common suffixes of substrings
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    longest_len = 0  # Length of the longest common substring
    end_s1 = 0  # End index of the longest common substring in s1

    # Fill dp matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_len:
                    longest_len = dp[i][j]
                    end_s1 = i  # Store the end index of the longest common substring in s1

    # If no common substring is found
    if longest_len == 0:
        return "", -1, -1

    # Starting index in s1
    start_s1 = end_s1 - longest_len
    # Find the starting index in s2 using the length of the substring
    start_s2 = s2.find(s1[start_s1:end_s1])

    # Return the longest common substring and starting indices in s1 and s2
    return s1[start_s1:end_s1], start_s1, start_s2

def count_close_ones(model: np.ndarray, data: np.ndarray, distance: int = 3, exclusive=False) -> int:
    """
    Counts the number of 1's in the 'model' time series that are within a specified distance
    of a 1 in the 'data' time series, ensuring that each 1 in 'data' can only be matched once.

    Parameters:
    - model (np.ndarray): The model time-series vector (1D array of 0s and 1s).
    - data (np.ndarray): The data time-series vector (1D array of 0s and 1s).
    - distance (int): The distance within which to count the close ones.

    Returns:
    - int: The count of 1's in 'model' that are within 'distance' time points from a 1 in 'data'.
    """
    # Ensure model and data are numpy arrays and have the same length
    if len(model) != len(data):
        raise ValueError("Model and data time-series must be of the same length.")

    # Find indices of 1s in model and data
    model_indices = np.where(model == 1)[0]
    data_indices = np.where(data == 1)[0]

    # Create a set to keep track of which data indices have been used
    used_data_indices = set()  # Change: Initialize a set to track matched data indices

    count = 0

    # For each 1 in the model, check if there is an unmatched 1 in the data within the specified distance
    for model_index in model_indices:
        # Find a data index within the range [model_index - distance, model_index + distance]
        for data_index in data_indices:
            if abs(model_index - data_index) <= distance and data_index not in used_data_indices:
                count += 1
                if exclusive:
                    used_data_indices.add(data_index)  # Change: Mark this data index as used
                break  # Change: Move to the next model_index after finding a match

    return count
def moving_average_smooth(vector, window_size=3):
    return np.convolve(vector, np.ones(window_size)/window_size, mode='same')
def exponential_smooth(vector, alpha=0.3):
    smoothed = np.zeros_like(vector, dtype=float)
    smoothed[0] = vector[0]  # Initialize the first value
    for t in range(1, len(vector)):
        smoothed[t] = alpha * vector[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed

def DTW(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes the Dynamic Time Warping (DTW) distance between two vectors.

    Parameters:
    - vector1 (np.ndarray): The first time-series vector.
    - vector2 (np.ndarray): The second time-series vector.

    Returns:
    - float: The DTW distance between vector1 and vector2.
    """
    n, m = len(vector1), len(vector2)
    # Create a cost matrix
    dtw_matrix = np.zeros((n + 1, m + 1)) + np.inf
    dtw_matrix[0, 0] = 0

    # Populate the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(vector1[i - 1] - vector2[j - 1])
            # Take the minimum cost path
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # Insertion
                                          dtw_matrix[i, j - 1],  # Deletion
                                          dtw_matrix[i - 1, j - 1])  # Match

    # Return the DTW distance (bottom-right corner of the matrix)
    return dtw_matrix[n, m]

def MSE(vector1: np.ndarray, vector2: np.ndarray) -> float:
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length.")
    # Compute MSE
    mse = np.mean((vector1 - vector2) ** 2)
    return mse

############################################################################################################
##### Event segmentation of narratives #####################################################################
#%% Load MilkyWay preprocessed data
blck = False
task = 'Milkyway'
if sys.platform == 'linux':
    pathDS = r'/home/itzik/Desktop/EventBoundaries'
else:
    pathDS = r'C:\Users\izika\OneDrive\Documents\Hebrew U\Modeling of cognition\brainiak\docs\examples\eventseg\4'
pathDS = os.path.join(pathDS, 'milkyway_vodka',task,'niftis_preprocessed')
file_names = [pathDS + '/' + f for f in os.listdir(pathDS) if f.endswith('.nii') and not f.startswith('.')]
files = [nl.image.load_img(f) for f in file_names]
#%% Let's zscore and look at the first subject
ff = file_names[1]
masker = nl.maskers.NiftiMasker(standardize=True)
ffz = masker.fit_transform(ff)
ffz = masker.inverse_transform(ffz)
all_TR = ffz
print(all_TR.shape)
first_TR = image.index_img(ffz, 0)
plotting.plot_stat_map(first_TR, threshold=1, cut_coords=(-4,-27,18))#, output_file=output_dir / "first_TR.png")
plt.show(block = blck)
#%% Z score all data
masker = nl.maskers.MultiNiftiMasker(standardize=True)
BOLD = masker.fit_transform(file_names)
BOLD = masker.inverse_transform(BOLD)
#%% Extract the story relevant TRS only: the story actually started at TR=15, and ended at TR=283.
# This is true for all the subjects, except Subj18 and Subj27 who started at TR=11 and ended at TR=279
BOLD_sliced = [image.index_img(BOLD[b], slice(15, 285)) for b in range(len(BOLD))]
outlier_ind = np.where(["subj18" in f for f in file_names])[0][0]
BOLD_sliced[outlier_ind] = image.index_img(BOLD[outlier_ind], slice(11, 281))
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
# label = b'S_temporal_transverse' #  b'G_temp_sup-Lateral' b'G_oc-temp_med-Lingual'
label = b'G_temporal_middle'#b'G_temp_sup-Plan_tempo'# b'Lat_Fis-ant-Horizont' #b'Pole_temporal' #  b'G_temp_sup-Lateral' b'G_oc-temp_med-Lingual'
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
segments_vals = np.arange(1, 15, 1)
score = [] ; nPerm = 1000 ; w = 5 ; nSubj = len(files)
within_across_all = np.zeros((len(segments_vals),nSubj, nPerm+1))
for i,nSegments in enumerate(segments_vals):
    within_across_all[i], _ = within_across_corr(all_region, nSegments, w, nPerm, verbose=0)
    score.append(within_across_all[i,:,0].mean())
    print(f"Number of HMM segments: {nSegments}, HMM score: {score[-1]}")
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(segments_vals, score, marker='o', color='black'); plt.title('num of events comparison, {}'.format(label))
plt.xlabel('Number of events'); plt.ylabel('mean(within-across) correlation'); ax.set_xticks(segments_vals)
plt.axhline(np.max(score), color='black', linestyle='--', linewidth=0.5)
plt.show(block = blck)
#%% For the best number of events, violin plot of within-across correlation
# ii = 6 ; label = atlas_destrieux['labels'][ii]
# within_across_all = allHMMruns_within_acrr[ii]
best_ind = np.argmax(score) #bestHMMPerRegion[ii]['nSegments'] - 2
plt.figure(figsize=(5,15))
plt.violinplot(within_across_all[best_ind,:,1:].mean(0), showextrema=True) # permuted
plt.scatter(1, within_across_all[best_ind,:,0].mean(0), label= 'Real events') # real
plt.gca().xaxis.set_visible(False)
plt.ylabel('Within vs across boundary correlation'); plt.legend()
plt.title('{} {} :\nHeld-out subject HMM with {} events ({} perms)'.format(side, label, segments_vals[best_ind], nPerm))
plt.show(block = blck)
#%% Loop over all cortical regions
segments_vals = np.arange(2, 50, 1)
nPerm = 1000 ; w = 5 ; nSubj = len(files)
#%%
input("WARNING: You are about to re-write HMM simulation data. Press Enter to continue...")
bestHMMPerRegion= {} # best HMM fit details for each region
allHMMruns_within_acrr= {} # raw results for all regions
for r in range(1, len(atlas_destrieux['labels']), 1):
    label = atlas_destrieux['labels'][r]
    regionInd = np.where(atlas_destrieux["map_"+side] == r)[0]
    all_region = all_surf[:,regionInd,:]
    within_across_all = np.zeros((len(segments_vals),nSubj, nPerm+1))
    for i,nSegments in enumerate(segments_vals):
        within_across_all[i], ev = within_across_corr(all_region, nSegments, w, nPerm, verbose=0)
        score = within_across_all[i,:,0].mean()
        print(f"Region {r}: {label}, number of segments: {nSegments}, HMM score: {score}")
        if r not in bestHMMPerRegion or score > bestHMMPerRegion[r]['score']:
            "+++++++++++++++++++++updating best++++++++++++++++++++++++++++"
            ev.fit(all_region.mean(0).T) # now fit to all subjects
            segments = np.argmax(ev.segments_[0], axis=1)
            HMM_ebs = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
            bestHMMPerRegion[r] = {
                'name': label,
                'nSegments': nSegments,
                'boundaries': HMM_ebs,
                'nBoundaries': len(HMM_ebs),
                'score': score
            }
    allHMMruns_within_acrr[r] = within_across_all
#%% SAve the results
# np.savez('HMMscorePerRegion_left_w5', HMMscorePerRegion=HMMscorePerRegion)
np.savez('HMMperRegionSliced_'+side+'_w'+str(w), bestHMMPerRegion=bestHMMPerRegion, allHMMruns_within_acrr=allHMMruns_within_acrr, nanRegions=[3,16])
#%%
HMMdata = np.load('HMMperRegionSliced_'+side+'_w'+str(w)+'.npz', allow_pickle=True)
bestHMMPerRegion = HMMdata['bestHMMPerRegion'].item() ; allHMMruns_within_acrr = HMMdata['allHMMruns_within_acrr'].item() ; AlgFail = HMMdata['nanRegions']
#%% ++++++++++++++++++++++++++++++ old format
# bestNumEvents= {}
# for r in range(len(HMMscorePerRegion)):
#     score = np.max(HMMscorePerRegion[r][:,:,0].mean(-1))
#     best = num_events[np.argmax(HMMscorePerRegion[r][:,:,0].mean(-1))]
#     bestNumEvents[r+1] = {
#         'name': atlas_destrieux['labels'][r + 1],
#         'numEvents': best,
#         'score': score
#     }
# #%%
# np.savez('bestNumEvents_left_w5', bestNumEvents=bestNumEvents)
#%%
# bestNumEvents = np.load('bestNumEvents_left_w5.npz', allow_pickle=True)['bestNumEvents'].item()
#%% +++++++++++++++++++++++++++++++++++++ end old format
plot_nSegs = np.zeros_like(surfL[0][:,0]) ; AlgFail = []
for r in bestHMMPerRegion:
    regionInd = np.where(atlas_destrieux["map_"+side] == r)[0]
    #show region on surface
    if np.isnan(bestHMMPerRegion[r]['score']):
        print(r, ": ",bestHMMPerRegion[r]['name'], ">>>>>>>>>HMM Algorithm failure")
        AlgFail.append(r)
        continue
    plot_nSegs[regionInd] = bestHMMPerRegion[r]['nSegments']
    print(r, ": ", bestHMMPerRegion[r]['name'], bestHMMPerRegion[r]['nSegments'])
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
plotting.plot_surf_roi(
        fsaverage["pial_" + side],
        plot_nSegs,
        hemi=side,
        view="lateral",
        bg_map=fsaverage["sulc_"+side],
        bg_on_data=True, darkness=0.25,
        title=f"Optimal HMM granularity, {side} hemi, '{task}'", title_font_size=14,
        colorbar = True, cmap = 'viridis', threshold=2
)
plt.show(block=blck)
#%% find max and min of bestNumEvents (exclude failing regions)
maxNumSegs = np.max([bestHMMPerRegion[r]['nSegments'] for r in bestHMMPerRegion if not np.isnan(bestHMMPerRegion[r]['score'])])
minNumSegs = np.min([bestHMMPerRegion[r]['nSegments'] for r in bestHMMPerRegion if not np.isnan(bestHMMPerRegion[r]['score'])])
print(f"Max number of events: {maxNumSegs}, Min number of events: {minNumSegs}")

##############################################################################
#################### MDL #####################################################
##############################################################################
#%% import word embeddings pickle
dataPath =  '/home/itzik/PycharmProjects/brainiak/docs/examples/eventseg/results/milkyway/'
task = 'milkyway'
hiddenLayer_data = pd.read_pickle(dataPath+'milkywaygpt2-xl-c_1024.pkl')

#%% PCA
k = 50
embeddings = np.array([np.array(a) for a in hiddenLayer_data]) # [np.array(a) for a in hiddenLayer_data]
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
# EBdata1 = np.load(r'partialMDL_EB_slicedBegin_milkyway.npz')
# EB_all1 = EBdata1['EBs']; bvals1 = EBdata1['bvals']; # tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']
# EBdata2 = np.load(r'supplMDL_EB_slicedBegin_milkyway.npz')
# EB_all2 = EBdata2['EBs']; bvals2 = EBdata2['bvals']; # tvals = EBdata['tvals'] ; segPts_all = EBdata['segPts']
# #%% Combine the two EBs ordered by the corresponding bvals
# bvals = np.unique(np.concatenate((bvals1, bvals2)))
# EB_all = np.zeros((len(bvals), YY.shape[-1]))
# for i, b in enumerate(bvals):
#     if b in bvals1:
#         ind1 = np.where(bvals1 == b)[0][0]
#         if b in bvals2:
#             print(b, "found in both")
#             ind2 = np.where(bvals2 == b)[0][0]
#             assert np.all(EB_all1[ind1,:]==EB_all2[ind2,:])
#         EB_all[i,:] = EB_all1[ind1,:]
#     elif b in bvals2:
#         ind2 = np.where(bvals2 == b)[0][0]
#         EB_all[i,:] = EB_all2[ind2,:]
#     else:
#         print("Error: b not found in either EBs")

#%% Run over multiple values of parameters b and tau
event_rep = 'const' ; sig = np.std(Y, axis=-1)
# extrabvals = np.concatenate((np.arange(332,336,1),np.arange(252,256,1),np.arange(151.5,161,0.5),np.arange(101,126,0.5)))
bvals = np.arange(101,500,5) # np.concatenate((np.arange(100,410,10),np.arange(425,525,25)))
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
np.savez('allMDL_EB_slicedBegin_'+task, EBs=EB_all, bvals=bvals)
#%%
#%% open previously saved numEB npy file
# EBdata = np.load(r'/home/itzik/PycharmProjects/EventBoundaries_deploy/numEB_monkey_narrative_.npz')
path=r'/home/itzik/PycharmProjects/brainiak/docs/examples/eventseg/'
path= r'/home/itzik/PycharmProjects/EventBoundaries/'
EBdata = np.load(path+'allMDL_EB_slicedBegin_'+task.lower()+'.npz')
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
#%% Look for weird back and forth transitions
for reg_i in bestHMMPerRegion:
    if reg_i in AlgFail:
        continue
    reg = bestHMMPerRegion[reg_i]
    if reg['nSegments'] - reg['nBoundaries'] != 1:
        print(reg_i, ":", reg['name'], " has ", reg['nSegments'], " states, but ", reg['nBoundaries'], " EB")

#%% For each region, find the b value closest to its num events
b_per_region = {}
inaccurate_b = []
for r in bestHMMPerRegion:
    if r in AlgFail:
        b_per_region[r] = np.nan
        continue
    numBounds = bestHMMPerRegion[r]['nBoundaries']
    while numBounds not in numEvents_b:
        inaccurate_b.append(r)
        print(f"Number of boundaries {numBounds} for region {bestHMMPerRegion[r]['name']} not found in EB hierarchy")
        numBounds += 1
        print(f"Adding {numBounds} instead")
        continue
    b_per_region[r] = numEvents_b[numBounds]
##########################################################################################
#%% #######################Time correlation vs. the HMM results###########################
##########################################################################################
#%% To compare to MDL model boundaries, we first need to transform MDL boundaries from story to TRs
textTiming_data = pd.read_excel('textTiming_by sentence_1.xlsx', sheet_name=3)
mw_indx = np.arange(1,265,4) # row numbers corresponding to millyWay story
word_timings = textTiming_data.iloc[mw_indx].iloc[:,7:]
words = [] ; sentence_len = [] ; sentence_onsets = []; sentence_offsets = []
onsets = [] ; words_onsets = []
for i, row in word_timings.iterrows():
    row_words = row.iloc[2:].dropna().values
    sentence_len.append(len(row_words))
    on  = row['from(TR)'] ; off = row['to(TR)']
    sentence_onsets.append(row['from(TR)']) ; sentence_offsets.append(row['to(TR)'])
    dt = (off - on+1)/len(row_words)
    for j,w in enumerate(row_words):
        w = w.lower()
        while len(w)>0 and w[-1] not in string.ascii_letters:
            w = w[:-1]
            # print(w)
        while len(w)>0 and w[0] not in string.ascii_letters:
            w = w[1:]
            # print(w)
        onsets.append(on + j*dt)
        words_onsets.append([w, on + j*dt])
        words.append(w)
    assert(on + j*dt - (off+1-dt) < 0.01)
for k,w in enumerate(words):
    assert(w==words_onsets[k][0])
# fix some anomalies
if words[108] == 'lamur':
    print("lamur")
    words[108] = 'lamour'
    words_onsets[108][0] = 'lamour'
if words[61] == 'consolt':
    print("consolt")
    words[61] = 'consult'
    words_onsets[61][0] = 'consult'
if words[508] == 'everybody':
    print("everybody")
    words[508] = 'everyone'
    words_onsets[508][0] = 'everyone'
if words[990] == 'vender':
    print("vender")
    words[990] = 'vending'
    words_onsets[990][0] = 'vending'

# also added the word 'the' before 'hypnosis' at index 553 row
# fixed typo in word 639
# added a word in 894-895 (I have instead of I've)
#%%
#%%
emptyTokens = []
tokens_ = np.load('milkyWayTokens.npy')
tokens = [t.lower() for t in tokens_ if t not in string.punctuation] # remove single char tokens who are just punctuation
for i, tt in enumerate(tokens):
    while len(tt) > 0 and tt[0] not in string.ascii_letters:
        tt = tt[1:]
    while len(tt) > 0 and tt[-1] not in string.ascii_letters:
        tt = tt[:-1]
    if len(tt) < 1:
        emptyTokens.append(i)
    tokens[i] = tt
#%%
#%% Print story with EB in Capital letters
# EB = np.where(EB_all[134-bvals[0]])[0]
# EB = model_ebs
for i, w in enumerate(tokens): #[initialCut_Y:]
    if i in emptyTokens:
        print(".", end='')
    if i in EB :
        print(w.upper(), end=' || ')
    else:
        print(w, end=' ')
    if i % 20 == 19:
        print('\n')
print()

#%% Time lock tokens in Y according to word onset TR
tokens_timings = np.zeros(Y.shape[-1]) #skips first token/word
k = 1 ; token_word = []
for i, tt in enumerate(tokens[1:]):
    if len(tt) < 1 and len(words[k]) >= 1: # stick empty token onto previous word
        tokens_timings[i] = words_onsets[k-1][1]
        token_word.append([tt, words_onsets[k-1][0]])
        continue
    sub, start_w, start_t = longest_common_substring(words[k], tt)
    # print("=====", sub, start_w, start_t, "=====")
    if (len(sub) < len(tt)) or (start_t != 0):
        print(f"pos {i+1} Word: {words[k]}, but token: {tt} doesn't match")
        print(f"perhaps word: {words[k-1]} ?")
        k -= 1
        sub, start_w, start_t = longest_common_substring(words[k], tt)
        # print("             --------", sub, start_w, start_t, "-----")
        if (len(sub) < len(tt) ) or (start_t != 0):
            print(f"ERROR: token {tt} in pos {i} matches neither {words[k+1]} nor predecessor {words[k]}")
            print("context tokens: ", np.array(tokens)[i-2:i+2], "// words:", np.array(words)[k-2:k+2])
            break
    tokens_timings[i] = words_onsets[k][1]
    token_word.append([tt, words_onsets[k][0]])
    k += 1
#%%
def cosine_similarity(v,u):
    return np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u))
#%%  compute cosine distance between each token and previous
cosDist = np.zeros(Y.shape[-1]-1)
for i in range(1, Y.shape[-1]):
    cosDist[i-1] = 1- cosine_similarity(Y[:,i], Y[:,i-1])
#%% load surprisal df
surprisal_data = pd.read_csv('results/milkyway/milkywaygpt2-xl-c_1024_surp_entr.csv')
surprisal = surprisal_data['Surprise'].values
entr = surprisal_data['Entropy'].values
#%% Per region, find relevnat MDL boundaries and compare
all_surf = np.array(surfL) if side == 'left' else np.array(surfR)
spMerge = False ; MRI_offset = 0 #todo delete after slicing data
correlation_results = {} ; gaussSig = 3 ; corrWin = 10
correlation_results['params'] = {'side': side, 'splitMerge': spMerge, 'gaussSig': gaussSig}
for label in atlas_destrieux['labels'][1:]:
    # fMRI part
    label_index = [atlas_destrieux['labels'].index(label)][0]
    if label_index in AlgFail:
        print( "---------------- skipping ",label_index, label, " due to HMM algorithm failure-------------------")
        continue
    regionInd = np.where(atlas_destrieux["map_"+side] == label_index)[0]
    all_region = all_surf[:,regionInd,:]
    assert (label == bestHMMPerRegion[label_index]['name'])
    print(">>>>>>>>Region: ", label)
    # nSeg = bestHMMPerRegion[label_index]['nSegments']
    # # todo HMM algorithm is not numerically stable: running again results in (slightly) different boundaries
    # ev = brainiak.eventseg.event.EventSegment(nSeg, split_merge=spMerge)
    # ev.fit(all_region.mean(0).T)
    # segments = np.argmax(ev.segments_[0], axis=1)
    # HMM_ebs = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
    HMM_ebs = bestHMMPerRegion[label_index]['boundaries']
    # MDL part
    b = b_per_region[label_index][0]
    b_ind = np.where(bvals == b)[0][0]
    model_ebs = np.where(EB_all[b_ind])[0]
    model_ebs_in_TR_space = tokens_timings[model_ebs]
    # round down to nearest TR (commented out: round to nearest)
    model_ebs_in_TR = np.array([int(bb) for bb in model_ebs_in_TR_space]) + MRI_offset  # np.rint(model_ebs_in_TR_space).astype(int)
    print(f"        MODEL====={len(model_ebs_in_TR)}=====>", model_ebs_in_TR)
    print(f"        HMM====={len(HMM_ebs)}=====>", HMM_ebs)
    model_1hot = np.zeros(all_surf.shape[-1]);  model_1hot[model_ebs_in_TR] = 1
    hmm_1hot = np.zeros(all_surf.shape[-1]); hmm_1hot[HMM_ebs] = 1
    # compute cross correlation between model and HMM boundaries 1 hot vectors
    cor = np.correlate(hmm_1hot, model_1hot, "same")[len(model_1hot)//2-corrWin:len(model_1hot)//2+corrWin+1] #first vec lags behind
    # folllowing Baldassano et al. find the number of model boundaries that are between 3 time points before and after an HMM boundary
    close_cor = count_close_ones(model_1hot, hmm_1hot, 3)/sum(model_1hot)
    close_cor_exc = count_close_ones(model_1hot, hmm_1hot, 3, exclusive=True)/sum(model_1hot)
    # Gaussian Smoothing correlation
    model_smooth = gaussian_filter1d(model_1hot, sigma=gaussSig, mode='constant', cval=0)
    hmm_smooth = gaussian_filter1d(hmm_1hot, sigma=gaussSig, mode='constant', cval=0)
    correlation_results[label_index] = {'name' : label, 'boundaries': len(HMM_ebs),
                                        'crossCor' : np.max(cor)/min(sum(hmm_1hot), sum(model_1hot)),
                                        'lag' : list((np.argwhere(cor == np.amax(cor)) - corrWin).flatten()), # pos lag means model leads HMM (good)
                                        'close': close_cor,
                                        'close_exc': close_cor_exc,
                                        'smoothMSE': MSE(model_smooth, hmm_smooth),
                                        'smoothDTW' : DTW(model_smooth, hmm_smooth)}

#%%
Good = [1,4,5,6,10,17,18,23,34,36,38,39,43,44,46,52,59,72]
goodLabels = [atlas_destrieux['labels'][g] for g in Good]
countFigs = 0
for label, label_index in zip(goodLabels, Good):
    if countFigs >14:
        break
    print("============", label, correlation_results[label_index]['smoothMSE'], "============")
    HMM_ebs = bestHMMPerRegion[label_index]['boundaries']
    b = b_per_region[label_index][0]
    b_ind = np.where(bvals == b)[0][0]
    model_ebs = np.where(EB_all[b_ind])[0]
    model_ebs_in_TR_space = tokens_timings[model_ebs]
    model_ebs_in_TR = np.array(
        [int(bb) for bb in model_ebs_in_TR_space]) + MRI_offset  # np.rint(model_ebs_in_TR_space).astype(int)
    print(f"        MODEL====={len(model_ebs_in_TR)}=====>", model_ebs_in_TR)
    print(f"        HMM====={len(HMM_ebs)}=====>", HMM_ebs)
    model_1hot = np.zeros(all_surf.shape[-1]);    model_1hot[model_ebs_in_TR] = 1
    hmm_1hot = np.zeros(all_surf.shape[-1]);    hmm_1hot[HMM_ebs] = 1
    model_smooth = gaussian_filter1d(model_1hot, sigma=gaussSig, mode='constant', cval=0) #  moving_average_smooth(model_1hot, window_size=6)
    hmm_smooth = gaussian_filter1d(hmm_1hot, sigma=gaussSig, mode='constant', cval=0) #moving_average_smooth(hmm_1hot, window_size=6)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.plot(model_smooth, color='red')
    plt.plot(hmm_smooth, color='blue')
    # add 1hot vectors as bars in the background
    plt.bar(range(len(model_1hot)), model_1hot * max(model_smooth), color='red', alpha=0.2, width=1,label='Model')
    plt.bar(range(len(hmm_1hot)), hmm_1hot * max(model_smooth), color='blue', alpha=0.2, width=1, label='HMM')
    plt.legend();
    plt.title(f"{label}: gauss_sig={gaussSig}, close score={correlation_results[label_index]['close_exc']:.3f}")
    plt.show() ; countFigs += 1

#%% Focusing on 1 region

label = b'S_temporal_sup'#b'S_temporal_transverse'
# label = b'G_oc-temp_med-Lingual'
label_index = [atlas_destrieux['labels'].index(label)][0]
regionInd = np.where(atlas_destrieux["map_"+side] == label_index)[0]
all_surf = np.array(surfL) if side == 'left' else np.array(surfR)
all_region = all_surf[:,regionInd,:]
nEvents = bestHMMPerRegion[label_index]['nSegments'] ; spMerge = False
ev = brainiak.eventseg.event.EventSegment(nEvents, split_merge=spMerge)
ev.fit(all_region.mean(0).T)
segments = np.argmax(ev.segments_[0], axis=1)
HMM_ebs = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0] # todo note: Returns to event 37 amidst 38
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
plt.hist(measure[HMM_ebs], bins=10, label='boundary') ; plt.xlabel(title); plt.ylabel('EB count')
plt.hist(measure[~HMM_ebs], bins=10, alpha=0.5, label='mid-segment')
plt.legend();plt.show()
figsToPDF.append(plt.gcf())
#%% Extract relevant MDL EBs
b = b_per_region[label_index][0]
b_ind = np.where(bvals == b)[0][0]
model_ebs = np.where(EB_all[b_ind])[0]
model_ebs_in_TR_space = tokens_timings[model_ebs]
# round down to nearest TR (commented out: round to nearest)
model_ebs_in_TR = np.array([int(bb) for bb in model_ebs_in_TR_space]) #  np.rint(model_ebs_in_TR_space).astype(int)

dt= Y.shape[-1]/all_surf.shape[-1] # TR duration in words [words per TR]
# HMM_ebs_inText = np.array([int(bb*dt) for bb in HMM_ebs])
print(f"MODEL====={len(model_ebs_in_TR)}=====>", model_ebs_in_TR)
print(f"HMM====={len(HMM_ebs)}=====>", HMM_ebs)
#%%
model_1hot = np.zeros(all_surf.shape[-1]); model_1hot[model_ebs_in_TR] = 1
hmm_1hot = np.zeros(all_surf.shape[-1]); hmm_1hot[HMM_ebs] = 1
#%% the story actually started at TR=15, and ended at TR=283. For all the subjects.
# except Subj18 and Subj27 who started at TR=11 and ended at TR=279
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
gaussSig = 5
model_smooth = gaussian_filter1d(model_1hot, sigma=gaussSig, mode='constant', cval=0)
hmm_smooth = gaussian_filter1d(hmm_1hot, sigma=gaussSig, mode='constant', cval=0)
plt.plot(model_smooth, label='Model', color='red')
plt.plot(hmm_smooth, label='HMM', color='blue')
# add 1hot vectors as bars in the background
plt.bar(range(len(model_1hot)), model_1hot * max(model_smooth), color='red', alpha=0.2, width=1)
plt.bar(range(len(hmm_1hot)), hmm_1hot * max(model_smooth), color='blue', alpha=0.2, width=1)
plt.legend(); plt.title(f"{label}: Gaussian smoothing, sig={gaussSig}"); plt.show()
#%% Plot both boundaries
fig = plt.figure()
waxis = np.arange(0, len(all_surf[0].T))
plt.vlines(model_ebs_in_TR, 1.5, 2.5, linewidth=3, alpha=0.7, color='grey', label='Model')
# plt.vlines(hmmBoundaries, 1.7, 2.4, linewidth=1, color=wa.color_palettes['Darjeeling Limited'][spMerge][1], label='HMM')
plt.scatter(HMM_ebs, 2*np.ones_like(HMM_ebs), marker='|', linewidths=3, s=500, \
                color=wa.color_palettes['Darjeeling Limited'][spMerge][1], label='HMM'+'+merge'*spMerge)
plt.legend(); plt.xlim(left=-3); plt.show()
figsToPDF.append(plt.gcf())
#%% plot model boundaries against some measure of likelihhod to transition
fig = plt.figure(figsize=(12, 5))
measure = np.array(modal_diffs); title = 'modal difference'
# measure = segmentsVar; title = 'segment variance'model_ebs_inTR = np.rint(model_ebs/dt).astype(int)
plt.plot(measure, label=title, color='grey'); plt.xlabel('Time point'); plt.ylabel(title)
plt.scatter(model_ebs_in_TR, measure[model_ebs_in_TR], color='red', label='Model boundaries')
plt.scatter(HMM_ebs, measure[HMM_ebs], color='blue', marker="*", label='HMM boundaries')
for bb in HMM_ebs:
    plt.plot([bb-1, bb, bb+1], [measure[bb-1],measure[bb],measure[bb+1]], color='blue', linewidth=2)
plt.legend(); plt.show()
figsToPDF.append(plt.gcf())
###########################################################################
###########################################################################
#%%  Sanity check with regions defined in Baldassano et al. 2017
case_name = 'Angular Gyr. H-O'#"Heschl's Gyrus H-O" #'SPMg H-O'# "Yeo 15-16" #  'Angular Gyr. H-O'
all_TR = BOLD_sliced[10]
if "Yeo" in case_name:
    atlas = datasets.fetch_atlas_yeo_2011().thick_17
    yeo_img = image.load_img(atlas)
    yeo_img_resampled = image.resample_to_img(yeo_img, all_TR, interpolation="nearest")
    label_index = [15, 16] ; label = case_name
    bool_masks = yeo_img_resampled.get_fdata() == label_index
    bool_mask = np.sum(bool_masks, axis=-1) > 0
    mask_img = nl.image.new_img_like(yeo_img_resampled, bool_mask)
    print(str(label_index), "mask // Shape:", bool_mask.shape, ", # voxels: ", np.sum(bool_mask))
else:
    if 'H-O' in case_name:
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
    else:
        input("define atlas >>>>")
        atlas =  datasets.fetch_atlas_juelich("maxprob-thr0-2mm") #
    print(f"The atlas contains {len(atlas.labels) - 1} non-overlapping regions")
    atlas.maps = image.resample_to_img(atlas.maps, all_TR, interpolation="nearest")
    label = "Angular Gyrus".lower()#'Supramarginal Gyrus, anterior division'.lower()# 'precuneous cortex' #'angular gyrus';
    label_index = [atlas.labels.index(l) for l in atlas.labels if label in l.lower()]
    bool_mask = reduce(lambda x, y: x + y, [(atlas.maps.get_fdata() == i) for i in label_index])
    mask_img = nl.image.new_img_like(atlas.maps, bool_mask)
    print(label, "mask // Shape:", bool_mask.shape, ", # voxels: ", np.sum(bool_mask))
plotting.plot_roi(mask_img, title=label, display_mode='tiled', draw_cross=False, cmap='viridis_r')
plt.show(block=blck)
#%%
initialCut = 18 ; initialCut_Y = 90
maskedBOLD = [masking.apply_mask(BB, mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True) for BB in BOLD_sliced]
maskedBOLD = np.array([BB[initialCut:,...].T for BB in maskedBOLD])
#%% Need toc check statistical significance for whole brain averaged over regions
# allBrain = np.array([BB.get_fdata() for BB in BOLD_sliced])
# allBrain = allBrain.reshape((allBrain.shape[0],-1,allBrain.shape[-1]))
#%% Find the best number of HMM segments for the region
segments_vals = np.arange(10, 50, 1)
score = [] ; nPerm = 1000 ; w = 5 ; nSubj = len(files)
within_across_all = np.zeros((len(segments_vals),nSubj, nPerm+1))
for i,nSegments in enumerate(segments_vals):
    within_across_all[i], _ = within_across_corr(maskedBOLD, nSegments, w, nPerm, verbose=0)
    score.append(within_across_all[i,:,0].mean())
    print(f"Number of HMM segments: {nSegments}, HMM score: {score[-1]}")
#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt.plot(segments_vals, score, marker='o', color='black'); plt.title('# segments comparison [sliced begin.], {}'.format(label))
plt.xlabel('Number of events'); plt.ylabel('mean(within-across) correlation'); ax.set_xticks(segments_vals)
plt.axhline(np.max(score), color='black', linestyle='--', linewidth=0.5)
plt.show(block = blck)
#%% For the best number of events, violin plot of within-across correlation
best_ind = np.argmax(score)
nSeg = segments_vals[best_ind]
plt.figure(figsize=(6,16))
plt.violinplot(within_across_all[best_ind,:,1:].mean(0), showextrema=True) # permuted
plt.scatter(1, within_across_all[best_ind,:,0].mean(0), label= 'Real events') # real
plt.gca().xaxis.set_visible(False)
plt.ylabel('Within vs across boundary correlation'); plt.legend()
plt.title('{}:\nHeld-out for HMM with {} events [sliced] ({} perms)'.format(case_name, nSeg, nPerm))
plt.show(block = blck)
# figsToPDF.append(plt.gcf())
#%% Extract HMM boundaries
ev = brainiak.eventseg.event.EventSegment(nSeg)
ev.fit(maskedBOLD.mean(0).T)
segments = np.argmax(ev.segments_[0], axis=1)
HMM_ebs = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
#%%# MDL part - find relevant boundaries based on language embeddings
b = numEvents_b[len(HMM_ebs)+2][0]
if initialCut>0:
    YY = Y[:,initialCut_Y:]
    tokens_timings_sliced = tokens_timings[initialCut_Y:] - initialCut
    model_ebs, _ =  EB_split(YY, b=b, rep='const', sig=np.var(YY))
    model_ebs_in_TR_space = tokens_timings_sliced[model_ebs]
    hmm_ebs_in_word_space = [np.where((tokens_timings_sliced >= eb) * (tokens_timings_sliced < eb + 1)) for eb in
                             HMM_ebs]
    # hmm_ebs_in_word_space = [np.where((tokens_timings>=eb)*(tokens_timings<eb+1)) for eb in HMM_ebs]
else:
    b_ind = np.where(bvals == b)[0][0]
    model_ebs = np.where(EB_all[b_ind])[0]
    model_ebs_in_TR_space = tokens_timings[model_ebs]

# round down to nearest TR (commented out: round to nearest)
model_ebs_in_TR = np.array([int(bb) for bb in model_ebs_in_TR_space]) # np.rint(model_ebs_in_TR_space).astype(int)
print(f"{case_name}:        MODEL====={len(model_ebs_in_TR)}=====>", model_ebs_in_TR)
print(f"                    HMM====={len(HMM_ebs)}=====>", HMM_ebs)
#%% Analyze agreement of embeddings' MDL boundaries, with fMRI HMM boundaries
corrWin = 50 ; gaussSig = 3
space = 'TRs'
if space == 'TRs':
    model_1hot = np.zeros(maskedBOLD.shape[-1]) ;  model_1hot[model_ebs_in_TR] = 1
    hmm_1hot = np.zeros(maskedBOLD.shape[-1]); hmm_1hot[HMM_ebs] = 1
    close_dist = np.rint(maskedBOLD.shape[-1] / (2*len(HMM_ebs)))
elif space == 'words':
    model_1hot = np.zeros(YY.shape[-1]);  model_1hot[model_ebs] = 1
    hmm_1hot = np.zeros(YY.shape[-1])
    close_dist = np.rint(YY.shape[-1] / (2*len(HMM_ebs)))
    for eb in hmm_ebs_in_word_space:
        hmm_1hot[eb] = 1
else:
    print("error")
# model_ebs_in_TR_space = np.around(model_ebs_in_TR_space, decimals=1)
# indxs = np.arange(0,maskedBOLD.shape[-1],0.1)
# model_1hot = np.zeros(maskedBOLD.shape[-1]*10) ;  model_1hot[(model_ebs_in_TR_space*10).astype(int)] = 1
# hmm_1hot = np.zeros(maskedBOLD.shape[-1]*10)
# for eb in HMM_ebs:
#     hmm_1hot[eb*10:eb*10+10] = 1
# compute cross correlation between model and HMM boundaries 1 hot vectors
cor = np.correlate(hmm_1hot, model_1hot, "same")[len(model_1hot)//2-corrWin:len(model_1hot)//2+corrWin+1] #first vec lags behind
# folllowing Baldassano et al. find the number of model boundaries that are between 3 time points before and after an HMM boundary
close_cor = count_close_ones(model_1hot, hmm_1hot, close_dist)/sum(model_1hot)
close_cor_exc = count_close_ones(model_1hot, hmm_1hot, close_dist, exclusive=True)/sum(model_1hot)
# Gaussian Smoothing correlation
model_smooth = gaussian_filter1d(model_1hot, sigma=gaussSig, mode='constant', cval=0)
hmm_smooth = gaussian_filter1d(hmm_1hot, sigma=gaussSig, mode='constant', cval=0)
print(case_name, ':', 'boundaries', len(HMM_ebs), f'({space} space)'
                                    '\ncrossCor', np.max(cor)/min(sum(hmm_1hot), sum(model_1hot)),
                                    '\nlag', np.argwhere(cor == np.amax(cor)) - corrWin, # pos lag means model leads HMM (good)
                                    '\nclose', close_cor,
                                    '\nclose_exc', close_cor_exc,
                                    '\nsmoothMSE', MSE(model_smooth, hmm_smooth),
                                    '\nsmoothDTW' , DTW(model_smooth, hmm_smooth))
#%% Plot the boundaries
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# plt.plot(model_smooth,  color='red')
# plt.plot(hmm_smooth,  color='blue')
# add 1hot vectors as bars in the background
if space == 'TRs':
    plt.bar(range(len(model_1hot)), model_1hot * max(model_smooth), color='red', alpha=0.2, width=1,label=f'model {np.sum(model_1hot, dtype=int)}')
    plt.bar(range(len(hmm_1hot)), hmm_1hot * max(model_smooth), color='blue', alpha=0.2, width=1,label=f'HMM {len(HMM_ebs)}')
elif space == 'words':
    plt.fill_between(range(len(model_1hot)), model_1hot, color='red', alpha=0.2,label=f'model {np.sum(model_1hot, dtype=int)}')
    plt.fill_between(range(len(hmm_1hot)), hmm_1hot, color='blue', alpha=0.2,label=f'HMM {len(HMM_ebs)}')
# plt.scatter(np.around(tokens_timings[initialCut_Y:]-initialCut,decimals=1), 0.5*np.ones(len(tokens_timings[initialCut_Y:])),\
#             color='black',label='words', s=5, marker='+')
plt.legend(); plt.xlabel(space); plt.gca().axes.get_yaxis().set_visible(False)
plt.title(f"{case_name}: gauss_sig={gaussSig}, match= {close_cor_exc:.2f}, loose match = {close_cor:.2f} [sliced beginning]")
plt.show()
#%% Compare HMM boundaries to cosine distance etc.
sup_measure  = entr
if space != 'words':
    input("Make sure 1-hot vectors are in word space....")
sup_measure_sliced = sup_measure[initialCut_Y-1:]
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# add 1hot vectors as bars in the background
plt.bar(range(len(hmm_1hot)), hmm_1hot * max(sup_measure_sliced+0.5), color='blue', alpha=0.2, width=1,label=f'HMM {len(HMM_ebs)}')
plt.plot(gaussian_filter1d(sup_measure_sliced,  sigma=3, mode='constant', cval=0), label='cosine dist, smoothed', color='black')
# for i in range(len(hmm_1hot)):
#     if hmm_1hot[i]:
#         plt.scatter(i,cosDist_sliced[i],  color='green', s=10)
plt.legend(); plt.xlabel('tokens'); plt.ylabel('-log(P) surprisal')
plt.title(f"{case_name}: EBs vs surprisal [sliced beginning]")
plt.show()
# todo write some quantitative measure of accuracy of surprisal in predicting boundaries
# compute mean cos dist across boundaries vs non-boundaries
#%% Apply MDL with the proper b to predicting BOLD data
# b = numEvents_b[len(HMM_ebs)+2][0]
# todo improve this by removing values from EB_nums_ and correspomding rows from within_across whenever skipping (instead of saving "actual")
nPerm=1000 ; w=5 ; nSubj = len(files)
EB_nums_actual = []
EB_nums_ = np.arange(len(HMM_ebs)-15, len(HMM_ebs)+21,1)
within_across_MDL_all = np.zeros((len(EB_nums_),nSubj, nPerm+1)) ; scoreMDL = []
for i,num in enumerate(EB_nums_):
    if num not in numEvents_b:
        print("******skipping ", num, "events")
        continue
    print("************", num, "events************"+ "===>(closest num events to HMM)"*(num == len(HMM_ebs)))
    b_ = numEvents_b[num][0] ; EB_nums_actual.append(num)
    within_across_MDL_all[i], _ = within_across_corr(maskedBOLD, len(model_ebs), w=w, nPerm=nPerm, verbose=0, MDL_b=b_)
    scoreMDL.append(within_across_MDL_all[i, :, 0].mean())
    print(f"MDL within_across score: {scoreMDL[-1]}")
 #%%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(EB_nums_actual, scoreMDL, '--', marker='o', color='black');
plt.title('# b values comparison [sliced begin.], {}'.format(label))
plt.xlabel('num events');
plt.ylabel('mean(within-across) correlation');
ax.set_xticks(EB_nums_actual)
plt.axhline(np.max(scoreMDL), color='black', linestyle='--', linewidth=0.5)
plt.show(block=blck)
# %% For the best number of events, violin plot of within-across correlation
nEB = EB_nums_actual[np.argmax(scoreMDL)] # 37 for AG but try 27 todo change to where it starts to plateau
best_ind = np.where(EB_nums_==nEB)[0][0]
plt.figure(figsize=(6, 16))
plt.violinplot(within_across_MDL_all[best_ind, :, 1:].mean(0), showextrema=True)  # permuted
plt.scatter(1, within_across_MDL_all[best_ind, :, 0].mean(0), label='Real events')  # real
plt.gca().xaxis.set_visible(False)
plt.ylabel('Within vs across boundary correlation'); plt.legend()
plt.annotate(f"{scoreMDL[np.argmax(scoreMDL)]:.2f}", (1, scoreMDL[np.argmax(scoreMDL)]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('{}:\nLOO for MDL b={}, {} EBs [sliced] ({} perms)'.format(case_name, numEvents_b[nEB][0], nEB, nPerm))
plt.show(block=blck)
#%% NOw compare to timings on words using the same b
best_b = numEvents_b[nEB][0]
bold_boundaries, _ = EB_split(maskedBOLD.mean(0), b=best_b, rep='const', sig=np.var(maskedBOLD)) ; bold_boundaries = np.array(bold_boundaries)
word_boundaries, _ =  EB_split(YY, b=best_b, rep='const', sig=np.var(YY)) ; word_boundaries = np.array(word_boundaries)
word_boundaries_in_TR_space = tokens_timings_sliced[word_boundaries]
bold_boundaries_in_word_space = [np.where((tokens_timings_sliced >= eb) * (tokens_timings_sliced < eb + 1)) for eb in
                             bold_boundaries]
# round down to nearest TR (commented out: round to nearest)

word_boundaries_in_TR = np.array([int(bb) for bb in word_boundaries_in_TR_space])
print(f"{case_name}: b={best_b}        MDL on words====={len(word_boundaries_in_TR)}=====>", word_boundaries_in_TR)
print(f"                    MDL on fMRI====={len(bold_boundaries)}=====>", bold_boundaries)
#%% Predict on BOLd
WvA = BOLDscore(word_boundaries_in_TR, maskedBOLD, w=5, verbose=2)
WvA_score = WvA[:,0]
print(f"MDL on words vs fMRI: {WvA_score.mean()}")
#%%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# plot distibution of WvA score
plt.hist(WvA_score, bins=5, label='WvA score')
plt.show()
#%% Analyze agreement of embeddings' MDL boundaries, with fMRI HMM boundaries
corrWin = 50 ; gaussSig = 3
space = 'TRs'
if space == 'TRs':
    words_1hot = np.zeros(maskedBOLD.shape[-1]) ;  words_1hot[word_boundaries_in_TR] = 1
    bold_1hot = np.zeros(maskedBOLD.shape[-1]); bold_1hot[bold_boundaries] = 1
    close_dist = np.rint(maskedBOLD.shape[-1]/(2*len(bold_boundaries)))
elif space == 'words':
    words_1hot = np.zeros(YY.shape[-1]);  words_1hot[word_boundaries] = 1
    bold_1hot = np.zeros(YY.shape[-1])
    close_dist = np.rint(YY.shape[-1]/(2*len(bold_boundaries)))
    for eb in bold_boundaries_in_word_space:
        bold_1hot[eb] = 1
else:
    print("error")
# model_ebs_in_TR_space = np.around(model_ebs_in_TR_space, decimals=1)
# indxs = np.arange(0,maskedBOLD.shape[-1],0.1)
# model_1hot = np.zeros(maskedBOLD.shape[-1]*10) ;  model_1hot[(model_ebs_in_TR_space*10).astype(int)] = 1
# hmm_1hot = np.zeros(maskedBOLD.shape[-1]*10)
# for eb in HMM_ebs:
#     hmm_1hot[eb*10:eb*10+10] = 1
# compute cross correlation between model and HMM boundaries 1 hot vectors
cor = np.correlate(bold_1hot, words_1hot, "same")[len(words_1hot)//2-corrWin:len(words_1hot)//2+corrWin+1] #first vec lags behind
# folllowing Baldassano et al. find the number of model boundaries that are between 3 time points before and after an HMM boundary
window = YY.shape[-1]/len(bold_boundaries)
close_cor = count_close_ones(words_1hot, bold_1hot, close_dist)/sum(words_1hot)
close_cor_exc = count_close_ones(words_1hot, bold_1hot, close_dist, exclusive=True)/sum(words_1hot)
# Gaussian Smoothing correlation
words_smooth = gaussian_filter1d(words_1hot, sigma=gaussSig, mode='constant', cval=0)
bold_smooth = gaussian_filter1d(bold_1hot, sigma=gaussSig, mode='constant', cval=0)
print(case_name, ':', 'boundaries', len(HMM_ebs), f'({space} space)'
                                    '\ncrossCor', np.max(cor)/min(sum(bold_1hot), sum(words_1hot)),
                                    '\nlag', np.argwhere(cor == np.amax(cor)) - corrWin, # pos lag means model leads HMM (good)
                                    '\nclose', close_cor,
                                    '\nclose_exc', close_cor_exc,
                                    '\nsmoothMSE', MSE(words_smooth, bold_smooth),
                                    '\nsmoothDTW' , DTW(words_smooth, bold_smooth))
#%% Plot the boundaries
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# plt.plot(words_smooth,  color='orange')
# plt.plot(bold_smooth,  color='teal')
# add 1hot vectors as bars in the background
if space == 'TRs':
    plt.bar(range(len(words_1hot)), words_1hot * max(words_smooth), color='orange', alpha=0.2, width=1,label=f'words {np.sum(words_1hot, dtype=int)}')
    plt.bar(range(len(bold_1hot)), bold_1hot * max(words_smooth), color='teal', alpha=0.2, width=1,label=f'Brain {len(bold_boundaries)}')
elif space == 'words':
    plt.fill_between(range(len(words_1hot)), words_1hot, color='orange', alpha=0.2,label=f'model {np.sum(words_1hot, dtype=int)}')
    plt.fill_between(range(len(bold_1hot)), bold_1hot, color='teal', alpha=0.2,label=f'HMM {len(bold_boundaries)}')
# plt.scatter(np.around(tokens_timings[initialCut_Y:]-initialCut,decimals=1), 0.5*np.ones(len(tokens_timings[initialCut_Y:])),\
#             color='black',label='words', s=5, marker='+')
plt.legend(); plt.xlabel(space); plt.gca().axes.get_yaxis().set_visible(False)
plt.title(f"[MDL-MDL] {case_name}: gauss_sig={gaussSig}, match= {close_cor_exc:.2f}, loose match = {close_cor:.2f} [sliced beginning]")
plt.show()

#%% HMM - HMM
ev_w = brainiak.eventseg.event.EventSegment(nSeg)
ev_w.fit(YY.T)
#%%
segmentsY = np.argmax(ev_w.segments_[0], axis=1)
HMM_ebsY = np.where(np.diff(np.argmax(ev_w.segments_[0], axis=1)))[0]
wordHMM_ebs_in_TR_space = tokens_timings[initialCut_Y:][HMM_ebsY] - initialCut
wordHMM_ebs_in_TR = np.array([int(bb) for bb in wordHMM_ebs_in_TR_space]) # np.rint(model_ebs_in_TR_space).astype(int)
print(f"{case_name}:        words HMM====={len(wordHMM_ebs_in_TR)}=====>", wordHMM_ebs_in_TR)
print(f"                    BOLD HMM====={len(HMM_ebs)}=====>", HMM_ebs)
#%%
wordHMM_1hot = np.zeros(maskedBOLD.shape[-1]);  wordHMM_1hot[wordHMM_ebs_in_TR] = 1
hmm_1hot = np.zeros(maskedBOLD.shape[-1]); hmm_1hot[HMM_ebs] = 1
closeness = maskedBOLD.shape[-1]/len(HMM_ebs)
corrWin = 50 ; gaussSig = 3
# compute cross correlation between model and HMM boundaries 1 hot vectors
cor_w = np.correlate(hmm_1hot, wordHMM_1hot, "same")[len(wordHMM_1hot)//2-corrWin:len(wordHMM_1hot)//2+corrWin+1] #first vec lags behind
# folllowing Baldassano et al. find the number of model boundaries that are between 3 time points before and after an HMM boundary
close_cor_w = count_close_ones(wordHMM_1hot, hmm_1hot, 3)/sum(wordHMM_1hot)
close_cor_exc_w = count_close_ones(wordHMM_1hot, hmm_1hot, 3, exclusive=True)/sum(wordHMM_1hot)
# Gaussian Smoothing correlation
wordHMM_smooth = gaussian_filter1d(wordHMM_1hot, sigma=gaussSig, mode='constant', cval=0)
hmm_smooth = gaussian_filter1d(hmm_1hot, sigma=gaussSig, mode='constant', cval=0)
print(case_name, ': ***HMM vs HMM***', 'boundaries', len(HMM_ebs),
                                    '\ncrossCor', np.max(cor_w)/min(sum(hmm_1hot), sum(wordHMM_1hot)),
                                    '\nlag', np.argwhere(cor_w == np.amax(cor_w)) - corrWin, # pos lag means model leads HMM (good)
                                    '\nclose', close_cor_w,
                                    '\nclose_exc', close_cor_exc_w,
                                    '\nsmoothMSE', MSE(wordHMM_smooth, hmm_smooth),
                                    '\nsmoothDTW' , DTW(wordHMM_smooth, hmm_smooth))
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
# plt.plot(wordHMM_smooth,  color='crimson')
# plt.plot(hmm_smooth,  color='darkblue')
# add 1hot vectors as bars in the background
plt.bar(range(len(wordHMM_1hot)), wordHMM_1hot * max(wordHMM_smooth), color='crimson', alpha=0.2, width=1,label=f'words {len(wordHMM_ebs_in_TR)}')
plt.bar(range(len(hmm_1hot)), hmm_1hot * max(wordHMM_smooth), color='darkblue', alpha=0.2, width=1,label=f'brain {len(HMM_ebs)}')
plt.legend()
plt.title(f"[HMM-HMM]{case_name}: match= {close_cor_exc_w:.2f}, loose match = {close_cor_w:.2f} [sliced beginning]")
plt.show()
#%%
figsToPDF.append(plt.gcf())
#%%
savefig(figsToPDF, os.getcwd(), savename='AG_HO_dualMDL_dualHMM', tight=False, prefix="figures")

#%%
lastb = numEvents_b[2][0] # lowest b that gives this number of events
for nn in np.arange(2, 52, 1):
    if nn not in numEvents_b.keys():
        b_new = lastb ; nevents = nn-1
        while nevents < nn and b_new > lastb-1:
            b_new -= 0.25
            print("trying to get ", nn, "events with b=", b_new)
            bn, _ = EB_split(Y, b=b_new, rep='const', sig=np.var(Y))
            nevents = len(bn)
            if nevents == nn:
                print("====> got ", nevents, "events!!!!!!!!!!")
            else:
                print("***still got ", nevents, "events")

    else:
        lastb = numEvents_b[nn][0]