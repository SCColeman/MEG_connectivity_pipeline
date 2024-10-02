#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess raw ds files.

@author: sebastiancoleman
"""

import mne
import os.path as op
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from joblib import Parallel, delayed

#%% functions

def isoutlier(data):
    
    data = np.asarray(data)  # Convert to numpy array
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median absolute deviation

    if mad == 0:
        return np.zeros_like(data, dtype=bool)  # No variation means no outliers

    # Calculate the modified Z-score
    modified_z_scores = 0.6745 * (data - median) / mad
    outliers = np.abs(modified_z_scores) > 3.5  # Default threshold

    return outliers

#%% paths

subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_VNS/subjects_dir'
data_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/derivatives2'

#%% load data

dir_list = sorted(glob(op.join(subjects_dir, '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]')))
subjects = [op.basename(d) for d in dir_list]

def preprocess_subject(subject):
    
    if not op.exists(op.join(deriv_path, subject)):
        os.makedirs(op.join(deriv_path, subject))
        
    fnames = sorted(glob(op.join(data_path, subject, '*EPI*.ds')))
    for f, fname in enumerate(fnames):
        
        # load
        basename = op.split(fname)[-1][:-3]
        raw = mne.io.read_raw_ctf(fname, preload=True, verbose=False)
        
        # picks
        raw.drop_channels(raw.info['bads'])
        meg_ch_names = raw.copy().pick('mag', verbose=False).ch_names
        picks = meg_ch_names + ['EKG']
        raw.pick(picks, verbose=False)
        
        # filtering
        raw.resample(250, verbose=False)
        raw.filter(1, 100, verbose=False)
        raw.notch_filter(60, verbose=False)
        
        # bad channels
        noise_data = raw.copy().pick('mag').filter(80, 100, verbose=False).get_data()
        chan_var = np.var(noise_data, 1)
        chan_var_z = zscore(chan_var)
        bad_chan_i = [i for i in range(len(chan_var_z)) if chan_var_z[i] > 3]
        bad_chan = []
        for i in bad_chan_i:
            bad_chan.append(raw.copy().pick('mag').ch_names[i])
            
        if bad_chan:
            raw.info['bads'] = bad_chan
            raw.drop_channels(raw.info['bads'])
        
        # bad segments
        noise = raw.copy().pick('mag').filter(80, 100, verbose=False).get_data()
        variances = np.array([np.mean(np.var(noise[:, i:i+500], axis=1)) for i in np.arange(0, noise.shape[1]-500+1, 500)])
        outliers = isoutlier(variances)
        annotations = raw.annotations
        for i, ind in enumerate(np.arange(0, noise.shape[1], 500)):
            if outliers[i]:
                onset = raw.times[ind]
                duration = 500 * (1/raw.info['sfreq'])
                description = 'BAD_segment'
                annotations += mne.Annotations(onset, duration, description, orig_time=annotations.orig_time)
        raw.set_annotations(annotations, verbose=False)
        
        # ICA
        ekg = raw.copy().pick('EKG').get_data()
        if np.sum(ekg) > 0:
            ica = mne.preprocessing.ICA(n_components=30, max_iter=10000, verbose=False)
            ica.fit(raw.copy().pick('mag'), reject_by_annotation=True, verbose=False)
            ecg_comps, _ = ica.find_bads_ecg(raw, 'EKG', verbose=False)
            ica.exclude = ecg_comps
            ica.apply(raw, verbose=False)
        raw.pick('mag', verbose=False)
        
        # save
        preproc_fname = basename + '_preproc-raw.fif'
        raw.save(op.join(deriv_path, subject, preproc_fname), overwrite=True, verbose=False)

# parallelise
Parallel(n_jobs=10)(delayed(preprocess_subject)(subject) for subject in subjects)

