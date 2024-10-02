#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate amplitude connectivity.

@author: sebastiancoleman
"""

import mne
import numpy as np
import os.path as op
from nilearn import image, datasets, plotting
import pandas as pd
from glob import glob
from joblib import Parallel, delayed

#%% set up paths

root = r"/d/gmi/1/sebastiancoleman/MEG_VNS"
data_path = op.join(root, "data")
deriv_path = op.join(root, "derivatives2")

dir_list = sorted(glob.glob(deriv_path + '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
subjects = [op.basename(d) for d in dir_list]

#%% load atlas

atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
atlas_img = image.load_img(atlas_file)
labels_fname = r'/d/gmi/1/sebastiancoleman/atlases/glasser52.csv'
labels = pd.read_table(labels_fname, delimiter=',', header=None)
names = labels.iloc[:,0].to_numpy()
coords = labels.iloc[:,1:].to_numpy()

#%% connectivity
        
def calculate_connectivity(subject):
    
    fmin = [1, 4, 8, 13, 30, 70]
    fmax = [4, 8, 13, 30, 50, 100]
    
    # load data
    fnames = sorted(glob(op.join(deriv_path, subject, '*_glasser_orth-raw.fif')))
    
    # calculate conn for each run
    for f, fname in enumerate(fnames):
        
        basename = op.basename(fname)[:-21]
        raw = mne.io.Raw(fname, preload=True, verbose=False)
        
        # change spikes to bad annotation
        annot = raw.annotations
        if len(annot) > 0:
            descriptions = annot.description
            onsets = annot.onset
            durations = annot.duration
            orig_time = annot.orig_time
            for i in range(len(annot)):
                if descriptions[i] == 'spike':
                    descriptions[i] = 'BAD_spike'
                    onsets[i] -= 0.2
                    durations[i] = 0.5
            new_annot = mne.Annotations(onsets, durations, descriptions, orig_time)
            raw.set_annotations(new_annot)
        
        # calculate conn for each freq
        conn_all = []
        for freq in range(len(fmin)):
            raw_filt = raw.copy().filter(fmin[freq], fmax[freq], verbose=False, picks='all').apply_hilbert(envelope=True, picks='all')
            data = raw_filt.get_data(reject_by_annotation='omit', verbose=False)
            conn = np.abs(np.corrcoef(data))
            conn[np.diag_indices(52)] = np.nan
            conn_all.append(conn)
        conn_all = np.dstack(conn_all)
        
        conn_name = op.join(deriv_path, subject, basename + '_aec.npy')
        np.save(conn_name, conn_all)   
        
        
Parallel(n_jobs=10)(delayed(calculate_connectivity)(subject) for subject in subjects)
