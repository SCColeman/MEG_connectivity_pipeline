#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute forward model.

@author: sebastiancoleman
"""

import mne
import os.path as op
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from nilearn import image, datasets, plotting
import pandas as pd
from joblib import Parallel, delayed

#%% paths

subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_VNS/subjects_dir'
data_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/derivatives2'

#%% load atlas , glasser 52

atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
atlas_img = image.load_img(atlas_file)
labels_fname = r'/d/gmi/1/sebastiancoleman/atlases/glasser52.csv'
labels = pd.read_table(labels_fname, delimiter=',', header=None)
names = labels.iloc[:,0].to_numpy()
coords = labels.iloc[:,1:].to_numpy()

#%% load data

dir_list = sorted(glob(op.join(deriv_path, '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]')))
subjects = [op.basename(d) for d in dir_list]

def forward_subject(subject):
    
    fnames = sorted(glob(op.join(deriv_path, subject, '*_preproc-raw.fif')))
    
    # single-shell conduction model
    conductivity = (0.3,)
    model = mne.make_bem_model(
            subject=subject, ico=4,
            conductivity=conductivity,
            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    # get mri-->MNI transform and apply inverse to atlas
    mri_mni_t = mne.read_talxfm(subject, subjects_dir=subjects_dir)['trans']
    
    # invert such that MNI-->mri
    mni_mri_t = np.linalg.inv(mri_mni_t)
    
    # apply MNI-->mri transform
    centroids_mri = mne.transforms.apply_trans(mni_mri_t, coords / 1000) # in m
    
    # create glasser source space
    rr = centroids_mri # positions
    nn = np.zeros((rr.shape[0], 3)) # normals
    nn[:,-1] = 1.
    
    # custom source space
    src = mne.setup_volume_source_space(
        subject,
        pos={'rr': rr, 'nn': nn},
        subjects_dir=subjects_dir,
        verbose=True,
    )
    
    # run forward model for each individual raw/trans
    for f, fname in enumerate(fnames):

        raw = mne.io.Raw(fname, preload=False)
        raw.pick('mag')
        basename = op.basename(fname)[:-16]
        trans_fname = op.join(deriv_path, subject, basename + '-trans.fif')
        
        # forward solution
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans_fname,
            src=src,
            bem=bem,
            meg=True,
            eeg=False
            )
        fwd_fname = basename +'_glasser-fwd.fif'
        mne.write_forward_solution(op.join(deriv_path, subject, fwd_fname), fwd, overwrite=True)
        
Parallel(n_jobs=10)(delayed(forward_subject)(subject) for subject in subjects)