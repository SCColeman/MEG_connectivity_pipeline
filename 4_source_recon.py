#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct source activity.

@author: sebastiancoleman
"""

import mne
from mne import beamformer
import os.path as op
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from nilearn import image, datasets, plotting
from scipy.stats import zscore
from mne_connectivity import symmetric_orth
import pandas as pd
from joblib import Parallel, delayed

#%% paths

subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_VNS/subjects_dir'
data_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/derivatives2'

#%% load atlas

atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
atlas_img = image.load_img(atlas_file)
labels_fname = r'/d/gmi/1/sebastiancoleman/atlases/glasser52.csv'
labels = pd.read_table(labels_fname, delimiter=',', header=None)
names = labels.iloc[:,0].to_numpy()
coords = labels.iloc[:,1:].to_numpy()

#%% load data

dir_list = sorted(glob(op.join(deriv_path, '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]')))
subjects = [op.split(d)[-1] for d in dir_list]

def beamformer_subject(subject):

    fnames = sorted(glob(op.join(deriv_path, subject, '*preproc-raw.fif')))

    for f, fname in enumerate(fnames):
        
        try:
            basename = op.split(fname)[-1][:-16]
            raw = mne.io.Raw(fname, preload=True, verbose=False)
            
            # calculate data covariance
            cov = mne.compute_raw_covariance(raw, reject_by_annotation=True, verbose=False)
            
            # construct beamformer from fwd and cov
            fwd_fname = op.join(deriv_path, subject, basename + '_glasser-fwd.fif')
            fwd = mne.read_forward_solution(fwd_fname, verbose=False)
            filters = mne.beamformer.make_lcmv(
                    raw.info,
                    fwd,
                    cov,
                    reg=0.05,
                    noise_cov=None,
                    pick_ori='max-power',
                    rank=None,
                    reduce_rank=True,
                    verbose=False,
                    )
            
            # apply beamformer
            stc =  beamformer.apply_lcmv_raw(raw, filters, verbose=False)
            source_data = stc.data
            source_data_orth = zscore(np.squeeze(symmetric_orth(np.expand_dims(source_data,0))), 1)
            
            # make source raw
            info = mne.create_info(list(names), raw.info['sfreq'], 'misc', verbose=False)
            source_raw = mne.io.RawArray(source_data, info, verbose=False)
            source_raw.set_meas_date(raw.info['meas_date'])
            source_raw.set_annotations(raw.annotations, verbose=False)
            source_raw.save(op.join(deriv_path, subject, basename + '_glasser-raw.fif'), overwrite=True, verbose=False)
            
            # orthogonalise
            source_raw_orth =  mne.io.RawArray(source_data_orth, info, verbose=False)
            source_raw_orth.set_meas_date(raw.info['meas_date'])
            source_raw_orth.set_annotations(raw.annotations, verbose=False)
            source_raw_orth.save(op.join(deriv_path, subject, basename + '_glasser_orth-raw.fif'), overwrite=True, verbose=False)
        except:
            print('Error with ' + fname)
 

# parallelise
Parallel(n_jobs=10)(delayed(beamformer_subject)(subject) for subject in subjects)