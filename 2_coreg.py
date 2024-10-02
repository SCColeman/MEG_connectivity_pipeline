#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coreg functional and anatomical.

@author: sebastiancoleman
"""

import mne
import os.path as op
import os
from glob import glob
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

#%% paths

subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_VNS/subjects_dir'
data_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_VNS/derivatives2'

#%% load data

dir_list = sorted(glob(op.join(subjects_dir, '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]')))
subjects = [op.basename(d) for d in dir_list]

def coreg_subject(subject):
    
    fnames = sorted(glob(op.join(deriv_path, subject, '*_preproc-raw.fif')))
    
    for f, fname in enumerate(fnames):
        
        # load raw for info
        raw = mne.io.Raw(fname, preload=False, verbose=False)
    
        # coreg using saved fiducial positions on MRI
        coreg = mne.coreg.Coregistration(raw.info, subject, subjects_dir)
        coreg.fit_fiducials()
        
        # save out transform
        trans = coreg.trans
        basename = op.basename(fname)[:-16]
        trans_fname = op.join(deriv_path, subject, basename + '-trans.fif')
        mne.write_trans(trans_fname, trans, overwrite=True)

Parallel(n_jobs=10)(delayed(coreg_subject)(subject) for subject in subjects)