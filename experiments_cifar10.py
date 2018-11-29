# -*- coding: utf-8 -*-
"""
From this script, experiments for cifar10 pictures can be started.
See "configuration" below for the different possible settings.
The results are saved automatically to the folder ./results

@author: Moritz Freidank
"""

# the following is needed to avoid some error that can be thrown when
# using matplotlib.pyplot in a linux shell
import matplotlib
matplotlib.use('Agg')

# standard imports
import numpy as np
import time
import os

# most important script - relevance estimator
from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import utils_visualise as utlV


# ------------------------ CONFIGURATION ------------------------
# -------------------> CHANGE SETTINGS HERE <--------------------

# pick neural network to run experiment for
netname = 'resnet101'

# pick for which layers the explanations should be computet
# (names depend on network, output layer of resnet is usually called 'fc')
blobnames = ['fc']

# pick image indices which are analysed (in alphabetical order as in the ./data folder) [0,1,2,...]
# (if None, all images in './data' will be analysed)
test_indices = None

# window size (i.e., the size of the pixel patch that is marginalised out in each step)
win_size = 10               # k in alg 1 (see paper)

# indicate whether windows should be overlapping or not
overlapping = True

# settings for sampling
sampl_style = 'conditional' # choose: conditional / marginal
num_samples = 10
padding_size = 2            # important for conditional sampling,
                            # l = win_size+2*padding_size in alg 1
                            # (see paper)

# set the batch size - the larger, the faster computation will be
# (if crashes with memory error, reduce the batch size)
batch_size = 128


# ------------------------ SET-UP ------------------------

net = utlC.get_network(netname)

# get the data
X_test, X_test_im, X_filenames = utlD.get_cifar10_data(net=net)

# get the label names of the 10 cifar10 classes
classnames = utlD.get_cifar10_classnames()

if not test_indices:
    test_indices = [i for i in range(X_test.shape[0])]

# make folder for saving the results if it doesn't exist
path_results = './results/'
os.makedirs(path_results, exist_ok=True)

# ------------------------ EXPERIMENTS ------------------------

# target function (mapping input features to output probabilities)
target_func = lambda x: utlC.forward_pass(net, x, blobnames)

# for the given test indices, do the prediction difference analysis
for test_idx in test_indices:

    # get the specific image (preprocessed, can be used as input to the target function)
    x_test = X_test[test_idx]
    # get the image for plotting (not preprocessed)
    x_test_im = X_test_im[test_idx]
    # prediction of the network
    y_pred = np.argmax(utlC.forward_pass(net, x_test, ['prob']))
    y_pred_label = classnames[y_pred]

    # get the path for saving the results
    if sampl_style == 'conditional':
        save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,padding_size,netname)
    elif sampl_style == 'marginal':
        save_path = path_results+'{}_{}_winSize{}_margSampl_numSampl{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,netname)

    if os.path.exists(save_path+'.npz'):
        print 'Results for ', X_filenames[test_idx], ' exist, will move to the next image. '
        continue

    print "doing test...", "file :", X_filenames[test_idx], ", net:", netname, ", win_size:", win_size, ", sampling: ", sampl_style

    start_time = time.time()

    if sampl_style == 'conditional':
        sampler = utlS.cond_sampler(win_size=win_size, padding_size=padding_size, image_dims=net.crop_dims, netname=netname)
    elif sampl_style == 'marginal':
        sampler = utlS.marg_sampler(X_test, net)

    pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
    pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)

    # plot and save the results
    utlV.plot_results(x_test, x_test_im, pred_diff[0], target_func, classnames, test_idx, save_path)
    np.savez(save_path, *pred_diff)
    print "--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60)
