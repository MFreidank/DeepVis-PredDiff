# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
    get_network(netname)
    forward_pass(net, x, blobnames='prob')

"""

# this is to supress some unnecessary output of caffe in the linux console
import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import torch.nn as nn
from torchvision import models


def get_network(netname, num_classes=10):
    if netname == "resnet101":
        net = models.resnet101(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        # XXX: load weights
        return net

    raise NotImplementedError(netname)

def forward_pass(net, x, blobnames=["fc"]):
    '''
    Defines a forward pass (modified for our needs)
    Input:      net         the network (pytorch model)
                x           the input, a batch of images
                blobnames   for which layers we want to return the output,
                            default is output layer ('fc')
    '''
    if blobnames != ["fc"]:
        raise NotImplementedError(blobnames)

    if isinstance(net, models.resnet.ResNet):
        # TODO: Full Forward pass for resnet network, collecting outputs at the blob layers
        # and return them.

        # collect outputs of the blobs we're interested in
        # returnVals = [np.copy(net.blobs[b].data[:]) for b in blobnames]
        return net(x)

    raise NotImplementedError(net)
