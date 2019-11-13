#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module allows the cloning of a Keras model
"""
import keras as K
from keras import optimizers


def clone_model(model):
    model_copy = K.models.clone_model(model)
    _input = model.input_shape
    model_copy.build(_input)  # input layer
    config = model.optimizer.get_config()
    if model.optimizer.__class__ == optimizers.Nadam:
        # optname = 'nadam'
        opt = optimizers.Nadam(lr=config['lr'], clipnorm=1.0, decay=config['decay'])
    elif model.optimizer.__class__ == optimizers.RMSprop:
        # optname = 'rmsprop'
        opt = optimizers.RMSprop(lr=config['lr'], clipnorm=1.0, decay=config['decay'])
    else:
        # optname = 'adam'
        opt = optimizers.Adam(lr=config['lr'], clipnorm=1.0, decay=config['decay'])

    model_copy.compile(optimizer=opt, loss=model.loss)
    model_copy.set_weights(model.get_weights())
    return model_copy
