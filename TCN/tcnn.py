#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module implements Temporal Convolutional Network


    Making the TCN architecture non-causal allows it to take the future into consideration to do its prediction.
    However, it is not anymore suitable for real-time applications.
    To use a non-causal TCN, specify padding='valid' or padding='same' when initializing the TCN layers.


    code based on:
        * https://github.com/philipperemy/keras-tcn
        * https://github.com/locuslab/TCN/
    ref.:
        * BAI, Shaojie; KOLTER, J. Zico; KOLTUN, Vladlen.
          An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
          arXiv preprint arXiv:1803.01271, 2018.
          https://arxiv.org/pdf/1803.01271

        * OORD, Aaron van den et al.
          Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.
          https://arxiv.org/pdf/1609.03499.pdf
"""
# from typing import List
# from typing import Tuple
import logging

import keras.backend as K
import keras.layers
from keras import optimizers
# from keras.engine.base_layer import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Dense, BatchNormalization
from keras.models import Input, Model


LOG = logging.getLogger('TCNN')
LOG.setLevel(logging.DEBUG)


def residual_block(x, dilation_rate,
                   nb_filters, kernel_size, padding,
                   dropout_rate=0,
                   activation='relu',
                   kernel_initializer='he_normal',
                   use_batch_norm=False):
    # type: (Layer, int, int, int, str, str, float, str, bool) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    :param x: The previous layer in the model
    :param dilation_rate: The dilation power of 2 we are using for this residual block
    :param nb_filters: The number of convolutional filters to use in this block
    :param kernel_size: The size of the convolutional kernel
    :param padding: The padding used in the convolutional layers, 'same' or 'causal'.
    :param activation: The final activation used in o = Activation(x + F(x))
    :param dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    :param kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
    :param use_batch_norm: Whether to use batch normalization in the residual layers or not.

    :return A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   padding=padding)(x)
        if use_batch_norm:
            # TODO:
            # should be WeightNorm here, but using BatchNormalization instead
            # check the original code in https://github.com/openai/weightnorm/tree/master
            # but it works with Keras 1.x
            # a ported version to Keras 2.x can be found in
            # https://github.com/krasserm/weightnorm/tree/master/keras_2
            # and it is also downloaded in the current TCN folder
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = keras.layers.add([prev_x, x])
    res_x = Activation(activation)(res_x)
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class TCN:
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='linear',
                 name='tcn',
                 kernel_initializer='he_normal',
                 use_batch_norm=False):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            LOG.info('An interface change occurred after the version 2.1.2.')
            LOG.info('Before: tcn.TCN(x, return_sequences=False, ...)')
            LOG.info('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            LOG.info('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        # 1D FCN.
        x = Conv1D(self.nb_filters, 1, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             activation=self.activation,
                                             dropout_rate=self.dropout_rate,
                                             kernel_initializer=self.kernel_initializer,
                                             use_batch_norm=self.use_batch_norm)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x


def get_opt(opt, lr, decay=0.0):
    """

    Args:
        opt: Optimizer name.
        lr: Learning rate.
        decay: Learning rate decay over each update.

    """
    assert opt in ['adam', 'rmsprop', 'nadam'], '{} is not a valid optimizer'.format(opt)

    if opt == 'adam':
        return optimizers.Adam(lr=lr, clipnorm=1.0, decay=decay)
    elif opt == 'rmsprop':
        return optimizers.RMSprop(lr=lr, clipnorm=1.0, decay=decay)
    elif opt == 'nadam':
        return optimizers.Nadam(lr=lr, clipnorm=1.0, decay=decay)
    else:
        raise Exception('Only Adam, Nadam and RMSProp are available here')


# https://github.com/keras-team/keras/pull/11373
# It's now in Keras@master but still not available with pip.
# TODO remove later.
def accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='linear',  # type:str,
                 opt='adam',
                 lr=0.002,
                 decay=0.0,
                 use_batch_norm=False,
                 ):
    # type: (...) -> keras.Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes (or values) we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        decay: Learning rate decay over each update.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """
    LOG.debug('num_feat={} num_classes={} nb_filters={} kernel_size={}'.format(num_feat, num_classes, nb_filters, kernel_size))
    LOG.debug('nb_stacks={} max_len={} padding={}'.format(nb_stacks, max_len, padding))
    LOG.debug('use_skip_connections={} return_sequences={} regression={}'.format(use_skip_connections, return_sequences, regression))

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))
    LOG.debug('input_layer.shape={}'.format(input_layer.shape))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, name, kernel_initializer, use_batch_norm)(input_layer)
    LOG.debug('x.shape={}'.format(x.shape))

    # obtain the optimizer object from Keras
    optimizer = get_opt(opt, lr, decay)

    # create regression or classification
    if regression:
        # regression
        x = Dense(num_classes)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(optimizer, loss='mean_squared_error')
    else:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=[accuracy])

    LOG.debug('model.x = {}'.format(input_layer.shape))
    LOG.debug('model.y = {}'.format(output_layer.shape))
    model.summary(print_fn=LOG.info)
    LOG.debug('model.loss {}'.format(model.loss))
    LOG.debug('opt.config {}'.format(model.optimizer.get_config()))

    return model
