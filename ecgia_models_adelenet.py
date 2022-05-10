#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ambiente virtual: venv -
# abrir visual code:  code --log trace
"""
Rede proposta pela Adele refatorada -- REVISADA
"""
#from tensorflow import keras
from typing_extensions import Concatenate
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Conv2D, MaxPooling2D, MaxPooling3D, BatchNormalization, Activation, concatenate, Dropout
import tensorflow as tf


class AdeleNet:
    def __init__(self, input_size_3d = (12, 144, 224, 1), input_size_2d = (141, 898, 1), covars = None, n_classes = 1):
        self.input_size_3d = input_size_3d
        self.input_size_2d = input_size_2d
        self.covars = covars
        self.n_classes = n_classes   
        self.model = self.get_adelenet()
        self.model2d = self.get_adelenet2d()
        self.model3d = self.get_adelenet3d()

    def get_net_2d(self):
        self.net_2d = Sequential()
        for i in range(6):
            self.net_2d.add(Conv2D(16, (3, 3), padding='same', input_shape=self.input_size_2d)) if i==0 else self.net_2d.add(Conv2D(16, (3, 3), padding='same', input_shape=self.input_size_2d))
            self.net_2d.add(BatchNormalization())
            self.net_2d.add(Activation("relu"))
            self.net_2d.add(Conv2D(16, (3, 3), padding='same'))
            self.net_2d.add(BatchNormalization())
            self.net_2d.add(Activation("relu"))
            self.net_2d.add(MaxPooling2D(pool_size=(2,3) if i<4 else (2,2)))
        self.net_2d.add(Flatten())
        return self.net_2d

    def get_net_3d(self):
        self.net_3d = Sequential()
        for i in range(6):
            self.net_3d.add(Conv3D(16, (3, 3, 3), padding='same', input_shape=self.input_size_3d)) if i==0 else self.net_3d.add(Conv3D(16, (3, 3, 3), padding='same'))
            self.net_3d.add(BatchNormalization())
            self.net_3d.add(Activation("relu"))
            self.net_3d.add(Conv3D(16, (3, 3, 3), padding='same'))
            self.net_3d.add(BatchNormalization())
            self.net_3d.add(Activation("relu"))
            self.net_3d.add(MaxPooling3D(pool_size=(2,2,2) if i<2 else (3, 2,2) if i<3 else (1,2,2)))
        self.net_3d.add(Flatten())
        return self.net_3d

    def get_adelenet2d(self):
        input_2d = Input(shape = self.input_size_2d)
        input_covar = Input(shape=(len(self.covars),)) if self.covars is not None and len(self.covars) > 0 else None
        self.get_net_2d()
        out_2d = self.net_2d(input_2d)
        if input_covar is not None:
            out_concatenate = concatenate([out_2d, input_covar], axis = 1)
            out_concatenate = Dense(16, activation='relu')(out_concatenate)
            out = Dense(self.n_classes, activation='sigmoid')(out_concatenate)
            model = Model(inputs = [input_2d, input_covar], outputs = out)
        else:
            out = Dense(16, activation='relu')(out_2d)
            out = Dense(self.n_classes, activation='sigmoid')(out)
            model = Model(inputs = input_2d, outputs = out)
        return model

    def get_adelenet3d(self):
        input_3d = Input(shape = self.input_size_3d)
        input_covar = Input(shape=(len(self.covars),)) if self.covars is not None and len(self.covars) > 0 else None
        self.get_net_3d()
        out_3d = self.net_3d(input_3d)
        if input_covar is not None:
            out_concatenate = concatenate([out_3d, input_covar], axis = 1)
            out_concatenate = Dense(16, activation='relu')(out_concatenate)
            out = Dense(self.n_classes, activation='sigmoid')(out_concatenate)
            model = Model(inputs = [input_3d, input_covar], outputs = out)
        else:
            out = Dense(16, activation='relu')(out_3d)
            out = Dense(self.n_classes, activation='sigmoid')(out)
            model = Model(inputs = input_3d, outputs = out)
        return model


    def get_adelenet(self):
        input_2d = Input(shape = self.input_size_2d)
        input_3d = Input(shape = self.input_size_3d)
        input_covar = Input(shape=(len(self.covars),)) if self.covars is not None and len(self.covars) > 0 else None
        self.get_net_2d()
        self.get_net_3d()
        out_2d = self.net_2d(input_2d)
        out_3d = self.net_3d(input_3d)
        if input_covar is not None:
            out_concatenate = concatenate([out_2d, out_3d, input_covar], axis = 1)
            out_concatenate = Dense(16, activation='relu')(out_concatenate)
            out = Dense(self.n_classes, activation='sigmoid')(out_concatenate)
            model = Model(inputs = [input_3d, input_2d, input_covar], outputs = out)
        else:
            out_concatenate = concatenate([out_2d, out_3d], axis = 1)
            out_concatenate = Dense(16, activation='relu')(out_concatenate)
            out = Dense(self.n_classes, activation='sigmoid')(out_concatenate)
            model = Model(inputs = [input_3d, input_2d], outputs = out)
        return model

if __name__ == "__main__":
    a = AdeleNet()
    a.model.summary()