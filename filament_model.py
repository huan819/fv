import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as keras
from keras.utils import plot_model
from loss import *

# def filament_unet(pretrained_weights = None,input_size = (512,512,1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     drop1 = Dropout(0.5)(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
#
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     drop2 = Dropout(0.5)(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
#
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     drop3 = Dropout(0.5)(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
#
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     # merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)#drop4
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)#conv3
#     merge7 = concatenate([drop3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     #merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)#conv2
#     merge8 = concatenate([drop2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     #merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)#conv1
#     merge9 = concatenate([drop1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
#
#     model = Model(input = inputs, output = conv10)
#
#     model.compile(optimizer = Adam(lr = 1e-4,beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])#,beta_1=0.9, beta_2=0.999,
#     # model.compile(optimizer=Nadam(lr=2e-4), loss=dices,
#     #               metrics=['accuracy', precision, fbeta_score, fmeasure, IoU, recall])
#     #model.summary()
#
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)
#
#     return model
# def filament_unet2(pretrained_weights = None,input_size = (512,512,1)):
#     inputs = Input(input_size)
#     conv1_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     conv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1_3 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     merge1 = concatenate([conv1_1, conv1_2,conv1_3], axis=3)
#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
#     drop1 = Dropout(0.5)(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
#
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     drop2 = Dropout(0.5)(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
#
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     drop3 = Dropout(0.5)(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
#
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     # merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)#drop4
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)#conv3
#     merge7 = concatenate([drop3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     #merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)#conv2
#     merge8 = concatenate([drop2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     #merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)#conv1
#     merge9 = concatenate([drop1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
#
#     model = Model(input = inputs, output = conv10)
#
#     model.compile(optimizer = Adam(lr = 1e-4,beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])#,beta_1=0.9, beta_2=0.999,
#     # model.compile(optimizer=Nadam(lr=2e-4), loss=dices,
#     #               metrics=['accuracy', precision, fbeta_score, fmeasure, IoU, recall])
#     #model.summary()
#
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)
#
#     return model


def vnet_ag(pretrained_weights=None, input_size=(512, 512, 1), num_class=1, is_training=True, stage_num=6,
         thresh=0.5):  # 二分类时num_classes设置成1，不是2，stage_num可自行改变，也即可自行改变网络深度
    keep_prob = 1.0 if is_training else 1.0  # 不使用dropout
    features = []
    # input_model = Input(input_size)
    # p_re_lu_1 = PReLU()(BatchNormalization()(
    #     Conv2D(16, 5, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    # inputs = p_re_lu_1

    input_model = Input(input_size)
    p_re_lu_1_1 = PReLU()(BatchNormalization()(
        Conv2D(16, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    p_re_lu_1_2 = PReLU()(BatchNormalization()(
        Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_1_1)))
    p_re_lu_1_3 = PReLU()(BatchNormalization()(
        Conv2D(16, 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_1_1)))
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_model)
    p_re_lu_1_1_1 = PReLU()(BatchNormalization()(
        Conv2D(16, 1, activation=None, padding='same', kernel_initializer='he_normal')(pool1)))
    concatenate_1 = concatenate([p_re_lu_1_1, p_re_lu_1_2, p_re_lu_1_3,p_re_lu_1_1_1], axis=-1)
    p_re_lu_1_4 = PReLU()(BatchNormalization()(Conv2D(16, 1, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_1)))
    inputs = p_re_lu_1_4

    p_re_lu_2 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (1 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_1)))
    # p_re_lu_3 = PReLU()(add([inputs,conv2d_2]))
    p_re_lu_3 = PReLU()(add([inputs, p_re_lu_2]))
    dropout_1 = Dropout(keep_prob)(p_re_lu_3)

    p_re_lu_4 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** 1), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(
            dropout_1)))
    p_re_lu_5 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_4)))
    p_re_lu_6 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_5)))
    p_re_lu_7 = PReLU()(add([p_re_lu_4, p_re_lu_6]))
    dropout_2 = Dropout(keep_prob)(p_re_lu_7)
    # attention
    ag_avg = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_2)
    ag_max = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_2)
    ag = concatenate([ag_avg, ag_max], axis=-1)
    ag_1 = ReLU()(BatchNormalization()(Conv2D(32, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(ag)))
    # ag_out1 = PReLU()(add([dropout_2, ag_1]))
    # ag_out1 = Dot()(dropout_2, ag_1)
    ag_out1 = PReLU()(Multiply()([dropout_2, ag_1]))


    p_re_lu_8 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** 2), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(
            ag_out1)))  # 检查
    p_re_lu_9 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_8)))
    p_re_lu_10 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_9)))
    p_re_lu_11 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_10)))
    p_re_lu_12 = PReLU()(add([p_re_lu_8, p_re_lu_11]))
    dropout_3 = Dropout(keep_prob)(p_re_lu_12)

    # attention
    ag_avg_2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_3)
    ag_max_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_3)
    ag2 = concatenate([ag_avg_2, ag_max_2], axis=-1)
    ag_2 = ReLU()(
        BatchNormalization()(Conv2D(64, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(ag2)))
    # ag_out1 = PReLU()(add([dropout_2, ag_1]))
    # ag_out1 = Dot()(dropout_2, ag_1)
    ag_out2 = PReLU()(Multiply()([dropout_3, ag_2]))


    p_re_lu_13 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** 3), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(
            ag_out2)))
    p_re_lu_14 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_13)))
    p_re_lu_15 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_14)))
    p_re_lu_16 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_15)))
    p_re_lu_17 = PReLU()(add([p_re_lu_13, p_re_lu_16]))
    dropout_4 = Dropout(keep_prob)(p_re_lu_17)

    p_re_lu_18 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** 4), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(
            dropout_4)))
    p_re_lu_19 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_18)))
    p_re_lu_20 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_19)))
    p_re_lu_21 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_20)))
    p_re_lu_22 = PReLU()(add([p_re_lu_18, p_re_lu_21]))
    dropout_5 = Dropout(keep_prob)(p_re_lu_22)

    p_re_lu_23 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** 5), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(
            dropout_5)))
    p_re_lu_24 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_23)))
    p_re_lu_25 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_24)))
    p_re_lu_26 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_25)))
    con6 = UpSampling2D(size=(32, 32))(p_re_lu_26)
    p_re_lu_27 = PReLU()(add([p_re_lu_23, p_re_lu_26]))

    p_re_lu_28 = PReLU()(BatchNormalization()(
        Conv2DTranspose(16 * (2 ** (6 - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(p_re_lu_27)))

    concatenate_1 = concatenate([p_re_lu_22, p_re_lu_28], axis=-1)
    p_re_lu_29 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_1)))
    p_re_lu_30 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_29)))
    p_re_lu_31 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_30)))

    con7 = UpSampling2D(size=(16, 16))(p_re_lu_28)
    p_re_lu_32 = PReLU()(add([p_re_lu_28, p_re_lu_31]))
    p_re_lu_33 = PReLU()(BatchNormalization()(
        Conv2DTranspose(16 * (2 ** (5 - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(p_re_lu_32)))

    concatenate_2 = concatenate([p_re_lu_17, p_re_lu_33], axis=-1)
    p_re_lu_34 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_2)))
    p_re_lu_35 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_34)))
    p_re_lu_36 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_35)))


    con8 = UpSampling2D(size=(8, 8))(p_re_lu_36)


    p_re_lu_37 = PReLU()(add([p_re_lu_33, p_re_lu_36]))
    p_re_lu_38 = PReLU()(BatchNormalization()(
        Conv2DTranspose(16 * (2 ** (4 - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(p_re_lu_37)))

    concatenate_3 = concatenate([p_re_lu_12, p_re_lu_38], axis=-1)
    p_re_lu_39 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_3)))
    p_re_lu_40 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_39)))
    p_re_lu_41 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_40)))

    con9 = UpSampling2D(size=(4, 4))(p_re_lu_41)

    p_re_lu_42 = PReLU()(add([p_re_lu_38, p_re_lu_41]))
    p_re_lu_43 = PReLU()(BatchNormalization()(
        Conv2DTranspose(16 * (2 ** (3 - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(p_re_lu_42)))
    con10 = UpSampling2D(size=(2, 2))(p_re_lu_43)


    concatenate_4 = concatenate([p_re_lu_7, p_re_lu_43], axis=-1)
    p_re_lu_44 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_4)))
    p_re_lu_45 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_44)))
    add_10 = (add([p_re_lu_43, p_re_lu_45]))
    p_re_lu_46 = PReLU()(add([p_re_lu_43, p_re_lu_45]))
    p_re_lu_47 = PReLU()(BatchNormalization()(
        Conv2DTranspose(16 * (2 ** (2 - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(p_re_lu_46)))

    concatenate_5 = concatenate([p_re_lu_3, p_re_lu_47], axis=-1)
    p_re_lu_48 = PReLU()(BatchNormalization()(
        Conv2D(16 * (2 ** (1 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_5)))
    add_11 = (add([p_re_lu_47, p_re_lu_48]))
    p_re_lu_49 = PReLU()(add([p_re_lu_47, p_re_lu_48]))

    #深度监督
    hypercolumn = concatenate(
        [
            # con6,
            # con7,
            con8,
            con9,
            con10,
            p_re_lu_49
        ]
    )
    conv_out = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(hypercolumn)
    print(conv_out)  # shape=(?, 512, 512, 1), dtype=float32)
    model = Model(inputs=input_model, outputs=conv_out)
    print(model.output_shape)
    model.compile(optimizer=Nadam(lr=2e-4), loss=dices,
                  metrics=['accuracy', precision, fbeta_score, fmeasure, IoU, recall])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
#Vnet
def vnet(pretrained_weights = None,input_size = (512,512,1),num_class=1,is_training=True,stage_num=6,thresh=0.5):#二分类时num_classes设置成1，不是2，stage_num可自行改变，也即可自行改变网络深度
    keep_prob = 1.0 if is_training else 1.0  # 不使用dropout
    features = []
    input_model = Input(input_size)
    p_re_lu_1_1 = PReLU()(BatchNormalization()(Conv2D(16, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    p_re_lu_1_2 = PReLU()(BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    p_re_lu_1_3 = PReLU()(BatchNormalization()(Conv2D(16, 5, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    concatenate_1 = concatenate([p_re_lu_1_1, p_re_lu_1_2, p_re_lu_1_3], axis=-1)
    p_re_lu_1_4 = PReLU()(BatchNormalization()(Conv2D(16, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_model)))
    inputs = p_re_lu_1_4


    p_re_lu_2 = PReLU()(BatchNormalization()(Conv2D(16*(2**(1-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(p_re_lu_1_4)))
    # p_re_lu_3 = PReLU()(add([inputs,conv2d_2]))
    p_re_lu_3 = PReLU()(add([inputs, p_re_lu_2]))
    dropout_1 = Dropout(keep_prob)(p_re_lu_3)


    p_re_lu_4 = PReLU()(BatchNormalization()(Conv2D(16*(2 ** 1), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(dropout_1)))
    p_re_lu_5 = PReLU()(BatchNormalization()(Conv2D(16*(2**(2-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(p_re_lu_4)))
    p_re_lu_6 = PReLU()(BatchNormalization()(Conv2D(16*(2**(2-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(p_re_lu_5)))
    p_re_lu_7 = PReLU()(add([p_re_lu_4, p_re_lu_6]))
    dropout_2 = Dropout(keep_prob)(p_re_lu_7)


    p_re_lu_8 = PReLU()(BatchNormalization()(Conv2D(16*(2**2), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(dropout_2)))#检查
    p_re_lu_9 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_8)))
    p_re_lu_10 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_9)))
    p_re_lu_11 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_10)))
    p_re_lu_12 = PReLU()(add([p_re_lu_8, p_re_lu_11]))
    dropout_3 = Dropout(keep_prob)(p_re_lu_12)


    p_re_lu_13 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** 3), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(dropout_3)))
    p_re_lu_14 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_13)))
    p_re_lu_15 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_14)))
    p_re_lu_16 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_15)))
    p_re_lu_17 = PReLU()(add([p_re_lu_13, p_re_lu_16]))
    dropout_4 = Dropout(keep_prob)(p_re_lu_17)


    p_re_lu_18 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** 4), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(dropout_4)))
    p_re_lu_19 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_18)))
    p_re_lu_20 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_19)))
    p_re_lu_21 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_20)))
    p_re_lu_22 = PReLU()(add([p_re_lu_18, p_re_lu_21]))
    dropout_5 = Dropout(keep_prob)(p_re_lu_22)


    p_re_lu_23 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** 5), 2, strides=(2, 2), activation=None, padding='same', kernel_initializer='he_normal')(dropout_5)))
    p_re_lu_24 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_23)))
    p_re_lu_25 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_24)))
    p_re_lu_26 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (6 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_25)))
    #深度监督
    con6 = UpSampling2D(size=(32,32))(p_re_lu_26)
    con6_1 = PReLU()(BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(con6)))
    p_re_lu_27 = PReLU()(add([p_re_lu_23, p_re_lu_26]))


    p_re_lu_28 = PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(6-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(p_re_lu_27)))


    concatenate_1 = concatenate([p_re_lu_22, p_re_lu_28], axis=-1)
    p_re_lu_29 = PReLU()(BatchNormalization()(Conv2D(16*(2**(5-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concatenate_1)))
    p_re_lu_30 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_29)))
    p_re_lu_31 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (5 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_30)))
    # 深度监督
    con7 = UpSampling2D(size=(16, 16))(p_re_lu_28)
    con7_1 = PReLU()(BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(con7)))
    p_re_lu_32 = PReLU()(add([p_re_lu_28, p_re_lu_31]))
    p_re_lu_33 = PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (5 - 2)), 2, strides=(2, 2), padding='valid', activation=None,kernel_initializer='he_normal')(p_re_lu_32)))


    concatenate_2 = concatenate([p_re_lu_17, p_re_lu_33], axis=-1)
    p_re_lu_34 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_2)))
    p_re_lu_35 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_34)))
    p_re_lu_36 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (4 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_35)))
    # 深度监督
    con8 = UpSampling2D(size=(8, 8))(p_re_lu_36)
    con8_1 = PReLU()(BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(con8)))
    p_re_lu_37 = PReLU()(add([p_re_lu_33, p_re_lu_36]))
    p_re_lu_38 = PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (4 - 2)), 2, strides=(2, 2), padding='valid', activation=None,kernel_initializer='he_normal')(p_re_lu_37)))


    concatenate_3 = concatenate([p_re_lu_12, p_re_lu_38], axis=-1)
    p_re_lu_39 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_3)))
    p_re_lu_40 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_39)))
    p_re_lu_41 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (3 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_40)))
    # 深度监督
    con9 = UpSampling2D(size=(4, 4))(p_re_lu_41)
    con9_1 = PReLU()(BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(con9)))
    p_re_lu_42 = PReLU()(add([p_re_lu_38, p_re_lu_41]))
    p_re_lu_43 = PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (3 - 2)), 2, strides=(2, 2), padding='valid', activation=None,kernel_initializer='he_normal')(p_re_lu_42)))
    # 深度监督
    con10 = UpSampling2D(size=(2, 2))(p_re_lu_43)
    con10_1 = PReLU()(
        BatchNormalization()(Conv2D(16, 3, activation=None, padding='same', kernel_initializer='he_normal')(con10)))
    concatenate_4 = concatenate([p_re_lu_7, p_re_lu_43], axis=-1)
    p_re_lu_44 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_4)))
    p_re_lu_45 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (2 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(p_re_lu_44)))
    add_10 = (add([p_re_lu_43, p_re_lu_45]))
    p_re_lu_46 = PReLU()(add([p_re_lu_43, p_re_lu_45]))
    p_re_lu_47 = PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (2 - 2)), 2, strides=(2, 2), padding='valid', activation=None,kernel_initializer='he_normal')(p_re_lu_46)))


    concatenate_5 = concatenate([p_re_lu_3, p_re_lu_47], axis=-1)
    p_re_lu_48 = PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (1 - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(concatenate_5)))
    add_11 = (add([p_re_lu_47, p_re_lu_48]))
    p_re_lu_49 = PReLU()(add([p_re_lu_47, p_re_lu_48]))

    hypercolumn = add(
        [
            con6_1,
            con7_1,
            con8_1,
            con9_1,
            con10_1,
            p_re_lu_49
        ]
    )
    print(con6)
    print(con7)
    print(con8)
    print(con9)
    print(con10)
    print(p_re_lu_49)
    conv_out = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(hypercolumn)
    # print(conv_out) #shape=(?, 512, 512, 1), dtype=float32)
    model = Model(inputs=input_model, outputs=conv_out)
    # print(model.output_shape)
    model.compile(optimizer=Nadam(lr=2e-4), loss=dices,metrics=['accuracy', precision, fbeta_score, fmeasure, IoU, recall])
    # model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy', metrics=['accuracy'])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model






# def resBlock(conv,stage,keep_prob,stage_num=5):#收缩路径
#
#     inputs=conv
#     for _ in range(3 if stage>3 else stage):
#         conv=PReLU()((Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
#         #print('conv_down_stage_%d:' %stage,conv.get_shape().as_list())#输出收缩路径中每个stage内的卷积
#     conv_add=PReLU()(add([inputs,conv]))
#     #print('conv_add:',conv_add.get_shape().as_list())
#     conv_drop=Dropout(keep_prob)(conv_add)
#
#     if stage<stage_num:
#         conv_downsample=PReLU()((Conv2D(16*(2**stage), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop)))
#         return conv_downsample,conv_add#返回每个stage下采样后的结果,以及在相加之前的结果
#     else:
#         return conv_add,conv_add#返回相加之后的结果，为了和上面输出保持一致，所以重复输出
#
# def up_resBlock(forward_conv,input_conv,stage):#扩展路径
#
#     conv=concatenate([forward_conv,input_conv],axis = -1)
#     print('conv_concatenate:',conv.get_shape().as_list())
#     for _ in range(3 if stage>3 else stage):
#         conv=PReLU()((Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
#         print('conv_up_stage_%d:' %stage,conv.get_shape().as_list())#输出扩展路径中每个stage内的卷积
#     conv_add=PReLU()(add([input_conv,conv]))
#     if stage>1:
#         conv_upsample=PReLU()((Conv2DTranspose(16*(2**(stage-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(conv_add)))
#         return conv_upsample
#     else:
#         return conv_add
# def vnet(pretrained_weights = None,input_size = (512,512,1),num_class=1,is_training=True,stage_num=5,thresh=0.5):#二分类时num_classes设置成1，不是2，stage_num可自行改变，也即可自行改变网络深度
#     keep_prob = 1.0 if is_training else 1.0#不使用dropout
#     features=[]
#     input_model = Input(input_size)
#     x=PReLU()((Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input_model)))
#
#     for s in range(1,stage_num+1):
#         x,feature=resBlock(x,s,keep_prob,stage_num)#调用收缩路径
#         features.append(feature)
#
#     conv_up=PReLU()((Conv2DTranspose(16*(2**(s-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(x)))
#
#     for d in range(stage_num-1,0,-1):
#         conv_up=up_resBlock(features[d-1],conv_up,d)#调用扩展路径
#     if num_class>1:
#         conv_out=Conv2D(num_class, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
#     else:
#         conv_out=Conv2D(num_class, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
#
#
#     model=Model(inputs=input_model,outputs=conv_out)
#     print(model.output_shape)
#
#     # model_dice=dice_loss(smooth=1e-5)
#     # model_dice=generalized_dice_loss_fun(smooth=1e-5)
#
#     model.compile(optimizer = Nadam(lr = 2e-4), loss = "binary_crossentropy", metrics = ['accuracy'])
#     #model.compile(optimizer = Nadam(lr = 2e-4), loss = dices, metrics=['accuracy',precision,fbeta_score,fmeasure,IoU,recall])
#     #model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy', metrics=['accuracy'])
#     #不使用metric
#     #model.compile(optimizer=Nadam(lr=2e-4), loss=dices,metrics=['accuracy', precision, fbeta_score, fmeasure, IoU, recall])
#     # model.compile(optimizer = Nadam(lr = 2e-4), loss = model_dice)
#     #plot_model(model, to_file='model.png')
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)
#     return model



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x



def unet_resnet_101(height=512, width=512, channel=1, classes=1):
    input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)

    # conv2_x  1/4
    conv2_1 = bottleneck_Block(conv1_2, 256, strides=(1, 1), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024)
    conv4_3 = bottleneck_Block(conv4_2, 1024)
    conv4_4 = bottleneck_Block(conv4_3, 1024)
    conv4_5 = bottleneck_Block(conv4_4, 1024)
    conv4_6 = bottleneck_Block(conv4_5, 1024)
    conv4_7 = bottleneck_Block(conv4_6, 1024)
    conv4_8 = bottleneck_Block(conv4_7, 1024)
    conv4_9 = bottleneck_Block(conv4_8, 1024)
    conv4_10 = bottleneck_Block(conv4_9, 1024)
    conv4_11 = bottleneck_Block(conv4_10, 1024)
    conv4_12 = bottleneck_Block(conv4_11, 1024)
    conv4_13 = bottleneck_Block(conv4_12, 1024)
    conv4_14 = bottleneck_Block(conv4_13, 1024)
    conv4_15 = bottleneck_Block(conv4_14, 1024)
    conv4_16 = bottleneck_Block(conv4_15, 1024)
    conv4_17 = bottleneck_Block(conv4_16, 1024)
    conv4_18 = bottleneck_Block(conv4_17, 1024)
    conv4_19 = bottleneck_Block(conv4_18, 1024)
    conv4_20 = bottleneck_Block(conv4_19, 1024)
    conv4_21 = bottleneck_Block(conv4_20, 1024)
    conv4_22 = bottleneck_Block(conv4_21, 1024)
    conv4_23 = bottleneck_Block(conv4_22, 1024)

    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 2048, strides=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048)
    conv5_3 = bottleneck_Block(conv5_2, 2048)

    up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv5_3), 1024, 2)
    merge6 = concatenate([conv4_23, up6], axis=3)
    conv6 = Conv2d_BN(merge6, 1024, 3)
    conv6 = Conv2d_BN(conv6, 1024, 3)

    up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 512, 2)
    merge7 = concatenate([conv3_4, up7], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([conv2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([conv1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 64, 2)
    conv10 = Conv2d_BN(up10, 64, 3)
    conv10 = Conv2d_BN(conv10, 64, 3)

    conv11 = Conv2d_BN(conv10, classes, 1, use_activation=None)
    activation = Activation('sigmoid', name='Classification')(conv11)
    # conv_out=Conv2D(classes, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv11)

    model = Model(inputs=input, outputs=activation)

    # print(model.output_shape) compounded_loss
    # model_dice=dice_p_bce
    # model_dice=compounded_loss(smooth=0.0005,gamma=2., alpha=0.25)
    # model_dice=dice_loss(smooth=1e-5)
    # model_dice=tversky_coef_loss_fun(alpha=0.3,beta=0.7)
    # model_dice=dice_coef_loss_fun(smooth=1e-5)
    # model.compile(optimizer = Nadam(lr = 2e-4), loss = model_dice, metrics = ['accuracy'])
    #不使用metric
    # model_dice=focal_loss(alpha=.25, gamma=2)
    adam = Adam(lr = 1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    # nadam = Nadam(lr=0.000002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)
    model.compile(optimizer = Nadam(lr = 2e-4), loss=dice,  metrics=['accuracy', precision,fbeta_score,fmeasure,IoU,recall])
    # model.compile(optimizer = Nadam(lr = 5e-4), loss = Focal_Loss,metrics=['accuracy'])
    # model.compile(optimizer = Nadam(lr = 5e-4), loss = dice,metrics=['accuracy', dice_coef,precision,fbeta_score,fmeasure,IoU,recall])
    # model.compile(optimizer = Nadam(lr = 5e-4), loss = Focal_Loss,metrics=['accuracy'])

    # model.compile(optimizer = Nadam(lr = 2e-4), loss = "categorical_crossentropy",metrics=['accuracy'])
    #plot_model(model, to_file='model.png')
    return model