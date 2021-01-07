from filament_model import *
from data import *
import pydotplus
import time

start = time.clock()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_gen_args = dict(rotation_range=0.2,  # 0.2
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(1, 'data/filament/train', 'image', 'label', data_gen_args,save_to_dir=None)
# model = filament_unet2()
model = filament_unet()
tensorboard = TensorBoard(log_dir='./log')
# model = vnet()
model.summary()
# model = unet_resnet_101()
model_checkpoint = ModelCheckpoint('unetor.hdf5', monitor='loss', verbose=1, save_best_only=True,save_weights_only=True)
model_EarlyStopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

history = model.fit_generator(myGene, steps_per_epoch=300, epochs=200, shuffle=True,callbacks=[model_checkpoint, model_EarlyStopping])

end = time.clock()
print('Running time: %s Seconds' % (end - start))
# model.summary()
# es = EarlyStopping(monitor='loss',patience=20, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
#
# model_checkpoint = ModelCheckpoint('unetor.hdf5',monitor='loss',verbose=1,save_best_only=True)
# history = model.fit_generator(
#         myGene,
#         steps_per_epoch=100,
#         epochs=320,
#         callbacks=[model_checkpoint,tensorboard, es,reduce_lr]
#     )