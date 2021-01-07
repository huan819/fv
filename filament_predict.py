from filament_model import *
from data import *
import pydotplus

testGene = testGenerator("data/filament/test103")
# model = vnet()
model = filament_unet()
model.load_weights("unet146.hdf5")
results = model.predict_generator(testGene,10,verbose=1)
saveResult("data/filament/test103",results)