import sys
sys.path.append('..')

#
# example for working directly with the code in this repo
#
from ml3d.tf.models import mynet
from ml3d.tf.datasets import ScanNet

model = mynet(num_classes=20, radius=0.05)
dataset = ScanNet('path/to/ScanNet')

out = model(dataset[0]['feats'], dataset[0]['points'])


#
# example with the packaged code
#
import open3d.ml.tf as ml3d # or import open3d.ml.torch as ml3d

model = ml3d.models.mynet(num_classes=20, radius=0.05) 
dataset = ml3d.datasets.ScanNet('/path/to/scannet')

out = model(dataset[0]['feats'], dataset[0]['points'])
