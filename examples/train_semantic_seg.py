import torch
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import ConfigSemanticKITTI, SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import randlanet
from tf2torch import load_tf_weights

# '../open3d-3dml-mockup/models/SemanticKITTI/snap-277357'
#cfg.dataset
#cfg.model
#arg
#constrcut config 
# cfg = config(.., .., ..)
# configuration for pipeline/dataset/network
for i in range(11,22):

	cfg     = ConfigSemanticKITTI
	cfg.dataset_path = '/home/yiling/d2T/intel2020/datasets/semanticKITTI/data_odometry_velodyne/dataset/sequences_0.06'
	cfg.test_scan_number = str(i)
	dataset = SemanticKITTI(cfg)

	# initialize with model and other options
	# semantic seg -> randlanet
	# pipeline: train/test/inference (model, dataset, cfg)

	gpt2_checkpoint_path = '/home/yiling/d2T/intel2020/RandLA-Net/models/SemanticKITTI/snap-277357'
	model   = randlanet(cfg)
	load_tf_weights(model, gpt2_checkpoint_path)
	# print("load finish")

	pipeline = SemanticSegmentation(model, dataset, cfg)

	device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	#device  = torch.device('cpu')

	#model.run_train(dataset, device)
	pipeline.run_test(device)
	#pipeline.run_train(device)

