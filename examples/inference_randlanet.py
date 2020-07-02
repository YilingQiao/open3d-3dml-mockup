# import open3d as o3d
import torch
import open3d as o3d
import numpy as np

from ml3d.datasets.semantickitti import ConfigSemanticKITTI, SemanticKITTI
from ml3d.torch.models import randlanet
from tf2torch import load_tf_weights

pointcloud_file = '../o3d-3dml/datasets/fragment.ply'

def main():

    cfg     = ConfigSemanticKITTI
    cfg.dataset_path = '/home/yiling/d2T/intel2020/datasets/semanticKITTI/data_odometry_velodyne/dataset/sequences_0.06'
   
    # gpt2_checkpoint_path = '/home/yiling/d2T/intel2020/RandLA-Net/models/SemanticKITTI/snap-277357'
    model   = randlanet(cfg)
    # load_tf_weights(model, gpt2_checkpoint_path)
    # print("load finish")

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device  = torch.device('cpu')

    pcd = o3d.io.read_point_cloud(pointcloud_file)
    points = np.asarray(pcd.points).astype(np.float32)
    #print(points)
    result = model.run_inference(points, device)
   
    # predictions = torch.max(result, dim=-2).indices

    # o3d.visualization.draw_geometries([pcd])
    # print(predictions.size())
    #o3d.visualization.draw()
                
if __name__ == '__main__':
    main()
