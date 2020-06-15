# import open3d as o3d
import torch
import open3d as o3d
import numpy as np

from pathlib import Path

from ml3d import Trainer
from ml3d.util import Config
from ml3d.dataloader.s3dis import data_loaders


config_file     = 'ml3d/config/semantic_segmentation/randlanet_semantickitti.py'
checkpoint_file = 'ml3d/checkpoint/randlanet_semantickitti.pth'
work_dir        = 'runs'
datasets_path   = 'datasets/s3dis'
datasets_path   = Path(datasets_path)

def main():
    cfg             = Config.load_from_file(config_file)
    cfg.work_dir    = work_dir

    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = data_loaders(datasets_path, 'active_learning')

    trainer         = Trainer(cfg, train_loader)
    trainer.train()

if __name__ == '__main__':
    main()
