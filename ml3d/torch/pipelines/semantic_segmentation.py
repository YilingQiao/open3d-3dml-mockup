#coding: future_fstrings
import torch, pickle
import torch.nn as nn
import helper_torch_util 
import numpy as np
from pprint import pprint
import time
from tqdm import tqdm
from sklearn.neighbors import KDTree
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
from os import makedirs
from os.path import exists, join, isfile, dirname, abspath
from ml3d.datasets.semantickitti import DataProcessing

import yaml

BASE_DIR = './'
#BASE_DIR = dirname(abspath(__file__))

data_config = join(BASE_DIR, 'utils', 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

remap_dict_val = DATA["learning_map"]
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious


def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies


class SemanticSegmentation():
    def __init__(self, model, dataset, cfg):
        '''
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None
        '''
        
        self.model      = model
        self.dataset    = dataset
        self.config     = cfg



    def run_inference(self, points, device):
        cfg = self.config
        grid_size   = cfg.grid_size

        input_inference = self.preprocess_inference(points, device)
        self.eval()
        scores = self(input_inference)

        pred = torch.max(scores, dim=-2).indices
        pred   = pred.cpu().data.numpy()
        return pred


    def run_test(self, device):
        #self.device = device
        model   = self.model
        dataset = self.dataset
        cfg     = self.config
        model.to(device)
        Log_file = open('log_test_' + dataset.name + '.txt', 'a')


        test_sampler = dataset.get_ActiveLearningSampler('test')
        test_loader = DataLoader(test_sampler, batch_size=cfg.val_batch_size)

        test_probs = [np.zeros(shape=[len(l), self.config.num_classes], dtype=np.float16)
                           for l in dataset.possibility]

        test_path = join('test', 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        save_path = join(test_path, dataset.test_scan_number, 'predictions')
        makedirs(save_path) if not exists(save_path) else None

        test_smooth = 0.98
        epoch_ind   = 0
        model.eval()

        while True:
            for batch_data in tqdm(test_loader, desc='test', leave=False):
                
                inputs          = dataset.preprocess(batch_data, device) 

                result_torch    = model(inputs)
               
               
                result_torch    = torch.reshape(result_torch,
                                                    (-1, cfg.num_classes))

                m_softmax       = nn.Softmax(dim=-1)
                result_torch    = m_softmax(result_torch)
                result_torch    = result_torch.cpu().data.numpy()
                stacked_probs   = result_torch

                stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size,
                                                           cfg.num_points,
                                                           cfg.num_classes])
              
                point_inds  = inputs['input_inds']
                cloud_inds  = inputs['cloud_inds']

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_inds[j, :]
                    c_i = cloud_inds[j][0]
                    test_probs[c_i][inds] = \
                                test_smooth * test_probs[c_i][inds] + \
                                (1 - test_smooth) * probs

            new_min = np.min(dataset.min_possibility)
            print(dataset.min_possibility)
            log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min), Log_file)
            #if True:  # 0.5
            if np.min(dataset.min_possibility) > 0.5:  # 0.5
            
                log_out(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)), Log_file)
                print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))

                # For validation set
                num_classes = 19
                gt_classes = [0 for _ in range(num_classes)]
                positive_classes = [0 for _ in range(num_classes)]
                true_positive_classes = [0 for _ in range(num_classes)]
                val_total_correct = 0
                val_total_seen = 0

                for j in range(len(test_probs)):
                    test_file_name = dataset.test_list[j]
                    frame = test_file_name.split('/')[-1][:-4]
                    proj_path = join(dataset.dataset_path, dataset.test_scan_number, 'proj')
                    proj_file = join(proj_path, str(frame) + '_proj.pkl')
                    if isfile(proj_file):
                        with open(proj_file, 'rb') as f:
                            proj_inds = pickle.load(f)
                    probs = test_probs[j][proj_inds[0], :]
                    pred = np.argmax(probs, 1)
                    if dataset.test_scan_number == '08':
                        label_path = join(dirname(dataset.dataset_path), 'sequences', dataset.test_scan_number,
                                          'labels')
                        label_file = join(label_path, str(frame) + '.label')
                        labels = DP.load_label_kitti(label_file, remap_lut_val)
                        invalid_idx = np.where(labels == 0)[0]
                        labels_valid = np.delete(labels, invalid_idx)
                        pred_valid = np.delete(pred, invalid_idx)
                        labels_valid = labels_valid - 1
                        correct = np.sum(pred_valid == labels_valid)
                        val_total_correct += correct
                        val_total_seen += len(labels_valid)
                        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, num_classes, 1))
                        gt_classes += np.sum(conf_matrix, axis=1)
                        positive_classes += np.sum(conf_matrix, axis=0)
                        true_positive_classes += np.diagonal(conf_matrix)
                    else:
                        store_path = join(test_path, dataset.test_scan_number, 'predictions',
                                          str(frame) + '.label')
                        pred = pred + 1
                        pred = pred.astype(np.uint32)
                        upper_half = pred >> 16  # get upper half for instances
                        lower_half = pred & 0xFFFF  # get lower half for semantics
                        lower_half = remap_lut[lower_half]  # do the remapping of semantics
                        pred = (upper_half << 16) + lower_half  # reconstruct full label
                        pred = pred.astype(np.uint32)
                        pred.tofile(store_path)
                log_out(str(dataset.test_scan_number) + ' finished', Log_file)
                if dataset.test_scan_number=='08':
                    iou_list = []
                    for n in range(0, num_classes, 1):
                        iou = true_positive_classes[n] / float(
                            gt_classes[n] + positive_classes[n] - true_positive_classes[n])
                        iou_list.append(iou)
                    mean_iou = sum(iou_list) / float(num_classes)

                    log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), Log_file)
                    log_out('mean IOU:{}'.format(mean_iou), Log_file)

                    mean_iou = 100 * mean_iou
                    print('Mean IoU = {:.1f}%'.format(mean_iou))
                    s = '{:5.2f} | '.format(mean_iou)
                    for IoU in iou_list:
                        s += '{:5.2f} '.format(100 * IoU)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')
                
                return
          
            epoch_ind += 1
            continue


    def run_train(self, device):
        #self.device = device
        model   = self.model
        dataset = self.dataset
        cfg     = self.config
        model.to(device)

        n_samples       = torch.tensor(cfg.class_weights, 
                            dtype=torch.float, device=device)
        ratio_samples   = n_samples / n_samples.sum()
        weights         = 1 / (ratio_samples + 0.02)

        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                           cfg.scheduler_gamma)

        first_epoch = 1
        logs_dir    = cfg.logs_dir
        '''
        if args.load:
            path = max(list((args.logs_dir / args.load).glob('*.pth')))
            print(f'Loading {path}...')
            checkpoint = torch.load(path)
            first_epoch = checkpoint['epoch']+1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        '''

       
        train_sampler   = dataset.get_ActiveLearningSampler('training')
        train_loader    = DataLoader(train_sampler, 
                                     batch_size=cfg.val_batch_size)
        
        model.train()

        with SummaryWriter(logs_dir) as writer:
            for epoch in range(first_epoch, cfg.max_epoch+1):
                print(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
                # metrics
                losses      = []
                accuracies  = []
                ious        = []
                step        = 0

                for batch_data in tqdm(train_loader, desc='Training', leave=False):
                    
                    labels = batch_data[1] 
                    
                    inputs = dataset.preprocess(batch_data, device) 

                    optimizer.zero_grad()

                    scores = model(inputs)
                    
                    scores, labels = self.filter_valid(scores, labels, device)

                    logp = torch.distributions.utils.probs_to_logits(scores, 
                                                            is_binary=False)

                    loss = criterion(logp, labels)
                    acc  = accuracy(scores, labels)
                    
                    loss.backward()

                    optimizer.step()

                    step = step + 1
                    if (step % 50==0):
                        print(loss)
                        print(acc[-1])

                    losses.append(loss.cpu().item())
                    #accuracies.append(accuracy(scores, labels))
                    #ious.append(intersection_over_union(scores, labels))


    def filter_valid(self, scores, labels, device):
        valid_scores = scores.reshape(-1, self.config.num_classes)
        valid_labels = labels.reshape(-1).to(device)
                
        ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
        for ign_label in self.config.ignored_label_inds:
            ignored_bool = torch.logical_or(ignored_bool, 
                            torch.eq(valid_labels, ign_label))
           
        valid_idx = torch.where(
            torch.logical_not(ignored_bool))[0].to(device)

        valid_scores = torch.gather(valid_scores, 0, 
            valid_idx.unsqueeze(-1).expand(-1, self.config.num_classes))
        valid_labels = torch.gather(valid_labels, 0, valid_idx)

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, 
                        self.config.num_classes, dtype=torch.int64)
        inserted_value = torch.zeros([1], dtype=torch.int64)
        
        for ign_label in self.config.ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ign_label],
                     inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = torch.gather(reducing_list.to(device), 
                                        0, valid_labels)

        valid_labels = valid_labels.unsqueeze(0)
        valid_scores = valid_scores.unsqueeze(0).transpose(-2,-1)


        return valid_scores, valid_labels

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        
        feature     = feature.squeeze(3)
        num_neigh   = pool_idx.size()[2]
        batch_size  = feature.size()[0]
        d           = feature.size()[1]

        pool_idx = torch.reshape(pool_idx, (batch_size, -1))

        pool_idx = pool_idx.unsqueeze(2).expand(batch_size,-1, d)
       

        feature = feature.transpose(1,2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features, (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        pool_features = pool_features.permute(0,3,1,2)
    
        return pool_features


    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, d, N] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature         = feature.squeeze(3)
        d               = feature.size(1)
        batch_size      = interp_idx.size()[0]
        up_num_points   = interp_idx.size()[1]

        interp_idx      = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx      = interp_idx.unsqueeze(1).expand(batch_size,d, -1)
        
        interpolated_features = torch.gather(feature, 2, interp_idx)
        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features
