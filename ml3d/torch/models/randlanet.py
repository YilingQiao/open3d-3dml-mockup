#coding: future_fstrings
import torch
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


class randlanet(nn.Module):
    def __init__(self, cfg):
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
        
        #self.d_out = self.config.d_out
        super(randlanet,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = cfg

        self.fc0   = nn.Linear(self.config.d_in, cfg.d_feature)
        self.batch_normalization = nn.BatchNorm2d(cfg.d_feature, eps=1e-6, momentum=0.99)
        # self.fc0_expand_dims = torch.unsqueeze        TODO
        #feature = tf.expand_dims(feature, axis=2)

        f_encoder_list = []
        d_encoder_list = []
        d_feature      = cfg.d_feature

        # ###########################Encoder############################
        for i in range(self.config.num_layers):
            name = 'Encoder_layer_' + str(i)
            self.init_dilated_res_block(d_feature, self.config.d_out[i], name)
            d_feature = self.config.d_out[i] * 2
            if i == 0:
                d_encoder_list.append(d_feature)

            d_encoder_list.append(d_feature)
        # ###########################Encoder############################

        feature = helper_torch_util.conv2d(True, d_feature, d_feature)
        setattr(self, 'decoder_0', feature)


        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            name = 'Decoder_layer_' + str(j)
            d_in  = d_encoder_list[-j-2] + d_feature
            d_out = d_encoder_list[-j-2] 

            f_decoder_i = helper_torch_util.conv2d_transpose(True, d_in, d_out)
            setattr(self, name, f_decoder_i)

            d_feature = d_encoder_list[-j-2] 
           
        # ###########################Decoder############################



        f_layer_fc1 = helper_torch_util.conv2d(True, d_feature, 64)
        setattr(self, 'fc1', f_layer_fc1)
        f_layer_fc2 = helper_torch_util.conv2d(True, 64, 32)
        setattr(self, 'fc2', f_layer_fc2)
        f_layer_fc3 = helper_torch_util.conv2d(False, 32, self.config.num_classes, activation=False)
        setattr(self, 'fc', f_layer_fc3)

        #self = self.to( torch.device('cuda:0'))



    def preprocess_inference(self, pc, device):
        cfg             = self.config
        batch_pc        = torch.from_numpy(pc).unsqueeze(0).to(device)
        
        features        = batch_pc
        input_points    = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(batch_pc, batch_pc, cfg.k_n)
            
            sub_points = batch_pc[:, :batch_pc.size(1) // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.size(1) // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        inputs = dict()
        #print(features)
        inputs['xyz']           = [arr.to(device) 
                                    for arr in input_points]
        inputs['neigh_idx']     = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_neighbors]
        inputs['sub_idx']       = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_pools]
        inputs['interp_idx']    = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_up_samples]
        inputs['features']      = features.to(device)

        return inputs

    def run_inference(self, points, device):
        cfg = self.config
        grid_size   = cfg.grid_size

        input_inference = self.preprocess_inference(points, device)
        self.eval()
        scores = self(input_inference)

        pred = torch.max(scores, dim=-2).indices
        pred   = pred.cpu().data.numpy()
        return pred


    def run_test(self, dataset, device):
        #self.device = device
        cfg = self.config
        self.to(device)
        self.Log_file = open('log_test_' + dataset.name + '.txt', 'a')


        test_sampler = dataset.get_ActiveLearningSampler('test')
        test_loader = DataLoader(test_sampler, batch_size=cfg.val_batch_size)

        self.test_probs = [np.zeros(shape=[len(l), self.config.num_classes], dtype=np.float16)
                           for l in dataset.possibility]

        test_path = join('test', 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        save_path = join(test_path, dataset.test_scan_number, 'predictions')
        makedirs(save_path) if not exists(save_path) else None

        test_smooth = 0.98
        epoch_ind = 0
        self.idx  = 0
        self.eval()

        while True:
            for batch_data in tqdm(test_loader, desc='test', leave=False):
                
                inputs = dataset.preprocess(batch_data, self.device) 

                result_torch = self(inputs)
               
               
                result_torch = torch.reshape(result_torch, (-1, self.config.num_classes))
                m_softmax    = torch.nn.Softmax(dim=-1)
                result_torch = m_softmax(result_torch)
                result_torch = result_torch.cpu().data.numpy()
             
                stacked_probs = result_torch

                if self.idx % 10 == 0:
                    print('step ' + str(self.idx))
                self.idx += 1
                stacked_probs = np.reshape(stacked_probs, [self.config.val_batch_size,
                                                           self.config.num_points,
                                                           self.config.num_classes])
              
                point_inds  = inputs['input_inds']
                cloud_inds  = inputs['cloud_inds']

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_inds[j, :]
                    c_i = cloud_inds[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

            new_min = np.min(dataset.min_possibility)
            print(dataset.min_possibility)
            log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min), self.Log_file)
            if np.min(dataset.min_possibility) > 0.5:  # 0.5
                log_out(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)), self.Log_file)
                print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))

                # For validation set
                num_classes = 19
                gt_classes = [0 for _ in range(num_classes)]
                positive_classes = [0 for _ in range(num_classes)]
                true_positive_classes = [0 for _ in range(num_classes)]
                val_total_correct = 0
                val_total_seen = 0

                for j in range(len(self.test_probs)):
                    test_file_name = dataset.test_list[j]
                    frame = test_file_name.split('/')[-1][:-4]
                    proj_path = join(dataset.dataset_path, dataset.test_scan_number, 'proj')
                    proj_file = join(proj_path, str(frame) + '_proj.pkl')
                    if isfile(proj_file):
                        with open(proj_file, 'rb') as f:
                            proj_inds = pickle.load(f)
                    probs = self.test_probs[j][proj_inds[0], :]
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
                log_out(str(dataset.test_scan_number) + ' finished', self.Log_file)
                if dataset.test_scan_number=='08':
                    iou_list = []
                    for n in range(0, num_classes, 1):
                        iou = true_positive_classes[n] / float(
                            gt_classes[n] + positive_classes[n] - true_positive_classes[n])
                        iou_list.append(iou)
                    mean_iou = sum(iou_list) / float(num_classes)

                    log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
                    log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

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


    def run_train(self, dataset, device):
        #self.device = device
        self.to(device)
        cfg = self.config

        print('Computing weights...', end='\t')
        samples_per_class = np.array(cfg.class_weights)

        n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=device)
        ratio_samples = n_samples / n_samples.sum()
        weights = 1 / (ratio_samples + 0.02)

       
        print('Done.')
        print('Weights:', weights)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.scheduler_gamma)

        first_epoch = 1


        logs_dir = cfg.logs_dir
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


       
        train_sampler = dataset.get_ActiveLearningSampler('training')
        train_loader = DataLoader(train_sampler, batch_size=cfg.val_batch_size)


        with SummaryWriter(logs_dir) as writer:
            for epoch in range(first_epoch, cfg.max_epoch+1):
                print(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
                
                self.train()

                # metrics
                losses = []
                accuracies = []
                ious = []
                step = 0

                for batch_data in tqdm(train_loader, desc='Training', leave=False):

                    labels = batch_data[1] 
                    inputs = dataset.preprocess(batch_data, self.device) 
                    optimizer.zero_grad()

                    scores = self.model(inputs)
                    scores, labels = self.filter_valid(scores, labels, device)

                    logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)


                    loss = criterion(logp, labels)
                    acc  = accuracy(scores, labels)
                    
                    loss.backward()

                    optimizer.step()

                    step = step + 1
                    if (step%50==0):
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

    def init_att_pooling(self, d, d_out, name):

        att_activation = nn.Linear(d, d)
        setattr(self, name + 'fc', att_activation)

        f_agg = helper_torch_util.conv2d(True, d, d_out)
        setattr(self, name + 'mlp', f_agg)
 

    def init_building_block(self, d_in, d_out, name):
       
        f_pc  = helper_torch_util.conv2d(True, 10, d_in)
        setattr(self, name + 'mlp1', f_pc)

        self.init_att_pooling(d_in * 2, d_out // 2, name + 'att_pooling_1')
        
        f_xyz = helper_torch_util.conv2d(True, d_in, d_out//2)
        setattr(self, name + 'mlp2', f_xyz)

        self.init_att_pooling(d_in * 2, d_out, name + 'att_pooling_2')
        
     
    def init_dilated_res_block(self, d_in, d_out, name):
        f_pc = helper_torch_util.conv2d(True, d_in, d_out//2)
        setattr(self, name + 'mlp1', f_pc)

        self.init_building_block(d_out//2, d_out, name + 'LFA')

        f_pc = helper_torch_util.conv2d(True, d_out, d_out * 2, activation=False)
        setattr(self, name + 'mlp2', f_pc)

        shortcut = helper_torch_util.conv2d(True, d_in, d_out * 2, activation=False)
        setattr(self, name + 'shortcut', shortcut)

    def forward_gather_neighbour(self, pc, neighbor_idx):
        # pc:           BxNxd
        # neighbor_idx: BxNxK
        B, N, K = neighbor_idx.size()
        d = pc.size()[2]
      

        extended_idx = neighbor_idx.unsqueeze(1).expand(B, d, N, K)

        extended_coords = pc.transpose(-2,-1).unsqueeze(-1).expand(B, d, N, K)
        
        features = torch.gather(extended_coords, 2, extended_idx)

        return features

    def forward_att_pooling(self, feature_set, name):
       
        # feature_set: BxdxNxK
        batch_size = feature_set.size()[0]
        num_points = feature_set.size()[2]
        num_neigh = feature_set.size()[3]
        d = feature_set.size()[1]

        #feature_set = 
        #f_reshaped = torch.reshape(feature_set, (-1, d, num_neigh))

        m_dense = getattr(self, name + 'fc')
        att_activation = m_dense(feature_set.permute(0,2,3,1)) # TODO

        m_softmax = nn.Softmax(dim=-2)
        att_scores = m_softmax(att_activation).permute(0,3,1,2)


        f_agg = att_scores * feature_set
        f_agg = torch.sum(f_agg, dim=-1, keepdim=True)
        #f_agg = torch.reshape(f_agg, (batch_size, num_points, 1, d))

        m_conv2d = getattr(self, name + 'mlp')
        f_agg = m_conv2d(f_agg)

        return f_agg


    def forward_relative_pos_encoding(self, xyz, neigh_idx):
        B, N, K = neigh_idx.size()
        neighbor_xyz = self.forward_gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        #xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.size()[-1], 1)

        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(torch.sum(torch.square(relative_xyz), dim=1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=1)
        
        return relative_feature


    def forward_building_block(self, xyz, feature, neigh_idx, name):
        
        f_xyz = self.forward_relative_pos_encoding(xyz, neigh_idx)
        m_conv2d = getattr(self, name + 'mlp1')
        f_xyz = m_conv2d(f_xyz)

        feature = feature.transpose(1, 2)
        f_neighbours = self.forward_gather_neighbour(torch.squeeze(feature, axis=3), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=1)

        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_1')

        m_conv2d = getattr(self, name + 'mlp2')
        f_xyz = m_conv2d(f_xyz)


        f_pc_agg = f_pc_agg.transpose(1, 2)
        f_neighbours = self.forward_gather_neighbour(torch.squeeze(f_pc_agg, axis=3), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=1)
        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_2')

        return f_pc_agg


    def forward_dilated_res_block(self, feature, xyz, neigh_idx, d_out, name):
        m_conv2d = getattr(self, name + 'mlp1')
        f_pc     = m_conv2d(feature)

        f_pc = self.forward_building_block(xyz, f_pc, neigh_idx, name + 'LFA')


        m_conv2d = getattr(self, name + 'mlp2')
        f_pc = m_conv2d(f_pc)


        m_conv2d = getattr(self, name + 'shortcut')
        shortcut = m_conv2d(feature)

        m_leakyrelu = nn.LeakyReLU(0.2)

        result = m_leakyrelu(f_pc + shortcut)
        return result


    def forward(self, inputs):
        xyz         = inputs['xyz']
     
        neigh_idx   = inputs['neigh_idx']
        sub_idx     = inputs['sub_idx']
        interp_idx  = inputs['interp_idx']
        feature     = inputs['features']
      


        m_dense = getattr(self, 'fc0')
        feature = m_dense(feature).transpose(-2,-1).unsqueeze(-1) # TODO


        m_bn    = getattr(self, 'batch_normalization')
        feature = m_bn(feature)



        m_leakyrelu = nn.LeakyReLU(0.2)
        feature     = m_leakyrelu(feature)



        # B d N 1
        # B N 1 d

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            name = 'Encoder_layer_' + str(i)
            f_encoder_i = self.forward_dilated_res_block(feature, xyz[i], neigh_idx[i],
                self.config.d_out[i], name)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################


        m_conv2d = getattr(self, 'decoder_0')
        feature = m_conv2d(f_encoder_list[-1])


        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            name = 'Decoder_layer_' + str(j)

            m_transposeconv2d = getattr(self, name)
            concat_feature = torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1)
            f_decoder_i = m_transposeconv2d(concat_feature)
           
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################
        m_conv2d = getattr(self, 'fc1')
        f_layer_fc1 = m_conv2d(f_decoder_list[-1])

        m_conv2d = getattr(self, 'fc2')
        f_layer_fc2 = m_conv2d(f_layer_fc1)


        m_dropout = nn.Dropout(0.5)
        f_layer_drop = m_dropout(f_layer_fc2)


        test_hidden = f_layer_fc2.permute(0,2,3,1)

        m_conv2d = getattr(self, 'fc')
        f_layer_fc3 = m_conv2d(f_layer_drop)

        f_out = f_layer_fc3.squeeze(3).transpose(1,2)


        return f_out

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
