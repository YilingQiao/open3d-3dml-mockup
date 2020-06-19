
import torch
import torch.nn as nn
import helper_torch_util
import numpy as np
from pprint import pprint


class Network_torch(nn.Module):
    #def __init__(self, dataset, config, d_in, num_classes):
    def __init__(self, cfg, d_in):
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
        super(Network_torch,self).__init__()
        self.config = cfg

        self.fc0   = nn.Linear(d_in, 8)
        self.batch_normalization = nn.BatchNorm2d(8, eps=1e-6, momentum=0.99)
        # self.fc0_expand_dims = torch.unsqueeze        TODO
        #feature = tf.expand_dims(feature, axis=2)

        f_encoder_list = []
        d_encoder_list = []
        d_feature = 8

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
        return result, result

    def forward(self, inputs):

       

        xyz         = [torch.from_numpy(arr).cuda() for arr in inputs['xyz']]
        neigh_idx   = [torch.from_numpy(arr).cuda().to(torch.int64) for arr in inputs['neigh_idx']]
        sub_idx     = [torch.from_numpy(arr).cuda().to(torch.int64) for arr in inputs['sub_idx']]
        interp_idx  = [torch.from_numpy(arr).cuda().to(torch.int64) for arr in inputs['interp_idx']]
        #features    = [torch.from_numpy(arr) for arr in inputs['features']]

        feature     = torch.from_numpy(inputs['features']).cuda()

        

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
            f_encoder_i, tmp = self.forward_dilated_res_block(feature, xyz[i], neigh_idx[i],
                self.config.d_out[i], name)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                test_hidden = f_encoder_i.permute(0,2,3,1)
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################


        m_conv2d = getattr(self, 'decoder_0')
        shortcut = m_conv2d(f_encoder_list[-1])


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

        m_conv2d = getattr(self, 'fc')
        f_layer_fc3 = m_conv2d(f_layer_drop)

        f_out = f_layer_fc3.squeeze(3)

        return f_out, test_hidden

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
