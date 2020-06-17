
import torch
import torch.nn as nn
import helper_torch_util


class Network_torch(nn.Module):
    #def __init__(self, dataset, config, d_in, num_classes):
    def __init__(self, cfg, d_in, num_classes):
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
     

        self.fc0   = nn.Linear(d_in, 8)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.batch_normalization = nn.BatchNorm2d(8, eps=1e-6, momentum=0.99)
        # self.fc0_expand_dims = torch.unsqueeze        TODO
        #feature = tf.expand_dims(feature, axis=2)

        f_encoder_list = []
        self.array_din  = [8, 32, 128, 256]
        self.array_dout = [16, 64, 128, 256]


        for i in range(self.config.num_layers):
            name = 'Encoder_layer_' + str(i)
            f_encoder_i = self.dilated_res_block(self.array_din[i], self.array_dout[i], name)

            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

       
    def dilated_res_block(self, d_in, d_out, num_neighbors, name)

        f_pc = helper_torch_util.conv2d(True, d_in, d_out//2)
        setattr(self, name + 'mlp1', f_pc)


    
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)


    def forward_building_block(self, xyz, feature, neigh_idx, d_out, name):
        d_in = self.now_din
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        m_conv2d = getattr(self, name + 'mlp1')
        f_xyz = m_conv2d(f_xyz)

        f_neighbours = self.gather_neighbour(torch.squeeze(feature, axis=2), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=-1)

        f_pc_agg = self.forward_att_pooling(f_concat, d_out // 2, name + 'att_pooling_1')

        m_conv2d = getattr(self, name + 'mlp2')
        f_neighbours = self.gather_neighbour(torch.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.forward_att_pooling(f_concat, d_out, name + 'att_pooling_2')

        return f_pc_agg


    def forward_dilated_res_block(self, feature, xyz, neigh_idx, d_out, name):
        m_conv2d = getattr(self, name + 'mlp1')
        f_pc     = m_conv2d(feature)

        self.now_din = d_out // 2
        f_pc = self.forward_building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA')

        m_conv2d = getattr(self, name + 'mlp2')
        f_pc = m_conv2d(f_pc)
        m_conv2d = getattr(self, name + 'shortcut')
        shortcut = m_conv2d(f_pc)

        return self.LeakyReLU(f_pc + shortcut)

    def forward(self, inputs):

        feature = inputs
        xyz = inputs

        for i in range(self.config.num_layers):
            name = 'Encoder_layer_' + str(i)
            f_encoder_i = self.forward_dilated_res_block(feature, xyz[i], neigh_idx[i]
                self.array_dout[i],name)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

def relative_pos_encoding():
    pass