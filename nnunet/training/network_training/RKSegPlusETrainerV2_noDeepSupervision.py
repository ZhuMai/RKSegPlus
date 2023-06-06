#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.network_architecture.generic_RKSegPlusE import Generic_RKSegPlusE, ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import nnUNetTrainerV2_noDeepSupervision
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn


class RKSegPlusETrainerV2_noDeepSupervision(nnUNetTrainerV2_noDeepSupervision):

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = False
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_RKSegPlusE(self.num_input_channels, self.base_num_features, self.num_classes,
                                    num_pool=len(self.net_num_pool_op_kernel_sizes),
                                    num_conv_per_stage=self.conv_per_stage, feat_map_mul_on_downscale=1,
                                    conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                                    dropout_op_kwargs=dropout_op_kwargs,
                                    nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=False,
                                    dropout_in_localization=False, final_nonlin=lambda x: x, weightInitializer=InitWeights_He(1e-2),
                                    pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes, conv_kernel_sizes=self.net_conv_kernel_sizes,
                                    upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                                    max_num_features=None, basic_block=ConvDropoutNormNonlin,
                                    seg_output_use_bias=False)
        self.print_to_log_file('Parameters:', sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class RKSegPlusETrainerV2_noDeepSupervision_150epochs(RKSegPlusETrainerV2_noDeepSupervision):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 150
