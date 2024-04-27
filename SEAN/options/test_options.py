"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    
    def __init__(self,tar_data_root,style_data_root):
        super().__init__()
        self.tar_data_root = tar_data_root
        self.style_data_root = style_data_root
    
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--loop_cnt', type=int)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(tar_data_root = self.tar_data_root)
        parser.set_defaults(style_data_root = self.style_data_root)

        parser.add_argument('--status', type=str, default='test')
        
        parser.add_argument('--out_dir', type=str, default='out', help='output directory')
        parser.add_argument('--dataset', type=str, default='Structured3D', help='output directory')
        self.isTrain = False
        return parser
