"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
from tkinter import N
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torch

from utils.panostretch import pano_connect_points

def colorize_segmap(segmap):
    PALETTE = [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)]
    
    colorized_segmap = np.zeros((segmap.shape[0],segmap.shape[1],3))
    segmap =segmap.astype('uint8')
    segmap[segmap==255] = 0
    values = np.unique(segmap)
    for value in values:
        colorized_segmap[segmap==value] = PALETTE[value]
    
    return colorized_segmap

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        
        label_paths, image_paths, layout_paths, instance_paths, \
            styleLabel_paths, styleLayout_paths = self.get_paths(opt)
        
        #util.natural_sort(label_paths)
        #util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.layout_paths = layout_paths
        
        if styleLabel_paths:
            styleLabel_paths = styleLabel_paths[:opt.max_dataset_size]
            self.styleLabel_paths = styleLabel_paths
        else:
            self.styleLabel_paths = []

        if styleLayout_paths:
            styleLayout_paths = styleLayout_paths[:opt.max_dataset_size]
            self.styleLayout_paths = styleLayout_paths
        else:
            self.styleLayout_paths = []
        
        size = len(self.label_paths)
        self.dataset_size = size
        self.dataset = opt.dataset

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        """ 7label id: [ceiling, floor, other, wall3(spilt in two halves), wall0, wall1, wall2] """
        label_path = self.label_paths[index]
        label = np.asarray(Image.open(label_path)).copy()
        layout = np.loadtxt(self.layout_paths[index]).astype('int')
        """Futniture Fusing Block Figure"""
        # layout[:,0] = layout[:,0] - layout[0][0]
        # label = np.roll(label,-layout[0][0],axis=1)
        wall_key = [4,5,6,3]
        for i in range(4):
            up =  pano_connect_points(layout[0+i*2],layout[(2+i*2)%8],z=-50).astype('int')
            down = pano_connect_points(layout[1+i*2],layout[(3+i*2)%8],z=50).astype('int')

            for col,row_up,row_down in zip(up[:,0],up[:,1],down[:,1]):
                label[:row_up,col] = 0
                label[row_up:row_down,col] = wall_key[i]
                label[row_down:,col] = 1

        label = Image.fromarray(label.astype('uint8'))
        label = label.resize((label.size[0]//2,label.size[1]//2),Image.NEAREST)
        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        
        """4 Label"""
        # layout = np.loadtxt(self.layout_paths[index]).astype('int')
        # old_label = np.asarray(Image.open(self.label_paths[index]).resize((1024,512),Image.NEAREST)).copy()
        # Image.fromarray(colorize_segmap(old_label).astype('uint8'))
        # label = np.zeros((old_label.shape[0],old_label.shape[1]))
        # # 4 labels
        # for i in range(4):
        #     up =  pano_connect_points(layout[0+i*2],layout[(2+i*2)%8],z=-50,w=label.shape[1],h=label.shape[0]).astype('int')
        #     down = pano_connect_points(layout[1+i*2],layout[(3+i*2)%8],z=50,w=label.shape[1],h=label.shape[0]).astype('int')

        #     for col,row_up,row_down in zip(up[:,0],up[:,1],down[:,1]):
        #         label[:row_up,col] = 0
        #         label[row_up:row_down,col] = 1
        #         label[row_down:,col] = 2
        
        # label = Image.fromarray(label.astype('uint8'))
        # label = label.resize((512,256),Image.NEAREST)
        
        # params = get_params(self.opt, label.size)
        # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # label_tensor = transform_label(label) * 255.0
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc


        # Style Image
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((512,256))

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        #if test
        if self.styleLabel_paths:
            # generate style label
            styleLabel_path = self.styleLabel_paths[index]
            styleLabel = Image.open(styleLabel_path)
            styleLabel = np.asarray(styleLabel.resize((512,256),Image.NEAREST)).astype('int').copy()
            style_layout = np.loadtxt(self.layout_paths[index]).astype('int')
            
            "Structured3D 7 labels"
            # 7 labels
            key = np.array([2,3,1,2,2,2,2,2,2,2,
                            2,2,2,2,2,2,2,2,2,2,
                            2,2,0,2,2,2,2,2,2,2,
                            2,2,2,2,2,2,2,2,2,2,2])
            
            values = np.unique(styleLabel)
            new_styleLabel = np.zeros_like(styleLabel)
            for value in values:
                new_styleLabel[styleLabel==value] = key[value]

            # only change 3 walls --> not need to rotate
            for i in range(0,6,2):
                tmp_left = style_layout[i][0]//2
                tmp_right = style_layout[i+2][0]//2
                new_styleLabel[:,tmp_left:tmp_right][new_styleLabel[:,tmp_left:tmp_right]==3] = 4+i//2
            new_styleLabel = Image.fromarray(new_styleLabel.astype('uint8'))
            
            """4 labels"""
            # "Structured3D 4 labels key"
            # if self.dataset == 'Structured3D':
            #     key = np.array([3,1,2,3,3,3,3,3,3,3,
            #                     3,3,3,3,3,3,3,3,3,3,
            #                     3,3,0,3,3,3,3,3,3,3,
            #                     3,3,3,3,3,3,3,3,3,3,3])
            
            #     "Stanford 4 labels key"
            # elif self.dataset == 'Stanford2D3D':
            #     image = np.asarray(image)
            #     # move out upper and lower black region
            #     for i in range(styleLabel.shape[0]):
            #         for j in range(styleLabel.shape[1]):
            #             if np.array_equal(image[i][j],np.array([0,0,0])):
            #                 styleLabel[i,j] = 0
            #     if 14 in np.unique(styleLabel):
            #         print(self.styleLabel_paths[index])
            #     styleLabel -= 1
            #     # 4:0,11:1,9:2,0:3,8:4,12:5,3:6,6:7,2:8,10:9,7:10,1:11,5:12
            #     key = np.array([3,3,3,0,3,3,3,3,2,3,3,1,3])
            # else:
            #     raise NotImplementedError(f'unknown dataset {self.dataset}')
            # values = list(np.unique(styleLabel))
            # if -1 in values:
            #     values.remove(-1)
            # new_styleLabel = np.zeros_like(styleLabel)
            # for value in values:
            #     new_styleLabel[styleLabel==value] = key[value]
            # new_styleLabel = Image.fromarray(new_styleLabel.astype('uint8'))
        
            """draw figure"""
            # back_dict = {0:22,1:2,2:40}
            # new_tar_segmap = new_styleLabel.copy()
            # for key in back_dict:
            #     new_tar_segmap[new_styleLabel==key] = back_dict[key]
            # save_new_styleLabel  = new_tar_segmap
            # save_new_styleLabel = Image.fromarray(colorize_segmap(save_new_styleLabel).astype('uint8')).save('new_pipeline/style_semmap.png')
            # import pdb; pdb.set_trace()
            # new_styleLabel = Image.fromarray(new_styleLabel.astype('uint8'))
            params = get_params(self.opt, new_styleLabel.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            styleLabel_tensor = transform_label(new_styleLabel) * 255.0
            styleLabel_tensor[styleLabel_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        
        else:
            styleLabel_tensor = []
        
        
        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'styleLabel' :styleLabel_tensor 
                      }
             
        
        # Give subclasses a chance to modify the final output

        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


    # Our codes get input images and labels
    def get_input_by_names(self, image_path, image, label_img):
        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)


        # input image (real images)]
        # image = Image.open(image_path)
        # image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        image_tensor.unsqueeze_(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
