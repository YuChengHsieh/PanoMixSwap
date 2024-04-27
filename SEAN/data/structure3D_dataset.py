import os.path as osp
from xml.dom import NotFoundErr
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

class structure3Ddataset(Pix2pixDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        #parser.set_defaults(load_size = 512)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(label_nc = 7)#ceiling,wall,floor,forground
        
        
        return parser
    
    def get_paths(self, opt):
        # structured3D
        # target
        if opt.dataset == 'Structured3D':
            label_paths = [osp.join(tar_data_root,'full/semantic.png') for tar_data_root in opt.tar_data_root]
            layout_paths = [osp.join(tar_data_root,'layout.txt') for tar_data_root in opt.tar_data_root]
            
            # style
            image_paths = [osp.join(style_data_root,'full/rgb_rawlight.png') for style_data_root in opt.style_data_root]
            styleLabel_paths = [osp.join(style_data_root,'full/semantic.png') for style_data_root in opt.style_data_root]
            styleLayout_paths = [osp.join(style_data_root,'layout.txt') for style_data_root in opt.style_data_root]
            
        
        # Stanford2D3D
        # target
        elif opt.dataset == 'Stanford2D3D':
            label_paths = []
            for root in opt.tar_data_root:
                split_root = root.split('/')
                split_root[6] = 'semantic'
                split_root[-1] = split_root[-1][:-4]
                split_root[-1] += 'rgb.png'
                label_paths.append('/'.join(split_root))
            layout_paths = opt.tar_data_root
            # import pdb; pdb.set_trace()
            
            # style
            image_paths,styleLabel_paths = [],[]
            for root in opt.style_data_root:
                split_root = root.split('/')
                split_root[6] = 'img'
                split_root[-1] = split_root[-1][:-4]
                split_root[-1] += 'rgb.png'
                image_paths.append('/'.join(split_root))
                split_root[6] = 'semantic'
                split_root[-1] = split_root[-1][:-7]
                split_root[-1] += 'rgb.png'
                styleLabel_paths.append('/'.join(split_root))
            styleLayout_paths = opt.style_data_root
        else:
            raise NotImplementedError(f'unknown dataset {opt.dataset}')
        
        return label_paths,image_paths,layout_paths,[],styleLabel_paths,styleLayout_paths