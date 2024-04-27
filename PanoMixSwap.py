import argparse
import sys
import os
import os.path as osp

from utils.utils import SourceImageProcessor, TargetImageProcessor


# SEAN module
sys.path.append('./SEAN')
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN import data
from SEAN.util import util

import numpy as np
from PIL import Image
import time
import pickle
import random
from tqdm import tqdm
from utils.utils import colorize_segmap

def change_style(model,data_i):
    
    start_time = time.time()
    generated = model(data_i, mode='inference')
    tar_img = util.tensor2im(generated)
    
    end_time = time.time()
    
    # print(f'Change style cost time: {end_time-start_time:.2f} seconds')
    return tar_img

def initialize_processor(opts:argparse,data_root:str,wall_cnt:int,state:int,\
    dataset:str,tar_img:np.ndarray=None, tar_segmap:np.ndarray=None,correct_shape:tuple=(512,1024,4)):
    """
        initialize target image processor and source image processor
        state 1: target image processor initialize target image
        state 2: source image processor initialize source image
        state 3: source image processor initialize target image -- special case,
        if we don't want to change wall furnitures
    """
    
    assert state in [1,2,3],'unknown state for process initialization'
    processor = TargetImageProcessor(data_root,wall_cnt,dataset,tar_img,tar_segmap,correct_shape=correct_shape) if state == 1 else \
        SourceImageProcessor(data_root,wall_cnt,dataset,correct_shape=correct_shape) if state == 2 else\
        SourceImageProcessor(data_root,wall_cnt,dataset,correct_shape=correct_shape)

    return processor

def cal_height(source:SourceImageProcessor,target:TargetImageProcessor,fur_X:list=None):
    # find points on source image's furnitures that exceed wall and will be paste into target image
    # print(col_cnt)

    # interpolate source image to target image based on source height and target height ratio
    interpolated_image_col = source.cal_interpolated_pixels(target)
    interpolated_segmap_col = source.cal_interpolated_pixels(target,semantic=True)

    # extrapolate source image to target image based on source height and target height ratio
    extrapolated_upImage_col, extrapolated_upSegmap_col = source.cal_extrapolated_pixels(target,up=True)
    extrapolated_downImage_col, extrapolated_downSegmap_col  = source.cal_extrapolated_pixels(target,up=False)
    
    # replace target wall with the source wall
    target.paste_src_furniture(interpolated_image_col,interpolated_segmap_col, \
        extrapolated_upImage_col,extrapolated_downImage_col,extrapolated_upSegmap_col,\
        extrapolated_downSegmap_col,furnitures_X=fur_X)
    # import pdb; pdb.set_trace()

def change_furniture(opts:argparse ,tar_data_root:str, fur_data_root:str, dataset:str,\
                     tar_img:np.ndarray=None, tar_segmap:np.ndarray=None,correct_shape:tuple=(512,1024,4)):
    start_time = time.time()
    # assert opts.wall.count(',') == 3 , 'wall length is incorrect'
    # src_wall = [int(i) for i in opts.wall.split(',')]
    src_wall = [0,1,2,3]
    # fur_X = [int(i) for i in opts.fur_X.split(',')] if len(opts.fur_X) > 0 else []
    
    target = initialize_processor(opts,tar_data_root,0,state=1,dataset=dataset,tar_img=tar_img,tar_segmap=tar_segmap,correct_shape=correct_shape)

    src_wall_cnt = -1
    for tar_wall in range(4):

        src_wall_cnt += 1
        target.change_wall_attribute(tar_wall)

        # change or unchange walls
        source = initialize_processor(opts,fur_data_root,src_wall[src_wall_cnt],state=2,dataset=dataset,correct_shape=correct_shape) if src_wall[src_wall_cnt] not in [-1,4] \
            else initialize_processor(opts,tar_data_root,tar_wall,state=3,dataset=dataset,correct_shape=correct_shape)
        
        """width alignment"""
        # perform panostretch transform
        source.panostretch_transform(getattr(target,'wall_width'))
        # np.savetxt('pipeline/panostretch_layout.txt',source.layout)

        # roll image & segmap & layout if wall count = 3
        if src_wall[src_wall_cnt] == 3 or (src_wall[src_wall_cnt] in [-1,4] and tar_wall == 3): 
            source.change_input_attribute()
        if tar_wall == 3:
            target.change_input_attribute()

        assert getattr(source,'wall_width') == getattr(target,'wall_width'), \
                'source width doesn''t match target width'

        """height alignment"""
        cal_height(source,target)
        
        if tar_wall == 3:
            target.change_input_attribute(phase=False)

    end_time = time.time()
    # print(f'Change furniture cost time: {end_time-start_time:.2f} seconds')
    
    return getattr(target,'image'),getattr(target,'segmap'),getattr(target,'layout')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help="Only Structured3D and Stanford2D3D")
    parser.add_argument('--layout_path', type=str, required=True, help="ground truth layout of your dataset")
    parser.add_argument('--image_path',type=str, required=True, help="images of your dataset")
    
    parser.add_argument('--results_root',type=str, default="./results", help="Path that augmented data should put in")
    parser.add_argument('--width', type=int, default=1024, help="images width")
    parser.add_argument('--height', type=int, default=512, help="images height")
    parser.add_argument('--layout_diff', action="store_true", help="if two images layout is too different then we will not prrocess it")
    args = parser.parse_args()
    
    if args.dataset not in ["Structured3D", "Stanford2D3D"]: 
        raise NotImplementedError(f"Unknown dataset {args.dataset}")

    try:
        with open(args.image_path, "rb") as fp:
            image_data_root = pickle.load(fp)
    except:
        raise FileNotFoundError(f"Image path {args.image_path} is wrong")
    
    try:
        with open(args.layout_path, "rb") as fp:
            layout_data_root = pickle.load(fp)
    except:
        raise FileNotFoundError(f"Layout path {args.layout_path} is wrong")
    
    assert len(image_data_root) == len(layout_data_root), "number of images is not equal to number of layouts"
    
    
    # determine the parts (style, layout, furniture) of images use
    length = len(image_data_root)
    style_data_root = random.sample(image_data_root,length)
    fur_data_root = random.sample(image_data_root,length)
    layout_data_root = random.sample(layout_data_root,length)
    
    opts = TestOptions(layout_data_root,style_data_root)
    opts.dataset = args.dataset

    # SEAN related
    model = Pix2PixModel(opts)
    model.eval()
    dataloader = data.create_dataloader(opts)
    
    start_time = time.time()
    for i,data_i in enumerate(tqdm(dataloader)):
        
        # setting for results
        save_path = osp.join(args.results_root,str(i),'panorama')
        if not osp.exists(osp.join(save_path,'full')):
            os.makedirs(osp.join(save_path,'full'))
        if i>100:
            break
        
        # check source and target layout's difference: 
        if(args.layout_diff):
            src_layout = np.loadtxt(osp.join(fur_data_root[i],'layout.txt'))
            tar_layout = np.loadtxt(osp.join(layout_data_root[i],'layout.txt'))
            infeasible = False
            for j in range(0,len(src_layout),2):
                # third wall
                if j == 6:
                    if abs((1024-src_layout[6][0]+src_layout[0][0]) - (1024-tar_layout[6][0]+tar_layout[0][0])) > 150:
                        infeasible = True
                        break
                elif abs((src_layout[j+2][0] - src_layout[j][0]) - (tar_layout[j+2][0] - tar_layout[j][0])) > 150:
                    infeasible = True
                    break
            
            if infeasible:
                continue
        
        
        tar_img = change_style(model,data_i)[0].astype('float') # cv2 resize problem --> change type as float 
        tar_segmap = data_i['label'][0].squeeze().cpu().numpy().astype('uint8') # cv2 resize problem --> change type as float
        draw_dict = {0:22,1:2}
        tar_segmap_draw = tar_segmap.copy()
        for key in draw_dict:
            tar_segmap_draw[tar_segmap==key] = draw_dict[key]
        
        back_dict = {0:22,1:2,3:1,4:1,5:1,6:1}
        new_tar_segmap = tar_segmap.copy()
        for key in back_dict:
            new_tar_segmap[tar_segmap==key] = back_dict[key]
        tar_segmap = new_tar_segmap
        
        image, segmap, layout = change_furniture(opts,layout_data_root[i],fur_data_root[i],dataset='Structured3D',tar_img=tar_img,tar_segmap=tar_segmap,correct_shape=(args.height,args.width,3))

        # Image.fromarray(tar_img.astype('uint8')).save(osp.join(save_path,'full','raw.png'))
        Image.fromarray(image.astype('uint8')).save(osp.join(save_path,'full','rgb_rawlight.png'))
        segmap[segmap==255] = 0
        Image.fromarray(segmap.astype('uint8')).save(osp.join(save_path,'full','semantic.png'))
        np.savetxt(osp.join(save_path,'layout.txt'),layout,fmt='% 4d')
        Image.fromarray(colorize_segmap(segmap).astype('uint8')).save(osp.join(save_path,'full','colorized_semantic.png'))
        
        """save original Structure, Style, Furniture image"""
        Image.open(osp.join(layout_data_root[i],'full','rgb_rawlight.png')).save(osp.join(save_path,'full','structure.png'))
        Image.open(osp.join(fur_data_root[i],'full','rgb_rawlight.png')).save(osp.join(save_path,'full','furniture.png'))
        
        # save style, source, target path
        with open(osp.join(save_path,'source_path.txt'),'w') as fp:
            style_path = data_i['path'][0]
            fp.write(f'style path : {style_path}\n')
            fp.write(f'furniture path : {fur_data_root[i]}\n')
            fp.write(f'target root : {layout_data_root[i]}\n')
            if infeasible:
                fp.write('infeasible!!!\n')
        i += 1
    print(f'Use {i} round to complete {i} images')
    end_time = time.time()
    
    
