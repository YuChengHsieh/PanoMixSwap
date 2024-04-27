from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from utils.panostretch import *
# from panostretch import *

def colorize_segmap(segmap):
    PALETTE = [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet (wall3)
       (255, 187, 120),		# bed (wall0)
       (188, 189, 34), 		# chair (wall1)
       (140, 86, 75),  		# sofa (wall2)
       (255, 152, 150),		# table 
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf 
       (196, 156, 148),		# picture (others)
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
       (100, 85, 144)
    ]  
    
    colorized_segmap = np.zeros((segmap.shape[0],segmap.shape[1],3))
    segmap =segmap.astype('uint8')
    segmap[segmap==255] = 0
    values = np.unique(segmap)
    for value in values:
        colorized_segmap[segmap==value] = PALETTE[value]
    
    return colorized_segmap

class WallImage:
    def __init__(self,data_root: str, wall_cnt: int, dataset: str, \
        correct_shape: Tuple=(512,1024,4)) -> None:
        if dataset == 'Structured3D':
            image = np.asarray(Image.open(osp.join(data_root,'full/rgb_rawlight.png')))[:,:,:3].copy()
            segmap = np.asarray(Image.open(osp.join(data_root,'full/semantic.png'))).copy()
            self.layout = np.loadtxt(osp.join(data_root,'layout.txt')).astype('int')
        elif dataset == 'Stanford2D3D':
            # image
            split_root = data_root.split('/')
            split_root[6] = 'img'
            split_root[-1] = split_root[-1][:-4]
            split_root[-1] += 'rgb.png'
            image = np.asarray(Image.open('/'.join(split_root))).copy()
            # segmap
            split_root[6] = 'semantic'
            split_root[-1] = split_root[-1][:-7]
            split_root[-1] += 'rgb.png'
            segmap = np.asarray(Image.open('/'.join(split_root))).copy()
            self.layout = np.loadtxt(data_root).astype('int')*(correct_shape[0]//512)
        else:
            raise NotImplementedError(f'unknown dataset name {dataset}')
        self.data_root = data_root


        if image.shape != correct_shape or segmap.shape != correct_shape[:-1]:
            self.image,self.segmap = self.resize_image(image,segmap,correct_size=correct_shape)
        else:
            self.image,self.segmap = image,segmap
        self.shape = self.image.shape

        assert self.shape == correct_shape, f'Image shape must be {correct_shape}'
        assert self.segmap.shape == correct_shape[:2], f'Segmentation shape must be {correct_shape[:2]}'
        assert self.layout.shape[0] == 8

        # things will not change ever
        self.original_image = self.image.copy()
        self.original_segmap = self.segmap.copy()
        
        
        self.wall_cnt = wall_cnt
        self.leftUp,self.leftDown,self.rightUp,self.rightDown = \
            self.layout[0+self.wall_cnt*2],self.layout[1+self.wall_cnt*2],\
            self.layout[(2+self.wall_cnt*2)%8],self.layout[(3+self.wall_cnt*2)%8]

        self.wall_width = self.rightUp[0] - self.leftUp[0] if self.rightUp[0] > self.leftUp[0] \
                else self.shape[1] - self.leftUp[0] + self.rightUp[0]

        self.connectedPtUp = pano_connect_points(self.leftUp,self.rightUp,z=-50,w=self.shape[1],h=self.shape[0]).astype('int')
        self.connectedPtDown = pano_connect_points(self.leftDown,self.rightDown,w=self.shape[1],h=self.shape[0],z=50).astype('int')
        self.move_pt = self.shape[1] - self.layout[6][0]
        
    @staticmethod
    def resize_image(image: np.ndarray,segmap: np.ndarray,correct_size: Tuple=(512,1024,4)) \
        -> Tuple[np.ndarray, np.ndarray]:

        dim = (correct_size[1],correct_size[0])
        image = cv2.resize(image,dim,interpolation=cv2.INTER_NEAREST)
        segmap = cv2.resize(segmap,dim,interpolation=cv2.INTER_NEAREST)

        # if image.shape[-1] != 4:
        #     new_axis = np.zeros((*correct_size[:-1],1),dtype='uint8')
        #     new_axis.fill(255)
        #     image = np.concatenate((image,new_axis),axis=2)

        return image,segmap
    
    @staticmethod
    def roll_wall(image:np.ndarray,segmap:np.ndarray,layout:np.ndarray,\
        orig_image:np.ndarray,orig_segmap:np.ndarray,move_pt:int)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            input:
                image, segmentation map, layout
                phase:
                    True: roll 
                    False: roll back to original image
            output:
                image, segmentation map, layout
            description:
                when wall_cnt = 3,  wall will spilt into 2 half.
                this function roll the image and segmap to 1 combined wall
        """
        image = np.roll(image,move_pt,axis=1)
        segmap = np.roll(segmap,move_pt,axis=1)
        layout[:,0] = (layout[:,0]+move_pt)%image.shape[1]
        orig_image = np.roll(orig_image,move_pt,axis=1)
        orig_segmap = np.roll(orig_segmap,move_pt,axis=1)
        return image,segmap,layout,orig_image,orig_segmap

    def change_wall_attribute(self, wall_cnt: int) -> None:
        """
            change attributes when process different wall
        """
        self.wall_cnt = wall_cnt
        self.leftUp,self.leftDown,self.rightUp,self.rightDown = \
            self.layout[0+self.wall_cnt*2],self.layout[1+self.wall_cnt*2], \
            self.layout[(2+self.wall_cnt*2)%8],self.layout[(3+self.wall_cnt*2)%8]

        self.wall_width = self.rightUp[0] - self.leftUp[0] if self.rightUp[0] > self.leftUp[0] \
                else self.shape[1] - self.leftUp[0] + self.rightUp[0]

        self.connectedPtUp = pano_connect_points(self.leftUp,self.rightUp,z=-50,w=self.shape[1],h=self.shape[0]).astype('int')
        self.connectedPtDown = pano_connect_points(self.leftDown,self.rightDown,z=50,w=self.shape[1],h=self.shape[0]).astype('int')

        
    def change_input_attribute(self,image: np.ndarray=None,segmap: np.ndarray=None,\
        layout: np.ndarray=None, phase: bool=True) -> None:
        """
            change input image, segmap, layout only when 2 conditions:
            1. panostretch transfrmation
            2. wall_cnt = 3 , perform roll wall
        """

        assert type(image) == type(segmap) == type(layout), \
            'image & segmap & layout must be changed simultaneously'

        # roll wall 3
        if image == segmap == layout == None:
            move_pt = self.move_pt if phase else -self.move_pt
            self.image,self.segmap,self.layout,self.original_image,self.original_segmap = \
                self.roll_wall(self.image,self.segmap,self.layout,self.original_image,self.original_segmap,move_pt)

        # panostretch transformation
        else:
            self.image,self.segmap,self.layout = image,segmap,layout
            self.move_pt = self.shape[1] - self.layout[6][0]
        
        self.change_wall_attribute(self.wall_cnt)

class SourceImageProcessor(WallImage):
    def __init__(self,data_root: str,wall_cnt: int, dataset: str, correct_shape:Tuple=(512,1024,4)) -> None:
        super().__init__(data_root,wall_cnt,dataset,correct_shape=correct_shape)
    def panostretch_transform(self,tar_width: int):
        
        src_width = self.wall_width
        
        # roll to make sure 
        # roll process wall into 0 wall
        roll_pt = int(-self.layout[self.wall_cnt*2][0]+self.shape[0]//4 -(src_width-self.shape[1]//4)//2)
        self.layout[:,0] += roll_pt
        self.layout[self.layout<0] += self.shape[1]
        self.layout = np.roll(self.layout,-self.wall_cnt*2,axis=0)
        self.image = np.roll(self.image,roll_pt,axis=1)
        self.segmap = np.roll(self.segmap,roll_pt,axis=1)
        kx, layout_x = cal_kx(self.layout[:,0],(tar_width-src_width)/2,self.wall_cnt,img_shape=self.shape[:2])
        # import pdb; pdb.set_trace()
        segmap ,_ = pano_stretch(self.segmap,self.layout,kx=kx,which_wall=self.wall_cnt,semantic=True)
        image, layout = pano_stretch(self.image,self.layout,kx=kx)
        
        src_width = layout[2][0] - layout[0][0]
        
        if src_width < tar_width:
            layout[2][0] += tar_width-src_width
            layout[3][0] += tar_width-src_width
            src_width += tar_width-src_width
        elif src_width > tar_width:
            layout[2][0] -= src_width-tar_width
            layout[3][0] -= src_width-tar_width
            src_width -= src_width-tar_width

        # roll back layout
        if self.wall_cnt >= 1:
            roll_pt = self.shape[1]-layout[8-self.wall_cnt*2][0]
            layout[:,0] += roll_pt
            layout[layout>=self.shape[1]] -= self.shape[1]
            layout = np.roll(layout,self.wall_cnt*2,axis=0)
            image = np.roll(image,roll_pt,axis=1)
            segmap = np.roll(segmap,roll_pt,axis=1)
        
        # assert src_width == tar_width,'source width doesn''t match target width'
        self.change_input_attribute(image,segmap,layout)
        # Save pipeline results
        # import pdb; pdb.set_trace()
        # image_save = np.roll(image,-16,axis=1)
        # layout_save = layout.copy()
        # layout_save[:,0] = layout[:,0]-16
        
        # image = np.roll(image,-layout[0][0],axis=1)
        # segmap = np.roll(segmap,-layout[0][0],axis=1)
        # Image.fromarray(image.astype('uint8')).save('FFB/HA/fur_stretched.png')
        # Image.fromarray(colorize_segmap(segmap).astype('uint8')).save('FFB/HA/fur_stretched_sem.png')
        # layout[:,0] = layout[:,0]-layout[0][0]
        # np.savetxt('FFB/HA/fur_stretched_layout.txt',layout,fmt='%3d')
        # import pdb; pdb.set_trace()
        # np.savetxt('panostretch_figure/fur_stretched.txt',layout,fmt='%.3d')
        # import pdb; pdb.set_trace()
        # print(f'perform panostretch transform on source wall {self.wall_cnt} '+
        #     f'with parameters kx = {kx:.2f} final source width = {self.wall_width} , target width = {tar_width} ')

    def get_exceeded_pts_and_label(self,col_cnt: int,up: bool,threshold: int=3,
                                   furnitures_X: list=[]) -> Tuple[int,int]:
        """
        get total extrapolation pts of every column
        """
        extended_pt = 0
        x = self.connectedPtUp[col_cnt][0]
        y = self.connectedPtUp[col_cnt][1] if up \
            else self.connectedPtDown[col_cnt][1]
    
        threshold_cnt = 0
        # add threshold to adjust the connectedPT and segmentation map gap 
        while  y >= 0 and y < self.shape[0]:
            label = self.segmap[y][x]
            if threshold_cnt >= threshold:
                extended_pt -= threshold
                break
            threshold_cnt = threshold_cnt + 1 if self.segmap[y][x] in [0,1,2,22,*furnitures_X] \
                or self.segmap[y][x] != label else 0
            extended_pt += 1
            y = y-1 if up else y+1
        
        label = self.segmap[y+threshold+1][x] if up else self.segmap[y-threshold-1][x]

        return extended_pt,label

    def cal_interpolated_pixels(self, target: TargetImageProcessor, \
        semantic: bool=False) -> np.ndarray:
        """
        interpolate the source height, make source height equals to target height of a column
        input: 
            col_cnt
        output: 
            pixels_values : np.array of query points' rgb values, be like array[(r,g,b),(r,g,b)]
            pts : np.array of query points' column, be like array([col,col])
        """
        assert isinstance(target,TargetImageProcessor), 'unknown object of target'
        src_height = self.connectedPtDown[:,1] - self.connectedPtUp[:,1]
        tar_height = target.connectedPtDown[:,1] - target.connectedPtUp[:,1]
        # import pdb; pdb.set_trace()
        # ratio = max(src_height/tar_height)
        # pts = np.array([ j*ratio + self.connectedPtUp[:,1].astype('int') for j in range(max(tar_height)) ]) 
        pts = np.array([ j*src_height/tar_height + self.connectedPtUp[:,1].astype('int') for j in range(max(tar_height)) ])  
        x = np.asarray([np.arange(self.leftUp[0],self.rightUp[0]+1) for _ in range(max(tar_height))])

        if semantic:
            ret_values = map_coordinates(self.segmap,[np.round(pts).astype('int'),x],mode='nearest')
        else:
            # image interpolation of a column
            ret_values =  np.stack([
                map_coordinates(self.image[...,j],[pts,x],mode='nearest')
                for j in range(self.shape[-1])],axis=-1)
            # cv2.imwrite('tmp.png',ret_values)
            # import pdb; pdb.set_trace()
        return ret_values

    def cal_extrapolated_pixels(self, target: TargetImageProcessor,
                                up: bool) -> np.ndarray:
        """
        
        """
        # src_height = self.connectedPtDown[:,1]-self.connectedPtUp[:,1]
        # tar_height = target.connectedPtDown[:,1]-target.connectedPtUp[:,1]
        # ratio = src_height/tar_height
        ratio = 0.75
        pts = []
        up_threshold = int(max(self.connectedPtUp[:,1])/ratio)
        down_threshold = int(min(self.connectedPtDown[:,1])/ratio)
        
        pts = np.asarray([self.connectedPtUp[:,1] - i*ratio for i in range(up_threshold)]) if up  else \
            np.asarray([self.connectedPtDown[:,1] + i*ratio for i in range(int(self.shape[0]/ratio)-down_threshold)])
        
        
        x = np.asarray([ np.arange(self.leftUp[0],self.rightUp[0]+1) for _ in range(pts.shape[0])])
        ret_rgb = np.stack([
                map_coordinates(self.image[...,j],[pts,x],mode='nearest')
                for j in range(self.shape[-1])],axis=-1)
        
        ret_segmap = map_coordinates(self.segmap,[np.round(pts),x],mode='nearest')
        # import pdb; pdb.set_trace()
        return ret_rgb,ret_segmap

    def filter_out_segmap_class(self,segmap_col:np.ndarray,col_cnt: int, interpolated_col_coors: list) -> np.ndarray:
        """
        filter out some unrecognizable segmentation labels 
        generated in interpolation
        """
        col_coor = col_cnt+self.leftUp[0]
        for i in range(len(segmap_col)):
            choices = [self.segmap[i_][j_] for j_ in range(max(0,col_coor-3),min(self.shape[1],col_coor+3))\
                for i_ in range(int(interpolated_col_coors[i])-3,int(interpolated_col_coors[i])+3)]
            if segmap_col[i] not in choices:
                # segmap label of query pixel is the most segmap label around the query pixel
                segmap_col[i] = max(choices,key=choices.count)
                

        return segmap_col



class TargetImageProcessor(WallImage):
    def __init__(self,data_root: str, wall_cnt: int,dataset: str,tar_img: np.ndarray, \
        tar_segmap:np.ndarray, correct_shape:Tuple = (512,1024,4)):
        super().__init__(data_root,wall_cnt,dataset,correct_shape=correct_shape)
        if tar_img is not None and tar_segmap is not None:
            image = tar_img
            segmap = tar_segmap
            
            if image.shape != correct_shape or segmap.shape != correct_shape[:-1]:
                self.image,self.segmap = self.resize_image(image,segmap,correct_size=correct_shape)
            else:
                self.image,self.segmap = image,segmap
            
            
            """ For Furniture Fusing Block figure """
            # from draw_layout import draw_layout_point
            # Image.fromarray(draw_layout_point(self.image,self.layout)).save('tmp.png')
            # layout = self.layout
            # layout[:,0] = layout[:,0]-layout[0][0]
            # self.change_input_attribute(self.image,segmap=self.segmap,layout=layout)
        self.dataset = dataset

    def get_replace_coor(self,col_cnt: int):
        """
            get the coordinate of target image that will be replaced by 
            source image
        """
        return [(i,col_cnt+self.leftUp[0]) for i in range(self.connectedPtUp[col_cnt][1],self.connectedPtDown[col_cnt][1])]
    
    def paste_src_furniture(self,interpolated_image: np.ndarray,interpolated_segmap: np.ndarray, 
        extrapolated_upImage: np.ndarray,extrapolated_downImage: np.ndarray, extrapolated_upSegmap: np.ndarray,
        extrapolated_downSegmap: np.ndarray, furnitures_X: list=[]) -> dict:

        """
            call per column
            paste source wall furnitures to target wall 
        """
        # paste up extend image 
        if self.dataset == 'Structured3D':
            up_mask = ~((extrapolated_upSegmap == 0) | (extrapolated_upSegmap == 1)\
                | (extrapolated_upSegmap == 2) | (extrapolated_upSegmap == 22))
            up_mask = up_mask[::-1]
            extrapolated_upImage = extrapolated_upImage[::-1]
            extrapolated_upSegmap = extrapolated_upSegmap[::-1]

            mask = ~((interpolated_segmap == 0) | (interpolated_segmap == 1) | \
                (interpolated_segmap == 2) | (interpolated_segmap == 22))
            
            down_mask = ~((extrapolated_downSegmap == 0) | (extrapolated_downSegmap == 1) | \
                (extrapolated_downSegmap == 2) | (extrapolated_downSegmap == 22))
        elif self.dataset == 'Stanford2D3D':
            up_mask = ~((extrapolated_upSegmap == 4) | (extrapolated_upSegmap == 9)\
                | (extrapolated_upSegmap == 12))
            up_mask = up_mask[::-1]
            extrapolated_upImage = extrapolated_upImage[::-1]
            extrapolated_upSegmap = extrapolated_upSegmap[::-1]

            mask = ~((interpolated_segmap == 4) | (interpolated_segmap == 9) | \
                (interpolated_segmap == 12))
            
            down_mask = ~((extrapolated_downSegmap == 4) | (extrapolated_downSegmap == 9) | \
                (extrapolated_downSegmap == 12))


        for i in range(self.wall_width):
            x = self.leftUp[0]+i
            y_up = self.connectedPtUp[i,1]
            y_down = self.connectedPtDown[i,1]
            #up
            tmp_up_mask = up_mask[-y_up:,i]
            if y_up - extrapolated_upImage.shape[0] >= 0 :
                self.image[y_up-extrapolated_upImage.shape[0]:y_up,x][tmp_up_mask] = extrapolated_upImage[-y_up:,i][tmp_up_mask]
                self.segmap[y_up-extrapolated_upImage.shape[0]:y_up,x][tmp_up_mask] = extrapolated_upSegmap[-y_up:,i][tmp_up_mask]
            elif y_up != 0: # exceed limit (if y_up == 0 , -y_up will call all length of extrapolated_upImage ) --> error
                self.image[:y_up,x][tmp_up_mask] = extrapolated_upImage[-y_up:,i][tmp_up_mask]
                self.segmap[:y_up,x][tmp_up_mask] = extrapolated_upSegmap[-y_up:,i][tmp_up_mask]
            
            #middle
            tmp_mask = mask[: y_down-y_up, i]
            self.image[y_up:y_down,x][tmp_mask] = interpolated_image[: y_down-y_up,i][tmp_mask]
            self.segmap[y_up:y_down,x][tmp_mask] = interpolated_segmap[: y_down-y_up,i][tmp_mask]
        
            #down
            tmp_down_mask = down_mask[:self.shape[0]-y_down,i]
            if y_down + extrapolated_downImage.shape[0] <= self.shape[0]: 
                self.image[y_down:y_down+extrapolated_downImage.shape[0],x][tmp_down_mask] = extrapolated_downImage[:self.shape[0]-y_down,i][tmp_down_mask]
                self.segmap[y_down:y_down+extrapolated_downImage.shape[0],x][tmp_down_mask] = extrapolated_downSegmap[:self.shape[0]-y_down,i][tmp_down_mask]
            else: #exceed limit
                self.image[y_down:self.shape[0],x][tmp_down_mask] = extrapolated_downImage[:self.shape[0]-y_down,i][tmp_down_mask]
                self.segmap[y_down:self.shape[0],x][tmp_down_mask] = extrapolated_downSegmap[:self.shape[0]-y_down,i][tmp_down_mask]
            
    def paste_origTarget_furniture(self,furnitures_X: list=[]) -> None:
        """
            paste orignal target images' ceiling or floor furnitures 
            to final image
        """
        for idx,col in enumerate(range(self.leftUp[0],self.rightUp[0]+1)):
            # repaste original target image ceiling furniture
            while self.original_segmap[self.connectedPtUp[idx][1],col] != 22:
                self.connectedPtUp[idx][1] -= 1

            for row in range(self.connectedPtUp[idx][1]):
                if self.original_segmap[row][col] not in [0,1,2,22,*furnitures_X]:
                    self.image[row][col] = self.original_image[row][col]
                    self.segmap[row][col] = self.original_segmap[row][col]
            
            # repaste target image floor furniture
            while self.original_segmap[self.connectedPtDown[idx][1],col] != 2:
                self.connectedPtDown[idx][1] += 1

            for row in range(self.connectedPtDown[idx][1],self.shape[0]):
                if self.original_segmap[row][col] not in [0,1,2,22,*furnitures_X]:
                    self.image[row][col] = self.original_image[row][col]
                    self.segmap[row][col] = self.original_segmap[row][col]
    
    def visualized_connected_points(self) -> None:
        for idx,col in enumerate(range(self.leftUp[0],self.rightUp[0]+1)):
            self.image[self.connectedPtUp[idx][1]][col] = np.array([0,0,0,255])
            self.image[self.connectedPtDown[idx][1]][col] = np.array([0,0,0,255])
            
    # def save_Image(self,out_dir,name):
    #     """
    #         save image to path './file_name/'
    #     """
    #     if not osp.exists(out_dir):
    #         os.mkdir(out_dir)

    #     Image.fromarray(self.image.astype('uint8')).save(f'{out_dir}/rgb_rawlight.png')
    #     Image.fromarray(self.segmap.astype('uint8')).save(f'{out_dir}/semantic.png')
    #     np.savetxt(f'{out_dir}/layout.txt',self.layout)
    #     Image.fromarray(colorize_segmap(self.segmap).astype('uint8')).save(f'{out_dir}/colorized_semantic.png')

if __name__ == '__main__':
    import pickle
    
    with open("root_path/root_path_train_8layout", "rb") as fp:   # Unpickling
        data_root = pickle.load(fp)
