import functools

import numpy as np
from scipy.ndimage import map_coordinates
from scipy import io


def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv


@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v


def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    
    # pstart = np.ceil(min(p1[0], p2[0]))
    # pend = np.floor(max(p1[0], p2[0]))
    
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs, h)

    return np.stack([coorxs, coorys], axis=-1)

def pano_stretch(img, corners, kx=1, ky=1, order=1,which_wall=None ,semantic=False):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''
    corners = corners.astype('int')
    if semantic:# resize to 3 channel if input is semantic
        img = np.expand_dims(img,2).repeat(3,axis=2)
    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    if semantic:
        refx,refy = np.round(refx),np.round(refy)

    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='mirror')
        for i in range(img.shape[-1])
    ], axis=-1)
    # if semantic:
    #     import pdb; pdb.set_trace()
    
    # Process corners
    corners_u0 = coorx2u(corners[:, 0], img.shape[1])
    corners_v0 = coory2v(corners[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    C2 = (np.sin(corners_u0) * ky)**2 + (np.cos(corners_u0) * kx)**2
    corners_v = np.arctan2(
            np.sin(corners_v0),
            np.cos(corners_v0) * np.sqrt(C2))

    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    stretched_corners = np.stack([cornersX, cornersY], axis=-1).astype('int')

    if semantic:# resize back to 1 channel if input is semantic
        stretched_img = stretched_img[:,:,0]
    
    return stretched_img, stretched_corners

def pano_stretch_layout(corners, kx=1, ky=1):
    # Process corners
    corners_u0 = coorx2u(corners[:, 0], 1024)
    corners_v0 = coory2v(corners[:, 1], 512)
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))

    C2 = (np.sin(corners_u0) * ky)**2 + (np.cos(corners_u0) * kx)**2
    corners_v = np.arctan2(
            np.sin(corners_v0),
            np.cos(corners_v0) * np.sqrt(C2))

    cornersX = u2coorx(corners_u, 1024)
    cornersY = v2coory(corners_v, 512)
    stretched_corners = np.stack([cornersX, cornersY], axis=-1).astype('int')
    return stretched_corners

def cal_kx(corners_x,diff_x,wall_cnt,img_shape=(512,1024)):
    # make sure that stretch left length equals to stretch right length
    # corners_x = corners_x-corners_x[wall_cnt]+109

    corners_u0 = coorx2u(corners_x, img_shape[1])
    new_corner = np.array([corners_x[i] - diff_x if i in [0,1,4,5] \
        else corners_x[i] + diff_x  for i in range(len(corners_x))])
    corner_u = coorx2u(new_corner, img_shape[1])
    # kx = np.round(np.tan(corner_u)*np.cos(corners_u0)/np.sin(corners_u0),3)
    kx = np.round(np.sin(corners_u0)/(np.tan(corner_u)*np.cos(corners_u0)),3)
    # return kx[wall_cnt*2],new_corner
    return kx[0],new_corner,

if __name__ == '__main__':
    tar_layout = np.asarray(io.loadmat('/media/public_dataset/Stanford2D3D/area_5b/pano/layout/\
camera_e3b71681093945618304805844a9bcac_office_25_frame_equirectangular_domain_.mat')['cor'])//4
    src_layout = np.loadtxt('./tmp_layout.txt')//4
    print(src_layout)
    # import pdb; pdb.set_trace()
    # tar_width = tar_layout[4][0] - tar_layout[2][0]
    # src_width = src_layout[4][0] - src_layout[2][0]
    tar_width = tar_layout[2][0] - tar_layout[0][0]
    src_width = src_layout[2][0] - src_layout[0][0]
    src_layout = np.array(([[0,100],[0,300],[350,100],[350,300]]))
    src_layout[:,0] = src_layout[:,0]-src_layout[0][0]+128 - (src_layout[2][0]-256)//2
    
    print(cal_kx(src_layout[:,0],(tar_width-src_width)//2,1))
    
    # print(pano_stretch_layout(src_layout,kx=1.4))
    # import pdb; pdb.set_trace()