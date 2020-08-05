import numpy as np
import os.path as osp
import scipy.io as sio

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


d = make_abs_path('deec.config')

#mean faces
mix_shp = np.load(osp.join(d, 'asian.npy'))
asian_shp = np.load(osp.join(d, 'euro.npy'))
european_shp = np.load(osp.join(d, 'mix.npy'))

tri = sio.loadmat(osp.join(d, 'tri.mat'))['tri']
uv_coord = sio.loadmat(osp.join(d, 'BFM_UV.mat'))['UV'] # between 0 and 1

upface_idx = np.load(osp.join(d, 'vertex_idx_noneck_0base.npy'))
upface_tri_new = np.load(osp.join(d,'tri_noneck_1base.npy'))

front_ind = np.load(osp.join(d,'front_full_ear_ind.npy'))
front_tri = np.load(osp.join(d,'front_full_ear_tri.npy'))

#zz1=np.load(osp.join(d,'wgt_colors_mid.npy'))
#zz2=np.load(osp.join(d,'wgt_colors_left.npy'))
#zz3=np.load(osp.join(d,'wgt_colors_right.npy'))
wgt = np.load(osp.join(d,'wgt_colors.npy'))
lf_cheek_ind = np.load(osp.join(d,'lf_cheek_ind.npy'))
rf_cheek_ind = np.load(osp.join(d,'rf_cheek_ind.npy'))

def process_uv(uv_coords, uv_h = 256, uv_w = 256,updown=True):
    uv_coords = uv_coords.copy()
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    if updown:
        uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def weight_colors(colors_lst,weight=wgt):
    zz0,zz1,zz2,zz3,zz4,zz5,zz6,zz7 = weight
    wgt_colors = np.uint8((colors_lst[0]*zz0.reshape(-1,1) + 
                           colors_lst[0]*zz1.reshape(-1,1) + 
                           colors_lst[2]*zz2.reshape(-1,1)+
                           colors_lst[1]*zz3.reshape(-1,1)+
                          colors_lst[2][lf_cheek_ind].mean(0)*zz4.reshape(-1,1)+
                          colors_lst[2]*zz5.reshape(-1,1)+
                           colors_lst[1][rf_cheek_ind].mean(0)*zz6.reshape(-1,1)+
                          colors_lst[1]*zz7.reshape(-1,1))/
                           (zz0+zz1+zz2+zz3+zz4+zz5+zz6+zz7).reshape(-1,1))
    return wgt_colors