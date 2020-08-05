import sys
sys.path.append("..")
import os
from utils.params import *
from utils.estimate_pose import matrix2angle,P2sRt
import cv2
import numpy as np
from .vexcol import mix_shp,asian_shp,european_shp

def param2pose(param):
    param = param * param_std + param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    pose = matrix2angle(R)  # yaw, pitch, roll
    return pose

def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp

def reconstruct_vertex_front(param, population='mix',norm=True):
    
    
    param = param * param_std + param_mean
    p, offset, alpha_shp, alpha_exp = _parse_param(param)
    vertex = (u_shp + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') /1000
    if population=='mix':
        vertex = vertex - european_shp + mix_shp    
    elif population=='asian':
        vertex = vertex - european_shp + asian_shp
    else:
        print('still no this population')

    if norm:
        vertex = (vertex - np.min(vertex))/np.ptp(vertex)  
    else:
        vertex =vertex * 1000

    return vertex

def write_obj(obj_name, vertices, triangles):
    ''' 
    Args:
        obj_name: str
        vertices: shape = (3, nver)
        triangles: shape = (3, ntri) one-based
    '''
    triangles = triangles.copy() # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[2, i], triangles[1, i], triangles[0, i])
            f.write(s)

def write_obj_with_colorsrgb(obj_name, vertices, triangles, colors):
    ''' 
    Args:
        obj_name: str
        vertices: shape = (3, nver)
        triangles: shape = (3, ntri) one-based
        colors: shape = (3, nver) RGB
    '''
    triangles = triangles.copy() # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i], colors[0,i],
                                               colors[1,i], colors[2,i])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[2, i], triangles[1, i], triangles[0, i])
            f.write(s)
            


def write_obj_with_texture(save_folder, obj_name, vertices, triangles, isomap, uv_coord):
    ''' Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (3, nver)
        triangles: shape = (3, ntri) one-based
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) or (nver, 2) max value<=1
    '''
    
    triangles = triangles.copy() # meshlab start with 1
    
    basename = obj_name.split('.')[0]
    mti_filename = basename + '.mtl'
    iso_filename = basename + '.isomap.png'
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    obj_name = os.path.join(save_folder,obj_name)
    
    # write obj
    with open(obj_name, 'w') as f:
        # first line
        s = 'mtllib '+ mti_filename + '\n'
        f.write(s)

        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i])
            f.write(s)
        for i in range(uv_coord.shape[0]):
            s = 'vt {:.6f} {:.6f} \n'.format(uv_coord[i, 0], uv_coord[i, 1])
            f.write(s)
        s = 'usemtl FaceTexture\n'
        f.write(s)
        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[2, i], triangles[2, i], triangles[1, i],triangles[1, i],  triangles[0, i],triangles[0, i])
            f.write(s)
    with open(os.path.join(save_folder,mti_filename), 'w') as f:
        s = 'newmtl FaceTexture\n'
        f.write(s)
        s = 'map_Kd ' + iso_filename + '\n'
        f.write(s)
    cv2.imwrite(os.path.join(save_folder,iso_filename), isomap)
        
def write_obj_with_expression(save_folder, obj_name, param, expression, triangles, uv_coord,ver_ind = None):
    ''' Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (3, nver)
        triangles: shape = (3, ntri) one-based
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) or (nver, 2) max value<=1
    '''
    
    triangles = triangles.copy() # meshlab start with 1
    
    basename = obj_name.split('.')[0]
    iso_filename = basename + '.isomap.png'
    
    for basename,(k,v) in enumerate(expression.items()):
        basename = str(basename+1)
        mti_filename = basename + '.mtl'
        obj_name = basename + '.obj'
        obj_name = os.path.join(save_folder,obj_name)
        new_mean_param = param.copy()
        new_mean_param[k] = v
        vertices = reconstruct_vertex_front(new_mean_param)
        if ver_ind is not None:
            vertices = vertices[:,ver_ind]
        # write obj
        with open(obj_name, 'w') as f:
            # first line
            s = 'mtllib '+ mti_filename + '\n'
            f.write(s)

            for i in range(vertices.shape[1]):
                s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i])
                f.write(s)
            for i in range(uv_coord.shape[0]):
                s = 'vt {:.6f} {:.6f} \n'.format(uv_coord[i, 0], uv_coord[i, 1])
                f.write(s)
            s = 'usemtl FaceTexture\n'
            f.write(s)
            # write f: ver ind/ uv ind
            for i in range(triangles.shape[1]):
                s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[2, i], triangles[2, i], triangles[1, i],triangles[1, i],  triangles[0, i],triangles[0, i])
                f.write(s)
        with open(os.path.join(save_folder,mti_filename), 'w') as f:
            s = 'newmtl FaceTexture\n'
            f.write(s)
            s = 'map_Kd ' + iso_filename + '\n'
            f.write(s)
            