'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''


import os
import cv2
import sys
import argparse
import numpy as np
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.project_on_mesh import compute_texture_map


def build_texture_from_image(source_img_fname, target_mesh_fname, target_scale_fname, texture_mapping, out_path):
    if not os.path.exists(source_img_fname):
        print('Source image not found - %s' % source_img_fname)
        return
    if not os.path.exists(target_mesh_fname):
        print('Target mesh not found - %s' % target_mesh_fname)
        return
    if not os.path.exists(target_scale_fname):
        print('Scale information for target mesh not found %s' % target_scale_fname)
        return
    if not os.path.exists(texture_mapping):
        print('Pre-computed FLAME texture mapping not found %s' % texture_mapping)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    source_img = cv2.imread(source_img_fname)
    target_mesh = Mesh(filename=target_mesh_fname)
    target_mesh.set_vertex_colors('white')
    target_scale = np.load(target_scale_fname)

    if sys.version_info >= (3, 0):
        texture_data = np.load(texture_mapping, allow_pickle=True, encoding='latin1').item()
    else:
        texture_data = np.load(texture_mapping, allow_pickle=True).item()
    texture_map = compute_texture_map(source_img, target_mesh, target_scale, texture_data)

    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_mesh_fname))[0] + '.obj')
    out_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_mesh_fname))[0] + '.png')

    cv2.imwrite(out_img_fname, texture_map)
    target_mesh.vt = texture_data['vt']
    target_mesh.ft = texture_data['ft']
    target_mesh.set_texture_image(out_img_fname)
    target_mesh.write_obj(out_mesh_fname)
    target_mesh.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build texture from image')
    parser.add_argument('--source_img', default='./data/imgHQ00088.jpeg', help='source image filename')
    parser.add_argument('--target_mesh', default='./results/imgHQ00088.obj', help='target mesh filename')
    parser.add_argument('--target_scale', default='./results/imgHQ00088_scale.npy', help='scale of the target mesh for the image projection')
    parser.add_argument('--texture_mapping', default='./data/texture_data.npy', help='pre-computed FLAME texture mapping')
    parser.add_argument('--out_path', default='./results', help='output path')
    args = parser.parse_args()

    build_texture_from_image(args.source_img, args.target_mesh, args.target_scale, args.texture_mapping, args.out_path)