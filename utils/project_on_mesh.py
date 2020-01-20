import numpy as np
from utils.landmarks import tf_project_points

def compute_texture_map(source_img, target_mesh, target_scale, texture_data):
    '''
    Given an image and a mesh aligned with the image (under scale-orthographic projection), project the image onto the
    mesh and return a texture map.
    :param source_img:      source image
    :param target_mesh:     mesh in FLAME mesh topology aligned with the source image
    :param target_scale:    scale of mesh for the projection
    :param texture_data:    pre-computed FLAME texture data
    :return:                computed texture map
    '''

    x_coords = texture_data.get('x_coords')
    y_coords = texture_data.get('y_coords')
    valid_pixel_ids = texture_data.get('valid_pixel_ids')
    valid_pixel_3d_faces = texture_data.get('valid_pixel_3d_faces')
    valid_pixel_b_coords = texture_data.get('valid_pixel_b_coords')

    pixel_3d_points = target_mesh.v[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                      target_mesh.v[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                      target_mesh.v[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]

    vertex_normals = target_mesh.estimate_vertex_normals()
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    n_dot_view = -pixel_3d_normals[:,2]

    proj_2d_points = np.round(target_scale*pixel_3d_points[:,:2], 0).astype(int)
    proj_2d_points[:, 1] = source_img.shape[0] - proj_2d_points[:, 1]

    texture = np.zeros((512, 512, 3))
    for i, (x, y) in enumerate(proj_2d_points):
        if n_dot_view[i] > 0.0:
            continue
        if x > 0 and x < source_img.shape[1] and y > 0 and y < source_img.shape[0]:
            texture[y_coords[valid_pixel_ids[i]].astype(int), x_coords[valid_pixel_ids[i]].astype(int), :3] = source_img[y, x]
    return texture