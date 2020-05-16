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
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from utils.landmarks import load_embedding, tf_get_model_lmks, create_lmk_spheres, tf_project_points
from utils.project_on_mesh import compute_texture_map

from tf_smpl.batch_smpl import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt


def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

def fit_lmk2d(target_img, target_2d_lmks, model_fname, lmk_face_idx, lmk_b_coords, weights, visualize):
    '''
    Fit FLAME to 2D landmarks
    :param target_2d_lmks      target 2D landmarks provided as (num_lmks x 3) matrix
    :param model_fname         saved FLAME model
    :param lmk_face_idx        face indices of the landmark embedding in the FLAME topology
    :param lmk_b_coords        barycentric coordinates of the landmark embedding in the FLAME topology
                                (i.e. weighting of the three vertices for the trinagle, the landmark is embedded in
    :param weights             weights of the individual objective functions
    :param visualize           visualize fitting progress
    :return: a mesh with the fitting results
    '''

    tf_trans = tf.Variable(np.zeros((1,3)), name="trans", dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1,3)), name="rot", dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1,12)), name="pose", dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1,300)), name="shape", dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1,100)), name="expression", dtype=tf.float64, trainable=True)
    smpl = SMPL(model_fname)
    tf_model = tf.squeeze(smpl(tf_trans,
                               tf.concat((tf_shape, tf_exp), axis=-1),
                               tf.concat((tf_rot, tf_pose), axis=-1)))

    with tf.Session() as session:
        # session.run(tf.global_variables_initializer())

        # Mirror landmark y-coordinates
        target_2d_lmks[:,1] = target_img.shape[0]-target_2d_lmks[:,1]

        lmks_3d = tf_get_model_lmks(tf_model, smpl.f, lmk_face_idx, lmk_b_coords)

        s2d = np.mean(np.linalg.norm(target_2d_lmks-np.mean(target_2d_lmks, axis=0), axis=1))
        s3d = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(lmks_3d-tf.reduce_mean(lmks_3d, axis=0))[:, :2], axis=1)))
        tf_scale = tf.Variable(s2d/s3d, dtype=lmks_3d.dtype)

        # trans = 0.5*np.array((target_img.shape[0], target_img.shape[1]))/tf_scale
        # trans = 0.5 * s3d * np.array((target_img.shape[0], target_img.shape[1])) / s2d
        lmks_proj_2d = tf_project_points(lmks_3d, tf_scale, np.zeros(2))

        factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))
        lmk_dist = weights['lmk']*tf.reduce_sum(tf.square(tf.subtract(lmks_proj_2d, target_2d_lmks))) / (factor ** 2)
        neck_pose_reg = weights['neck_pose']*tf.reduce_sum(tf.square(tf_pose[:,:3]))
        jaw_pose_reg = weights['jaw_pose']*tf.reduce_sum(tf.square(tf_pose[:,3:6]))
        eyeballs_pose_reg = weights['eyeballs_pose']*tf.reduce_sum(tf.square(tf_pose[:,6:]))
        shape_reg = weights['shape']*tf.reduce_sum(tf.square(tf_shape))
        exp_reg = weights['expr']*tf.reduce_sum(tf.square(tf_exp))

        session.run(tf.global_variables_initializer())

        if visualize:
            def on_step(verts, scale, faces, target_img, target_lmks, opt_lmks, lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0, neck_pose_reg=0.0, jaw_pose_reg=0.0, eyeballs_pose_reg=0.0):
                import cv2
                import sys
                import numpy as np
                from psbody.mesh import Mesh
                from utils.render_mesh import render_mesh

                if lmk_dist>0.0 or shape_reg>0.0 or exp_reg>0.0 or neck_pose_reg>0.0 or jaw_pose_reg>0.0 or eyeballs_pose_reg>0.0:
                    print('lmk_dist: %f, shape_reg: %f, exp_reg: %f, neck_pose_reg: %f, jaw_pose_reg: %f, eyeballs_pose_reg: %f' % (lmk_dist, shape_reg, exp_reg, neck_pose_reg, jaw_pose_reg, eyeballs_pose_reg))

                plt_target_lmks = target_lmks.copy()
                plt_target_lmks[:, 1] = target_img.shape[0] - plt_target_lmks[:, 1]
                for (x, y) in plt_target_lmks:
                    cv2.circle(target_img, (int(x), int(y)), 4, (0, 0, 255), -1)

                plt_opt_lmks = opt_lmks.copy()
                plt_opt_lmks[:,1] = target_img.shape[0] - plt_opt_lmks[:,1]
                for (x, y) in plt_opt_lmks:
                    cv2.circle(target_img, (int(x), int(y)), 4, (255, 0, 0), -1)

                if sys.version_info >= (3, 0):
                    rendered_img = render_mesh(Mesh(scale*verts, faces), height=target_img.shape[0], width=target_img.shape[1])
                    for (x, y) in plt_opt_lmks:
                        cv2.circle(rendered_img, (int(x), int(y)), 4, (255, 0, 0), -1)
                    target_img = np.hstack((target_img, rendered_img))

                cv2.imshow('img', target_img)
                cv2.waitKey(10)
        else:
            def on_step(*_):
                pass

        print('Optimize rigid transformation')
        vars = [tf_scale, tf_trans, tf_rot]
        loss = lmk_dist
        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1, 'ftol': 5e-6})
        optimizer.minimize(session, fetches=[tf_model, tf_scale, tf.constant(smpl.f), tf.constant(target_img), tf.constant(target_2d_lmks), lmks_proj_2d], loss_callback=on_step)

        print('Optimize model parameters')
        vars = [tf_scale, tf_trans[:2], tf_rot, tf_pose, tf_shape, tf_exp]
        loss = lmk_dist + shape_reg + exp_reg + neck_pose_reg + jaw_pose_reg + eyeballs_pose_reg

        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 0, 'ftol': 1e-7})
        optimizer.minimize(session, fetches=[tf_model, tf_scale, tf.constant(smpl.f), tf.constant(target_img), tf.constant(target_2d_lmks), lmks_proj_2d,
                                             lmk_dist, shape_reg, exp_reg, neck_pose_reg, jaw_pose_reg, eyeballs_pose_reg], loss_callback=on_step)

        print('Fitting done')
        np_verts, np_scale = session.run([tf_model, tf_scale])
        return Mesh(np_verts, smpl.f), np_scale


def run_2d_lmk_fitting(model_fname, flame_lmk_path, texture_mapping, target_img_path, target_lmk_path, out_path, visualize):
    if 'generic' not in model_fname:
        print('You are fitting a gender specific model (i.e. female / male). Please make sure you selected the right gender model. Choose the generic model if gender is unknown.')
    if not os.path.exists(flame_lmk_path):
        print('FLAME landmark embedding not found - %s ' % flame_lmk_path)
        return
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return
    if not os.path.exists(target_lmk_path):
        print('Landmarks of target image not found - s' % target_lmk_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lmk_face_idx, lmk_b_coords = load_embedding(flame_lmk_path)

    target_img = cv2.imread(target_img_path)
    lmk_2d = np.load(target_lmk_path)

    weights = {}
    # Weight of the landmark distance term
    weights['lmk'] = 1.0
    # Weight of the shape regularizer
    weights['shape'] = 1e-3
    # Weight of the expression regularizer
    weights['expr'] = 1e-3
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
    weights['neck_pose'] = 100.0
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
    weights['jaw_pose'] = 1e-3
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer
    weights['eyeballs_pose'] = 10.0

    result_mesh, result_scale = fit_lmk2d(target_img, lmk_2d, model_fname, lmk_face_idx, lmk_b_coords, weights, visualize)

    if sys.version_info >= (3, 0):
        texture_data = np.load(texture_mapping, allow_pickle=True, encoding='latin1').item()
    else:
        texture_data = np.load(texture_mapping, allow_pickle=True).item()
    texture_map = compute_texture_map(target_img, result_mesh, result_scale, texture_data)

    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.obj')
    out_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.png')

    cv2.imwrite(out_img_fname, texture_map)
    result_mesh.set_vertex_colors('white')
    result_mesh.vt = texture_data['vt']
    result_mesh.ft = texture_data['ft']
    result_mesh.set_texture_image(out_img_fname)
    result_mesh.write_obj(out_mesh_fname)
    np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)

    if visualize:
        mv = MeshViewers(shape=[1,2], keepalive=True)
        mv[0][0].set_static_meshes([Mesh(result_mesh.v, result_mesh.f)])
        mv[0][1].set_static_meshes([result_mesh])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build texture from image')
    # Path of the Tensorflow FLAME model (generic, female, male gender)
    # Choose the generic model if gender is unknown
    parser.add_argument('--model_fname', default='./models/generic_model.pkl', help='Path of the FLAME model')
    # Path of the landamrk embedding file into the FLAME surface
    parser.add_argument('--flame_lmk_path', default='./data/flame_static_embedding.pkl', help='Path of the landamrk embedding file into the FLAME surface')
    # Pre-computed texture mapping for FLAME topology meshes
    parser.add_argument('--texture_mapping', default='./data/texture_data.npy', help='pre-computed FLAME texture mapping')
    # Target image (used for visualization only)
    parser.add_argument('--target_img_path', default='./data/imgHQ00088.jpeg', help='Path of the target image')
    # 2D landmark file that should be fitted (landmarks must be corresponding with the defined FLAME landmarks)
    # see "img1_lmks_visualized.jpeg" or "see the img2_lmks_visualized.jpeg" for the order of the landmarks
    parser.add_argument('--target_lmk_path', default='./data/imgHQ00088_lmks.npy', help='2D landmark file that should be fitted (landmarks must be corresponding with the defined FLAME landmarks)')
    # Output path
    parser.add_argument('--out_path', default='./results', help='Path of the fitting output')
    # Visualize fitting
    parser.add_argument('--visualize', default='True', help='Visualize fitting progress and final fitting result')

    args = parser.parse_args()

    run_2d_lmk_fitting(args.model_fname, args.flame_lmk_path, args.texture_mapping, args.target_img_path, args.target_lmk_path, args.out_path, str2bool(args.visualize))