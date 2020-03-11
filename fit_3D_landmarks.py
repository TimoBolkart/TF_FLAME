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
import six
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.landmarks import load_embedding, tf_get_model_lmks, create_lmk_spheres

from tf_smpl.batch_smpl import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

def fit_lmk3d(target_3d_lmks, template_fname, model_fname, lmk_face_idx, lmk_b_coords, weights, show_fitting=True):
    '''
    Fit FLAME to 3D landmarks
    :param target_3d_lmks:      target 3D landmarks provided as (num_lmks x 3) matrix
    :param template_fname:      template mesh in FLAME topology (only the face information are used)
    :param model_fname:      saved Tensorflow FLAME model
    :param lmk_face_idx:        face indices of the landmark embedding in the FLAME topology
    :param lmk_b_coords:        barycentric coordinates of the landmark embedding in the FLAME topology
                                (i.e. weighting of the three vertices for the trinagle, the landmark is embedded in
    :param weights:             weights of the individual objective functions
    :return: a mesh with the fitting results
    '''

    template_mesh = Mesh(filename=template_fname)

    tf_trans = tf.Variable(np.zeros((1,3)), name="trans", dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1,3)), name="pose", dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1,12)), name="pose", dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1,300)), name="shape", dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1,100)), name="expression", dtype=tf.float64, trainable=True)
    smpl = SMPL(model_fname)
    tf_model = tf.squeeze(smpl(tf_trans,
                               tf.concat((tf_shape, tf_exp), axis=-1),
                               tf.concat((tf_rot, tf_pose), axis=-1)))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        lmks = tf_get_model_lmks(tf_model, template_mesh, lmk_face_idx, lmk_b_coords)
        lmk_dist = tf.reduce_sum(tf.square(1000 * tf.subtract(lmks, target_3d_lmks)))
        neck_pose_reg = tf.reduce_sum(tf.square(tf_pose[:3]))
        jaw_pose_reg = tf.reduce_sum(tf.square(tf_pose[3:6]))
        eyeballs_pose_reg = tf.reduce_sum(tf.square(tf_pose[6:]))
        shape_reg = tf.reduce_sum(tf.square(tf_shape))
        exp_reg = tf.reduce_sum(tf.square(tf_exp))

        # Optimize global transformation first
        vars = [tf_trans, tf_rot]
        loss = weights['lmk'] * lmk_dist
        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1, 'ftol': 5e-6})
        print('Optimize rigid transformation')
        optimizer.minimize(session)

        # Optimize for the model parameters
        vars = [tf_trans, tf_rot, tf_pose, tf_shape, tf_exp]
        loss = weights['lmk'] * lmk_dist + weights['shape'] * shape_reg + weights['expr'] * exp_reg + \
               weights['neck_pose'] * neck_pose_reg + weights['jaw_pose'] * jaw_pose_reg + weights['eyeballs_pose'] * eyeballs_pose_reg

        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1, 'ftol': 5e-6})
        print('Optimize model parameters')
        optimizer.minimize(session)

        print('Fitting done')

        if show_fitting:
            # Visualize landmark fitting
            mv = MeshViewer()
            mv.set_static_meshes(create_lmk_spheres(target_3d_lmks, 0.001, [255.0, 0.0, 0.0]))
            mv.set_dynamic_meshes([Mesh(session.run(tf_model), template_mesh.f)] + create_lmk_spheres(session.run(lmks), 0.001, [0.0, 0.0, 255.0]), blocking=True)
            six.moves.input('Press key to continue')

        return Mesh(session.run(tf_model), template_mesh.f)


def run_3d_lmk_fitting():
    # Path of the Tensorflow FLAME model
    model_fname = './models/generic_model.pkl'
    # model_fname = './models/female_model.pkl'
    # model_fname = './models/male_model.pkl'

    # Path of a template mesh in FLAME topology
    template_fname = './data/template.ply'

    # Path of the landamrk embedding file into the FLAME surface
    flame_lmk_path = './data/flame_static_embedding.pkl'
    # 3D landmark file that should be fitted (landmarks must be corresponding with the defined FLAME landmarks)
    # see "img1_lmks_visualized.jpeg" or "see the img2_lmks_visualized.jpeg" for the order of the landmarks
    target_lmk_path = './data/landmark_3d.npy'

    # Output filename
    out_mesh_fname = './results/landmark_3d.ply'

    lmk_face_idx, lmk_b_coords = load_embedding(flame_lmk_path)
    lmk_3d = np.load(target_lmk_path)

    weights = {}
    # Weight of the landmark distance term
    weights['lmk'] = 1.0
    # Weight of the shape regularizer
    weights['shape'] = 1.0
    # Weight of the expression regularizer
    weights['expr']  = 1.0
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
    weights['neck_pose'] = 100.0
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
    weights['jaw_pose'] = 1.0
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer
    weights['eyeballs_pose'] = 10.0
    # Show landmark fitting (default: red = target landmarks, blue = fitting landmarks)
    show_fitting = True

    result_mesh = fit_lmk3d(lmk_3d, template_fname, model_fname, lmk_face_idx, lmk_b_coords, weights, show_fitting=show_fitting)

    if not os.path.exists(os.path.dirname(out_mesh_fname)):
        os.makedirs(os.path.dirname(out_mesh_fname))

    result_mesh.write_ply(out_mesh_fname)


if __name__ == '__main__':
    run_3d_lmk_fitting()