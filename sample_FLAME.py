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
import argparse
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.landmarks import load_binary_pickle, load_embedding, tf_get_model_lmks, create_lmk_spheres
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

def sample_FLAME(template_fname, tf_model_fname, num_samples):
    '''
    Sample the FLAME model to demonstrate how to vary the model parameters.FLAME has parameters to
        - model identity-dependent shape variations (paramters: shape),
        - articulation of neck (paramters: pose[0:3]), jaw (paramters: pose[3:6]), and eyeballs (paramters: pose[6:12])
        - model facial expressions, i.e. all expression motion that does not involve opening the mouth (paramters: exp)
        - global translation (paramters: trans)
        - global rotation (paramters: rot)
    :param template_fname:      template mesh in FLAME topology (only the face information are used)
    :param tf_model_fname:      saved Tensorflow FLAME model
    '''

    template_mesh = Mesh(filename=template_fname)
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')

    graph = tf.get_default_graph()
    tf_model = graph.get_tensor_by_name(u'vertices:0')

    with tf.Session() as session:
        saver.restore(session, tf_model_fname)

        # Workaround as existing tf.Variable cannot be retrieved back with tf.get_variable
        tf_trans = [x for x in tf.trainable_variables() if 'trans' in x.name][0]
        tf_rot = [x for x in tf.trainable_variables() if 'rot' in x.name][0]
        tf_pose = [x for x in tf.trainable_variables() if 'pose' in x.name][0]
        tf_shape = [x for x in tf.trainable_variables() if 'shape' in x.name][0]
        tf_exp = [x for x in tf.trainable_variables() if 'exp' in x.name][0]

        mv = MeshViewer()

        for i in range(num_samples):
            assign_trans = tf.assign(tf_trans, np.random.randn(3))
            assign_rot = tf.assign(tf_rot, np.random.randn(3) * 0.03)
            assign_pose = tf.assign(tf_pose, np.random.randn(12) * 0.03)
            assign_shape = tf.assign(tf_shape, np.random.randn(300) * 1.0)
            assign_exp = tf.assign(tf_exp, np.random.randn(100) * 0.5)
            session.run([assign_trans, assign_rot, assign_pose, assign_shape, assign_exp])

            mv.set_dynamic_meshes([Mesh(session.run(tf_model), template_mesh.f)], blocking=True)
            raw_input('Press key to continue')

def sample_VOCA_template(template_fname, tf_model_fname, out_mesh_fname):
    '''
    VOCA animates static templates in FLAME topology. Such templates can be obtained by sampling the FLAME shape space.
    This function randomly samples the FLAME identity shape space to generate new templates.
    :param template_fname:  template mesh in FLAME topology (only the face information are used)
    :param tf_model_fname:  saved Tensorflow FLAME model
    :param out_mesh_fname:  filename of the VOCA template
    :return:
    '''

    template_mesh = Mesh(filename=template_fname)
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')

    graph = tf.get_default_graph()
    tf_model = graph.get_tensor_by_name(u'vertices:0')

    with tf.Session() as session:
        saver.restore(session, tf_model_fname)

        # Workaround as existing tf.Variable cannot be retrieved back with tf.get_variable
        tf_shape = [x for x in tf.trainable_variables() if 'shape' in x.name][0]

        assign_shape = tf.assign(tf_shape, np.hstack((np.random.randn(100), np.zeros(200))))
        session.run([assign_shape])

        Mesh(session.run(tf_model), template_mesh.f).write_ply(out_mesh_fname)


def draw_random_samples():
    # Path of the Tensorflow FLAME model
    tf_model_fname = './models/tf_generic_model'
    # Path of a tempalte mesh in FLAME topology
    template_fname = './data/template.ply'
    # Number of samples
    num_samples = 10

    sample_FLAME(template_fname, tf_model_fname, num_samples)

def draw_VOCA_template_sample():
    # Path of the Tensorflow FLAME model
    tf_model_fname = './models/tf_generic_model'
    # Path of a tempalte mesh in FLAME topology
    template_fname = './data/template.ply'
    # Output mesh path
    out_mesh_fname = './FLAME_samples/voca_template.ply'

    if not os.path.exists(os.path.dirname(out_mesh_fname)):
        os.makedirs(os.path.dirname(out_mesh_fname))

    sample_VOCA_template(template_fname, tf_model_fname, out_mesh_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice operated character animation')
    parser.add_argument('--option', default='random_sample', help='sample random FLAME meshes or VOCA templates')

    args = parser.parse_args()
    option = args.option
    if option == 'sample_VOCA_template':
        draw_VOCA_template_sample()
    else:
        draw_random_samples()
