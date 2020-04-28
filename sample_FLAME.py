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
import argparse
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.landmarks import load_binary_pickle, load_embedding, tf_get_model_lmks, create_lmk_spheres

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

def sample_FLAME(model_fname, num_samples, out_path, visualize, sample_VOCA_template=False):
    '''
    Sample the FLAME model to demonstrate how to vary the model parameters.FLAME has parameters to
        - model identity-dependent shape variations (paramters: shape),
        - articulation of neck (paramters: pose[0:3]), jaw (paramters: pose[3:6]), and eyeballs (paramters: pose[6:12])
        - model facial expressions, i.e. all expression motion that does not involve opening the mouth (paramters: exp)
        - global translation (paramters: trans)
        - global rotation (paramters: rot)
    :param model_fname              saved FLAME model
    :param num_samples              number of samples
    :param out_path                 output path to save the generated templates (no templates are saved if path is empty)
    :param visualize                visualize samples
    :param sample_VOCA_template     sample template in 'zero pose' that can be used e.g. for speech-driven animation in VOCA
    '''

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

        if visualize:
            mv = MeshViewer()
        for i in range(num_samples):
            if sample_VOCA_template:
                assign_shape = tf.assign(tf_shape, np.hstack((np.random.randn(100), np.zeros(200)))[np.newaxis,:])
                session.run([assign_shape])
                out_fname = os.path.join(out_path, 'VOCA_template_%02d.ply' % (i+1))
            else:
                # assign_trans = tf.assign(tf_trans, np.random.randn(3)[np.newaxis,:])
                assign_rot = tf.assign(tf_rot, np.random.randn(3)[np.newaxis,:] * 0.03)
                assign_pose = tf.assign(tf_pose, np.random.randn(12)[np.newaxis,:] * 0.02)
                assign_shape = tf.assign(tf_shape, np.hstack((np.random.randn(100), np.zeros(200)))[np.newaxis,:])
                assign_exp = tf.assign(tf_exp, np.hstack((0.5*np.random.randn(50), np.zeros(50)))[np.newaxis,:])
                session.run([assign_rot, assign_pose, assign_shape, assign_exp])
                out_fname = os.path.join(out_path, 'FLAME_sample_%02d.ply' % (i+1))

            sample_mesh = Mesh(session.run(tf_model), smpl.f)
            if visualize:
                mv.set_dynamic_meshes([sample_mesh], blocking=True)
                key = six.moves.input('Press (s) to save sample, any other key to continue ')
                if key == 's':
                    sample_mesh.write_ply(out_fname)
            else:
                sample_mesh.write_ply(out_fname)

def main(args):
    if not os.path.exists(args.model_fname):
        print('FLAME model not found - %s' % args.model_fname)
        return
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if args.option == 'sample_FLAME':
        sample_FLAME(args.model_fname, int(args.num_samples), args.out_path, str2bool(args.visualize), sample_VOCA_template=False)
    else:
        sample_FLAME(args.model_fname, int(args.num_samples), args.out_path, str2bool(args.visualize), sample_VOCA_template=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample FLAME shape space')
    parser.add_argument('--option', default='sample_FLAME', help='sample random FLAME meshes or VOCA templates')
    parser.add_argument('--model_fname', default='./models/generic_model.pkl', help='Path of the FLAME model')
    parser.add_argument('--num_samples', default='5', help='Number of samples')
    parser.add_argument('--out_path', default='./FLAME_samples', help='Output path')
    parser.add_argument('--visualize', default='True', help='Visualize fitting progress and final fitting result')
    args = parser.parse_args()
    main(args)

