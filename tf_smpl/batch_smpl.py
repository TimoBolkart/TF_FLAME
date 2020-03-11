""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pickle

import tensorflow as tf
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float64):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            dd = pickle.load(f, encoding="latin1")
        self.dtype = dtype
        # Mean template vertices
        self.v_template = tf.Variable(undo_chumpy(dd['v_template']),
                                      name='v_template',
                                      dtype=self.dtype,
                                      trainable=False)

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        self.num_verts = dd['shapedirs']
        self.num_joints = dd['J'].shape[0]

        # Shape blend shape basis: num_verts x 3 x num_betas
        # reshaped to 3*num_verts x num_betas, transposed to num_betas x 3*num_verts
        shapedir = np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(shapedir, name='shapedirs', dtype=self.dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = tf.Variable(dd['J_regressor'].T.todense(),
                                       name="J_regressor",
                                       dtype=self.dtype,
                                       trainable=False)

        # Pose blend shape basis: num_verts x 3 x 9*num_joints, reshaped to 3*num_verts x 9*num_joints
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(posedirs, name='posedirs', dtype=self.dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(undo_chumpy(dd['weights']),
                                   name='lbs_weights',
                                   dtype=self.dtype,
                                   trainable=False)

    def __call__(self, trans, beta, theta, name=''):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x num_betas
          theta: N x 3*num_joints (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x num_joints x 3 joint location after shaping
                 & posing with beta and theta

        Returns:
          - Verts: N x num_verts x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta]):
            num_batch = beta.shape[0].value

            # 1. Add shape blend shapes
            # (N x num_betas) x (num_betas x 3*num_verts) = N x num_verts x 3
            v_shaped = tf.reshape(tf.matmul(beta, self.shapedirs, name='shape_bs'),
                                  [-1, self.size[0], self.size[1]]) + self.v_template

            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x num_joints x 3 x 3
            Rs = tf.reshape(batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, self.num_joints, 3, 3])
            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3, dtype=self.dtype), [-1, 9*(self.num_joints-1)])

            # (N x 9*(num_joints-1))) x (9*(num_joints-1), 3*num_verts) -> N x num_verts x 3
            v_posed = tf.reshape(tf.matmul(pose_feature, self.posedirs),
                                 [-1, self.size[0], self.size[1]]) + v_shaped

            #4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

            # 5. Do skinning:
            # W is N x num_verts x num_joints
            W = tf.reshape(tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, self.num_joints])

            # (N x num_verts x num_joints) x (N x num_joints x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, self.num_joints, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat([v_posed, tf.ones([num_batch, v_posed.shape[1], 1], dtype=self.dtype)], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            return tf.add(v_homo[:, :, :3, 0], trans)
