import numpy as np
import torch
from .gain import gain
from typing import Union, Iterable

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm
from .utils import xavier_init, binary_sampler, uniform_sampler, normalization, renormalization, rounding

class GAIN:
    def __init__(self, dim, alpha=100, hint_rate=0.9, session=None):
        """
        Initialize the GAIN model.

        Args:
            dim: int, number of features
            alpha: float, hyperparameter for generator loss
            hint_rate: float, probability of revealing hint
            session: optional TF session (creates one if None)
        """
        self.dim = dim
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.sess = session or tf.Session()

        # Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, dim])
        self.M = tf.placeholder(tf.float32, shape=[None, dim])
        self.H = tf.placeholder(tf.float32, shape=[None, dim])

        # Generator variables
        self.G_W1 = tf.Variable(xavier_init([dim*2, dim]))
        self.G_b1 = tf.Variable(tf.zeros([dim]))
        self.G_W2 = tf.Variable(xavier_init([dim, dim]))
        self.G_b2 = tf.Variable(tf.zeros([dim]))
        self.G_W3 = tf.Variable(xavier_init([dim, dim]))
        self.G_b3 = tf.Variable(tf.zeros([dim]))
        self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

        # Discriminator variables
        self.D_W1 = tf.Variable(xavier_init([dim*2, dim]))
        self.D_b1 = tf.Variable(tf.zeros([dim]))
        self.D_W2 = tf.Variable(xavier_init([dim, dim]))
        self.D_b2 = tf.Variable(tf.zeros([dim]))
        self.D_W3 = tf.Variable(xavier_init([dim, dim]))
        self.D_b3 = tf.Variable(tf.zeros([dim]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # Build graph
        self._build_graph()

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

    def _generator(self, x, m):
        inputs = tf.concat([x, m], axis=1)
        h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.G_W2) + self.G_b2)
        out = tf.nn.sigmoid(tf.matmul(h2, self.G_W3) + self.G_b3)
        return out

    def _discriminator(self, x, h):
        inputs = tf.concat([x, h], axis=1)
        h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.D_W2) + self.D_b2)
        logits = tf.matmul(h2, self.D_W3) + self.D_b3
        return tf.nn.sigmoid(logits)

    def _build_graph(self):
        self.G_sample = self._generator(self.X, self.M)
        Hat_X = self.X * self.M + self.G_sample * (1 - self.M)
        D_prob = self._discriminator(Hat_X, self.H)

        # Losses
        self.D_loss = -tf.reduce_mean(self.M * tf.log(D_prob + 1e-8) + (1 - self.M) * tf.log(1 - D_prob + 1e-8))
        G_loss_temp = -tf.reduce_mean((1 - self.M) * tf.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((self.M * self.X - self.M * self.G_sample) ** 2) / tf.reduce_mean(self.M)
        self.G_loss = G_loss_temp + self.alpha * MSE_loss
        self.MSE_loss = MSE_loss

        # Solvers
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)

    def train_batch(self, X_batch, M_batch):
        """
        Train one batch.
        Args:
            X_batch: np.array, batch of normalized data with missing entries filled as zeros
            M_batch: np.array, corresponding mask (1=observed, 0=missing)
        """
        Z_mb = uniform_sampler(0, 0.01, X_batch.shape[0], self.dim)
        H_mb_temp = binary_sampler(self.hint_rate, X_batch.shape[0], self.dim)
        H_mb = M_batch * H_mb_temp
        X_input = M_batch * X_batch + (1 - M_batch) * Z_mb

        # Update discriminator
        self.sess.run([self.D_solver, self.D_loss], feed_dict={self.X: X_input, self.M: M_batch, self.H: H_mb})
        # Update generator
        self.sess.run([self.G_solver, self.G_loss, self.MSE_loss], feed_dict={self.X: X_input, self.M: M_batch, self.H: H_mb})

    def impute(self, X_full, M_full):
        """
        Impute missing values.
        Args:
            X_full: np.array, normalized data
            M_full: np.array, mask
        Returns:
            imputed: np.array
        """
        Z_mb = uniform_sampler(0, 0.01, X_full.shape[0], self.dim)
        X_input = M_full * X_full + (1 - M_full) * Z_mb
        imputed = self.sess.run(self.G_sample, feed_dict={self.X: X_input, self.M: M_full})
        imputed = M_full * X_full + (1 - M_full) * imputed
        return imputed
