import numpy as np
import tensorflow as tf


class RBM(object):
    def __init__(self, visible_dim, hidden_dim, learning_rate, no_iterations):
        self._graph = tf.Graph()

        # initialize the graph
        with self._graph.as_default():
            self.no_iter = no_iterations
            self._visible_bias = tf.Variable(tf.random_uniform([1, visible_dim], 0, 1), name='visible_bias')
            self._hidden_bias = tf.Variable(tf.random_uniform([1, hidden_dim], 0, 1), name='hidden_bias')
            self._hidden_states = tf.Variable(tf.zeros([1, hidden_dim], tf.float32), name='hidden_states')
            self._visible_cdstates = tf.Variable(tf.zeros([1, visible_dim], tf.float32), name='visible_cdstates')
            self._hidden_cdstates = tf.Variable(tf.zeros([1, hidden_dim], tf.float32), name='hidden_cdstates')
            self._weights = tf.Variable(tf.random_normal([visible_dim, hidden_dim], 0.01), name='weights')
            self._learning_rate = tf.Variable(tf.fill([visible_dim, hidden_dim], learning_rate), name='learning_rate')

            self._input_sample = tf.placeholder(tf.float32, [visible_dim], name='input_sample')

            # Gibbs sampling
            input_matrix = tf.transpose(tf.stack([self._input_sample for i in range(hidden_dim)]))
            _hidden_prob = tf.sigmoid(tf.add(tf.multiply(input_matrix, self._weights), tf.stack([self._hidden_bias[0] for i in range(visible_dim)])))
            self._hidden_states = self.calculate_state(_hidden_prob)
            _visible_prob = tf.sigmoid(tf.add(tf.multiply(self._hidden_states, self._weights), tf.transpose(tf.stack([self._visible_bias[0] for i in range(hidden_dim)]))))
            self._visible_cdstates = self.calculate_state(_visible_prob)
            self._hidden_cdstates = self.calculate_state(tf.sigmoid(tf.multiply(self._visible_cdstates, self._weights) + self._hidden_bias))

            # Contrastive Divergence
            pos_gradient_matrix = tf.multiply(input_matrix, self._hidden_states)
            neg_gradient_matrix = tf.multiply(self._visible_cdstates, self._hidden_cdstates)

            new_weights = self._weights
            new_weights.assign_add(tf.multiply(pos_gradient_matrix, self._learning_rate))
            new_weights.assign_sub(tf.multiply(neg_gradient_matrix, self._learning_rate))
            
            self._training = tf.assign(self._weights, new_weights)

            # initialize and run the session
            self._sess = tf.Session()
            initialization = tf.global_variables_initializer()
            self._sess.run(initialization)

    def train(self, input_vecs):
        for iter_no in range(self.no_iter):
            for input_vec in input_vecs:
                self._sess.run(self._training, feed_dict={self._input_sample: input_vec})
        
    def calculate_state(self, prob):
        return tf.floor(prob + tf.random_uniform(tf.shape(prob), 0, 1))


