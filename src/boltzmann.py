import os
import sys
import json
import numpy as np
import tensorflow as tf


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


class RBM:
    """
    The base Restricted Boltzmann Machine implementation
    ---
    Attributes
    -----------
    n_visible: int
        dimensionality of visible layer (load config)
    n_hidden: int
        dimensionality of hidden layer (load config)
    learning_rate: int
        learning rate of the RBM
    momentum: int
        momentum of the RBM
    x/y: tf.placeholder
        visible/hidden layer
    w: tf.Variable
        weights between visible and hidden layers
    b_visible: tf.Variable
        bias for visible layer
    b_hidden: tf.Variable
        bias for hidden layer
    delta_w: tf.Variable
        regularization term for weights
    delta_b_visible: tf.Variable
        regularization term for visible layer
    delta_b_hidden: tf.Variable
        regularization term for hidden layer
    batch_size: int
        batch size during training (load config)
    epochs: int
        epochs during training (load config)
    save_dir
        directory for model saving (load config)
    """
    def __init__(self,
                learning_rate=0.01,
                momentum=0.95,
                xavier_const=1.0,
                err_function='mse',
                use_tqdm=False):
        # check value errors
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0,1]')
        if err_function not in ['mse', 'cosine']:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')
        if use_tqdm:
            from tqdm import tqdm
            self._use_tqdm = use_tqdm
            self._tqdm = tqdm
        
        self.config = json.load(open('./config/config.json', 'r'))
        self.save_dir = self.config['boltzmann']['save_dir']
        self.n_visible = self.config['boltzmann']['visible_dim']
        self.n_hidden = self.config['boltzmann']['hidden_dim']
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.b_visible = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.b_hidden = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_b_visible = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_b_hidden = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        """initialized by BBRBM or GBRBM classes
        """
        pass
    
    def get_free_energy(self):
        pass
    
    def get_err(self, batch_x):
        """compute the error given the visible layer
        """
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})
    
    def transform(self, batch_x):
        """compute the hidden layer activation probabilities P(h=1|v=X)
        """
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        """compute the visible layer activation probabilities P(v=1|h=h)
        """
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})
    
    def reconstruct(self, batch_x):
        """reconstruct the visible layer given the original visible layer
        """
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self, data, shuffle=True, verbose=True):
        """fit the model to provided data
        # para data: training data to learn representations
        """
        n_data, errs = data.shape[0], []
        batch_size = self.config['boltzmann']['batch_size']
        n_epochs = self.config['boltzmann']['epochs']
        # check batch size
        if batch_size > 0:
            n_batch = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batch = 1
        # check shuffle 
        if shuffle:
            data_copy = data.copy()
            inds = np.arange(n_data)
        else:
            data_copy = data

        # begin fitting
        for e in range(n_epochs):
            epoch_errs = np.zeros((n_batch,))
            epoch_errs_ptr = 0
            # shuffle training data
            if shuffle:
                np.random.shuffle(inds)
                data_copy = data_copy[inds]
            
            r_batch = range(n_batch)
            if verbose and self._use_tqdm:
                r_batch = self._tqdm(r_batch, desc='Epoch: %d' % e, ascii=True, file=sys.stdout)
            # compute errors for one batch
            for b in r_batch:
                batch_x = data_copy[b*batch_size:(b+1)*batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1
            
            errs = np.hstack([errs, epoch_errs])
            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: %f' % err_mean)
                    self._tqdm.write('')
                else:
                    print('Train error: %f' % err_mean)
                    print('')
                sys.stdout.flush()

        return errs

    def get_weights(self):
        return self.sess.run(self.w), self.sess.run(self.b_visible), self.sess.run(self.b_hidden)
    
    def set_weights(self, w, b_visible, b_hidden):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.b_visible.assign(b_visible))
        self.sess.run(self.b_hidden.assign(b_hidden))

    def save_weights(self, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.b_visible,
                                name + '_h': self.b_hidden})
        return saver.save(self.sess, os.path.join(self.save_dir, name, name))

    def load_weights(self, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.b_visible,
                                name + '_h': self.b_hidden})
        saver.restore(self.sess, os.path.join(self.save_dir, name, name))


class BBRBM(RBM):
    """
    Bernoulli-Bernoulli RBM
    """
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b_hidden)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.b_visible)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.b_hidden)

        pos_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        neg_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old + self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])
        
        delta_w_new = f(self.delta_w, pos_grad - neg_grad)
        delta_b_visible_new = f(self.delta_b_visible, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_b_hidden_new = f(self.delta_b_hidden, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_b_visible = self.delta_b_visible.assign(delta_b_visible_new)
        update_delta_b_hidden = self.delta_b_hidden.assign(delta_b_hidden_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_b_visible = self.b_visible.assign(self.b_visible + delta_b_visible_new)
        update_b_hidden = self.b_hidden.assign(self.b_hidden + delta_b_hidden_new)

        self.update_deltas = [update_delta_w, update_delta_b_visible, update_delta_b_hidden]
        self.update_weights = [update_w, update_b_visible, update_b_hidden]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b_hidden)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.b_visible)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.b_visible)


class GBRBM(RBM):
    """
    Gaussian-Bernoulli RBM
    """
    def __init__(self, sample_visible=False, sigma=1, **kwargs):
        self.sample_visible = sample_visible
        self.sigma = sigma
        RBM.__init__(self, **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b_hidden)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.b_visible
        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.b_hidden)

        pos_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        neg_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)
        
        def f(x_old, x_new):
            return self.momentum * x_old + self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new = f(self.delta_w, pos_grad - neg_grad)
        delta_b_visible_new = f(self.delta_b_visible, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_b_hidden_new = f(self.delta_b_hidden, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_b_visible = self.delta_b_visible.assign(delta_b_visible_new)
        update_delta_b_hidden = self.delta_b_hidden.assign(delta_b_hidden_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_b_visible = self.b_visible.assign(self.b_visible + delta_b_visible_new)
        update_b_hidden = self.b_hidden.assign(self.b_hidden + delta_b_hidden_new)

        self.update_deltas = [update_delta_w, update_delta_b_visible, update_delta_b_hidden]
        self.update_weights = [update_w, update_b_visible, update_b_hidden]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b_hidden)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.b_visible)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.b_visible)