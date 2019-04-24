import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest

# from src.utility import load_cuave

# mfccs, audio, frames_1, _, labels = load_cuave()
# use_audio = True
# if use_audio:
#         audio = np.reshape(audio, (audio.shape[0], int(audio.shape[1]*audio.shape[2])))
#         labels = labels[:,0]
#         print(audio.shape, labels.shape)
#         X_train, X_test, y_train, y_test = train_test_split(audio, labels, test_size=0.25, random_state=1)
#         num_features = 534*4
# else:
#         frames_1 = np.reshape(frames_1, (frames_1.shape[0], int(frames_1.shape[1]*frames_1.shape[2]*frames_1.shape[3])))
#         labels = labels[:,0]
#         print(frames_1.shape, labels.shape)
#         X_train, X_test, y_train, y_test = train_test_split(frames_1, labels, test_size=0.25, random_state=1)
#         num_features = 75*50*4

def tree_models(X, y, num_feat, num_class):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    num_features = num_feat
    num_steps = 400
    num_classes = num_class
    num_trees = 10
    max_nodes = 1000

    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.int64, shape=[None])

    hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()

    forest_graph = tensor_forest.RandomForestGraphs(hparams)

    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    infer_op, _, _, = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

    rf_sess = tf.Session()

    rf_sess.run(init_vars)

    for i in range(1, num_steps + 1):
        _, l = rf_sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
        if i % 50 == 0 or i == 1:
            acc = rf_sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
            print("Step %i, Loss: %f, Acc: %f" % (i, l, acc))

    print("Test Accuracy:", rf_sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))