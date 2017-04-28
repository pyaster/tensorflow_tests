
from main import *

import argparse
import sys


def test_placeholders(FLAGS):
    "Simply dump a placeholder to TensorBoard"
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [])
    sess = tf.Session()

    summary = tf.summary.scalar("x", x)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/placeholder',
                                         sess.graph)

    r = sess.run(tf.global_variables_initializer())
    s = sess.run(summary, feed_dict={x: 1.57})
    train_writer.add_summary(s)

    train_writer.close()


def test_merge(FLAGS):
    "A simple function that make a loop computation and write down into TB"
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32)
    k = np.random.random() + 0.1

    # Create a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # define a single summary
    summary_x = tf.summary.scalar("x", x)

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/foo',
                                         sess.graph)

    # write some summaries directly
    for i in range(0, 5):
        summary = sess.run(summary_x, feed_dict={x: k * i * i})
        train_writer.add_summary(summary, i)

    # write some summaries using merge_all
    # (we have only one define summary)
    merged = tf.summary.merge_all()
    for i in range(5, 10):
        summary = sess.run(merged, feed_dict={x: k * i * i})
        train_writer.add_summary(summary, i)

    train_writer.close()


def test_cyclic_flows(FLAGS):
    """Feed the model for 1st time and let it
    evolve in a closed iteration fashion"""
    tf.reset_default_graph()

    # Define a placeforlder to feed the graph
    x0 = tf.placeholder(tf.float64)

    # Build the model
    x = tf.Variable(0, dtype=tf.float64)
    y = tf.Variable(tf.zeros_like(x))
    x_ = x + x
    y_ = x * x

    step = [x.assign(x_), y.assign(y_)]

    # define the initialization step
    tf.summary.scalar("x", x)
    tf.summary.scalar("y", y)
    summary_y = tf.summary.scalar("y", y)

    # Create a session
    sess = tf.Session()

    # setup the graph and initial conditions
    init = x.assign(x0)

    # init requires a value to set 'x' variable
    # that is done by 'x0' placeholder for the 1st time
    sess.run(init, feed_dict={x0: 1.05})

    # setup summaries and TB log
    merged = tf.summary.merge_all()  # collect all sumaries in a single one
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test_cyclic_flows',
                                         sess.graph)

    # perform the computation
    for i in range(10):
        r = sess.run(step)
        print(r)

        summary = sess.run(summary_y)
        train_writer.add_summary(summary, i)

    train_writer.close()

    foo = 1


if __name__ == '__main__':
    FLAGS, unparsed = cmd_parser()
    log_dir(FLAGS)

    test_placeholders(FLAGS)
    test_merge(FLAGS)
    test_cyclic_flows(FLAGS)
