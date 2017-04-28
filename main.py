
# Helpers for command parse arguments
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import io

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def log_dir(FLAGS):
    "Prepare Tensorboard log directory"
    if FLAGS.delete_logs:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        print("Cleaning Logdir: %s" % FLAGS.log_dir)
    else:
        print("Logdir: %s" % FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)


def cmd_parser():
    "Parse common argument options"
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tf/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tf/logs',
                        help='Summaries log directory')
    parser.add_argument('--rerror', type=float, default=0.0,
                        help='Minimum Relative Error to Stop.')
    parser.add_argument('--delete_logs', type=bool, default=False,
                        help='Delete TensorFlow board logs before start')

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
