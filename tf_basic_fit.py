from main import *


def test_fit(FLAGS):
    "Basic Linear Regressions with implicit named variables (W, b)"
    tf.reset_default_graph()

    W_r = 0.25
    b_r = 1.2

    x_data = np.random.rand(100).astype(np.float32)
    # noise = np.random.normal(scale=0.01, size=len(x_data))
    noise = 0
    y_data = x_data * W_r + b_r + noise

    # Inference Graph
    with tf.name_scope('inference'):
        W = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name='W')
        b = tf.Variable(tf.zeros([1]), name='b')
        y = W * x_data + b

        variable_summaries(W, 'W')
        variable_summaries(b, 'b')

    # Training Graph
    with tf.name_scope('train'):
        v_loss = tf.square(y - y_data)
        loss = tf.reduce_mean(v_loss)

        optimizer = tf.train.GradientDescentOptimizer(0.5)
        step = optimizer.minimize(loss)

        tf.summary.scalar('loss', loss)

    # Create a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/fit',
                                         sess.graph)

    merged = tf.summary.merge_all()
    for i in range(30):
        _, err = sess.run([step, loss])
        print(err)
        summary = sess.run(merged)
        train_writer.add_summary(summary, i)

    train_writer.close()

    foo = 1

if __name__ == '__main__':
    FLAGS, unparsed = cmd_parser()
    log_dir(FLAGS)

    test_fit(FLAGS)


