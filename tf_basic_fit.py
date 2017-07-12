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


def test_fit_theta(FLAGS):
    "Basic Linear Regressions with implicit named variables (W, b)"
    tf.reset_default_graph()

    # Create the input data
    Theta_r = tf.constant([1.2, 0.25],  name="Theta_r", shape=(1, 2))
    X = tf.random_uniform([1000], 0.0, 10)
    X = tf.reshape(X, [1, -1])
    bias = tf.ones_like(X)

    X = tf.concat([bias, X], 0)
    X = tf.Variable(X, trainable=False, name='X')

    Y_r = tf.matmul(Theta_r, X)

    # Inference Graph
    with tf.name_scope('inference'):
        # Create the model
        Theta = tf.Variable(tf.random_uniform([1, 2], -0.1, 0.1), name='Theta')
        Y = tf.matmul(Theta, X)
        variable_summaries(Theta, 'Theta')

    # Training Graph
    with tf.name_scope('train'):
        # training
        v_loss = tf.square(Y - Y_r)
        loss = tf.reduce_mean(v_loss)

        # optimizer = tf.train.GradientDescentOptimizer(0.005)
        # optimizer = tf.train.AdamOptimizer()
        optimizer = tf.train.MomentumOptimizer(0.01, 0.01)
        # optimizer = tf.contrib.keras.optimizers.Nadam()


        step = optimizer.minimize(loss)

        tf.summary.scalar('loss', loss)

    # Create a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/vectorialFit',
                                         sess.graph)

    merged = tf.summary.merge_all()
    err0 = 10**6
    for i in xrange(10000):
        sess.run(step)

        if not i % 50:
            summary = sess.run(merged)
            train_writer.add_summary(summary, i)

            print('-'*40)
            print("Iteration: %d" % i)
            print("Theta : \t%s" % sess.run(Theta))
            print("Theta*: \t%s" % sess.run(Theta_r))

            err = sess.run(loss)
            print("Error:\t\t%s" % err)

            r = (err0 - err) / err0
            print("Rel.Error:\t%s" % r)
            if err < 1e-09 or r < 0.001:
                break
            err0 = err

    train_writer.close()

    foo = 1

if __name__ == '__main__':
    FLAGS, unparsed = cmd_parser()
    log_dir(FLAGS)

    test_fit_theta(FLAGS)
    test_fit(FLAGS)



