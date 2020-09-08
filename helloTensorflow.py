import tensorflow as tf

hello = tf.constant('hello, tensorflow!')

sess = tf.Session()

a = tf.constant(2016)
b = tf.constant(10)

print(sess.run(a+b))
