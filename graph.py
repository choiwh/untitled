import tensorflow as tf
import matplotlib.pylab as plt
# tf Graph Input
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Try to find values for W and b that compute y_data = W * x_data
# We know that W should be 1
# random_uniform -> random function -10.0 ~ 10.0
W = tf.Variable(tf.random_uniform([1],-10.0,10.0))

# Place Holder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X

# Cost function
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# Minimize
# W = W- alpha(step) * 1/m * Sigma(W*X-Y) * X
# Sigma( W * X - Y) * X -> Derivative

descent = W - tf.mul(0.1 , tf.reduce_mean(tf.mul((tf.mul(W , X) - Y),X)))
update = W.assign(descent) # Operation

# Before Starting, initalize the Variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# For Graphs
W_val = []
cost_val = []

# Fit the line.
for step in xrange(100) :
       sess.run(update, feed_dict={X:x_data, Y:y_data})
       cost_val.append(sess.run(cost,feed_dict={X:x_data, Y:y_data}))
       W_val.append(sess.run(W))
       print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

# Show the Graph
plt.plot(W_val, cost_val)
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()
