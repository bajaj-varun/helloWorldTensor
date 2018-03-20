import tensorflow as tf

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)

liner_model = W*x+b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Tensorboard
writer = tf.summary.FileWriter("./log", sess.graph)

print(sess.run(liner_model, {x:[1.0,2.0,3.0,4.0]}))