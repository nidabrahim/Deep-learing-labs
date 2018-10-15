import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

logs_path = './tensorflow_logs'


X=[[0,0],[0,1],[1,0],[1,1]] 
Y = [[0] ,[1] ,[1] ,[0]]

x_ = tf.placeholder(tf.float32 , shape=[None,2]) 
y_ = tf.placeholder(tf.float32 , shape=[None,1])


# nombre de neurones cachés
hidden_units = 3
with tf.name_scope("Model") as scope:
	# matrice des poids et biais de la première couche
	b1 = tf . Variable ( tf . zeros ([hidden_units]))
	W1= tf.Variable(tf.random_uniform([2,hidden_units], -1.0, 1.0)) 
	#activation non linéaire de la couche cachée
	O=tf.nn.sigmoid(tf.matmul(x_,W1)+b1)
	# matrice des poids et biais de la sedonde couche
	W2= tf.Variable(tf.random_uniform([hidden_units,1], -1.0, 1.0))
	b2 = tf . Variable ( tf . zeros ([1]))
	#sortie du réseau
	y = tf.nn.sigmoid(tf.matmul(O, W2) + b2)



with tf.name_scope("cost") as scope:
	# Fonction de perte quadratique
#	cost = tf .reduce_sum(tf .square(y_ - y), reduction_indices=[0])
	cost = tf .reduce_sum(tf .square(y_ - y))

with tf.name_scope("optim") as scope:
	# Optimisation par descente de gradient avec un learning rate de 0.1
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.name_scope('Precision'):
	correct_prediction = abs(y_ - y) < 0.5
	cast = tf.cast(correct_prediction , "float") 
	precision = tf .reduce_mean(cast)

# Initialisation des variables
init = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)
tf.summary.scalar("precision", precision)

# Suivi des variables du modèle
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()


num_iter = 2000 



# Création d'une session TF pour exécuter le programme
with tf.Session() as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	#nombre d'iterations 
	for i in range(num_iter):
		_, c, summary = sess.run([train_step, cost, merged_summary_op],feed_dict={x_: X, y_: Y})  
		summary_writer.add_summary(summary, i)
		if i % 100 == 0:
			loss, acc = sess.run([cost, precision], feed_dict={x_: X, y_: Y})
			print("Iteration " + str(i) + ", Cout  = ",loss, ", Précision  = " + "{:.5f}".format(acc))


	plt.figure()
	c1 = plt.scatter([1,0], [0,1], marker='s', color='red', s=100)
	c0 = plt.scatter([1,0], [1,0], marker='o', color='gray', s=100)
	# Generation de  points dand [-1,2]x[-1,2]
	DATA_x = (np.random.rand(10**6,2)*3)-1
	DATA_y = sess.run(y,feed_dict={x_: DATA_x})
	# Predictions
	ind = np.where(np.logical_and(0.49 < DATA_y, DATA_y< 0.51))[0]
	DATA_ind = DATA_x[ind]
	# Surfaces de séparation
	ss = plt.scatter(DATA_ind[:,0], DATA_ind[:,1], marker='_', color='blue', s=2)
	

	plt.legend((c1, c0, ss), ('Classe 1', 'Classe 0', 'Surfaces de séparation'), scatterpoints=1)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.axis([-1,2,-1,2])
	plt.show()

