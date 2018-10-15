#from __future__ import print_function
import numpy as np
import tensorflow as tf

''' 
-------------------------------------------------------------------------
Paramètres du réseau
-------------------------------------------------------------------------
''' 
learning_rate = 0.001
num_epochs = 15
batch_size = 100

# Nombre de neurones sur les deux couches cachées
num_hidden_1 = 256 
num_hidden_2 = 256 


'''
-------------------------------------------------------------------------
Données MNIST
-------------------------------------------------------------------------
''' 

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/mnistdata", one_hot=True)

num_examples = mnist.train.images.shape[0] 
num_input = mnist.train.images.shape[1]
num_classes = mnist.train.labels.shape[1]

print('num input',num_input)

# Espaces réservés qui vont être remplis par les tenseurs représentant l'ensemble des images et des labels lors de l'apprentissage
#(en utilisant l'argument {feed_dict} le la méthode run())
x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None,num_classes])



'''
-------------------------------------------------------------------------
Modèle du PMC
première couche cachée (avec ReLU) : y_1 = max(w_1^Tx+b_1,0)
deuxième couche cachée (avec ReLU) : y_2 = max(w_2^Ty_1+b_2,0)
-------------------------------------------------------------------------
'''
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Stockage des poids et biais dans des variables TF
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construction du modèle
pred = multilayer_perceptron(x, weights, biases)

# Fonction de perte et procédure d'optimisation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialisation des variables
init = tf.global_variables_initializer()

# Création d'une session TF pour exécuter le programme
with tf.Session() as sess:
    sess.run(init)

    # Entraînement
    total_batch = int(num_examples/batch_size)
    for epoch in range(num_epochs):
        avg_cost = 0.
        # Entraînement sur les batchs d'images
        for step in range (total_batch):
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            indices = perm[0:batch_size]
            batch_x = mnist.train.images[indices]
            batch_y = mnist.train.labels[indices]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
            # Perte moyenne
            avg_cost += c / total_batch
        print("Itération :", '%04d' % (epoch+1), "cout =", "{:.9f}".format(avg_cost))

    # Test du PMC
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Précision :", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
