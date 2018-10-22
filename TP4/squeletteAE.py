import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''
-------------------------------------------------------------------------
Données MNIST
-------------------------------------------------------------------------
''' 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/mnistdata", one_hot=False)

num_examples = mnist.train.images.shape[0] 
num_input = mnist.train.images.shape[1]

'''
-------------------------------------------------------------------------
Paramètres de l'autoencodeur
-------------------------------------------------------------------------
''' 
learning_rate = 0.01
num_iter = 20
batch_size = 256
n_hidden_1 = 256 
n_hidden_2 = 128 #Cas d'un autoencodeur à deux couches

'''
-------------------------------------------------------------------------
Paramètres d'affichage
-------------------------------------------------------------------------
''' 
display_step = 1
num_exemples = 10


X = tf.placeholder("float", [None, num_input])


'''
-------------------------------------------------------------------------
Définition des poids TODO
-------------------------------------------------------------------------
''' 

poids = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])), 
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])), 
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, num_input]))
}
biais = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input]))
}


'''
-------------------------------------------------------------------------
Encodeur. 
Les potentiels sont des produits scalaires + biais, les fonctions d'activation des sigmoïdes

-------------------------------------------------------------------------
''' 
def encoder(x):
    layer_1 =  tf.nn.sigmoid(tf.add(tf.matmul(x, poids['encoder_h1']), biais['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, poids['encoder_h2']), biais['encoder_b2']))
    return layer_2


'''
-------------------------------------------------------------------------
Décodeur 
Les potentiels sont des produits scalaires + biais, les fonctions d'activation des sigmoïdes

-------------------------------------------------------------------------
''' 
def decoder(x):
    layer_1 =  tf.nn.sigmoid(tf.add(tf.matmul(x, poids['decoder_h1']), biais['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, poids['decoder_h2']), biais['decoder_b2']))

    return layer_2

'''
-------------------------------------------------------------------------
Modèle
-------------------------------------------------------------------------
''' 
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prédiction
y_pred = decoder_op
# Sortie théorique
y_true = X

'''
-------------------------------------------------------------------------
Cout et optimisation
-------------------------------------------------------------------------
''' 
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(num_examples/batch_size)
    # Entrainement 
    for epoch in range(num_iter):
        # Par batchs
        for i in range(total_batch):
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            indices = perm[0:batch_size]
            batch_xs = mnist.train.images[indices]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Itération:", '%04d' % (epoch+1), "cout=", "{:.9f}".format(c))


    # Comparaison image théorique et reconstruction de l'autoencodeur
    AE = sess.run(y_pred, feed_dict={X: mnist.test.images[:num_exemples]})
    
    f, a = plt.subplots(2, num_exemples, figsize=(num_exemples, 2))
    for i in range(num_exemples):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(AE[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
