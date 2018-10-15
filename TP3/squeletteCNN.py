import tensorflow as tf
import numpy as np


''' 
-------------------------------------------------------------------------
Paramètres du réseau
-------------------------------------------------------------------------
'''
learning_rate = 0.001
num_epochs = 2
batch_size = 100
dropout = 0.75 # ici probabilité de garder le neurone

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


# Affichage des informations par pas de temps
display_step = 10

# Espaces réservés qui vont être remplis par les tenseurs représentant l'ensemble des images et des labels lors de l'apprentissage
x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) 

'''
-------------------------------------------------------------------------
Convolution 2D avec RELU
-------------------------------------------------------------------------
''' 
def conv2d(x, W, b, strides=1):
    #TODO

'''
-------------------------------------------------------------------------
Pooling max
-------------------------------------------------------------------------
''' 
def maxpool2d(x, k=2):
    #TODO


'''
-------------------------------------------------------------------------
Modèle du réseau convolutif :
CONV1-RELU-CONV2-RELU-FCL-Prediction
-------------------------------------------------------------------------
'''
def conv_net(x, poids, biais, dropout):
    # Mise en forme de l'image d'entrée
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = #TODO
    conv2 = #TODO

    # Mise en forme des activations de la seconde couche cachée pour l'entrée de la couche complètement connectée
    fcl = tf.reshape(conv2, [-1, poids['wfcl'].get_shape().as_list()[0]])
    fcl = tf.add(tf.matmul(fcl, poids['fcl']), biais['bfcl'])
    fcl = tf.nn.relu(fcl)
    
    # Dropout 
    fcl = #TODO

    # Couche de sortie
    out = #TODO

    return out



#Stockage des poids et biais dans des variables TF
'''
TODO : initialiser avec une loi normale des variables tensorFlow :
    - wconv1 et bcconv1 pour CONV1. wconv1 est un banc de 32 filtres 5*5*1. bconv1 a une taille adaptée
    - wconv2 et bconv2 pour CONV2. wcconv2 est un banc de 32 filtres 5*5*32*64. bconv2 a une taille adaptée
    - wfcl et bfcl pour la couche complètement connectée : wfcl est un banc de filtres 7*7*64 à 1024 sorties. bfcl a une taille adaptée
    - out (poids et biais) ont une taille adaptée pour la classification des données MNIST
'''



# Construction du modèle
pred = conv_net(x, poids, biais, keep_prob)


# Fonction de perte et procédure d'optimisation
#TODO : entropie croisée avec logits, algorithme d'optimisation ADAM


# Evaluation du modèle
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialisation des variables
init = tf.global_variables_initializer()


# Création d'une session TF pour exécuter le programme
with tf.Session() as sess:
    sess.run(init)
    
    # Entraînement
    total_batch = int(num_examples/batch_size)
    for epoch in range(num_epochs):
        # Entraînement sur les batchs d'images
        for step in range (total_batch):
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            indices = perm[0:batch_size]
            batch_x = mnist.train.images[indices]
            batch_y = mnist.train.labels[indices]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})

            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
                print("Iteration " + str(epoch * total_batch + step) + ", Précision  = " + "{:.5f}".format(acc))

    # Test
   
    print("Test:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],y: mnist.test.labels[:256],keep_prob: 1.}))
