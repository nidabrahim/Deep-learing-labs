import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
mnist = input_data.read_data_sets("../data/mnistdata", one_hot=True)

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
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

'''
-------------------------------------------------------------------------
Pooling max
-------------------------------------------------------------------------
''' 
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

'''
-------------------------------------------------------------------------
Modèle du réseau convolutif :
CONV1-RELU-CONV2-RELU-FCL-Prediction
-------------------------------------------------------------------------
'''
def conv_net(x, poids, biais, dropout):
    # Mise en forme de l'image d'entrée
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, poids['wconv1'], biais['bconv1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, poids['wconv2'], biais['bconv2'])
    conv2 = maxpool2d(conv2, k=2)

    # Mise en forme des activations de la seconde couche cachée pour l'entrée de 
    # la couche complètement connectée
    fcl = tf.reshape(conv2, [-1, poids['wfcl'].get_shape().as_list()[0]])
    fcl = tf.add(tf.matmul(fcl, poids['wfcl']), biais['bfcl'])
    fcl = tf.nn.relu(fcl)
    
    # Dropout
    fcl_do = tf.nn.dropout(fcl, dropout)

    # Couche de sortie
    out = tf.add(tf.matmul(fcl_do, poids['out']), biais['out'])

    return out

#Stockage des poids et biais dans des variables TF
poids = {
    # CONV1 : noyaux 5x5 , 1 entrée, 32 sorties
    'wconv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # CONV2 : noyaux5x5 conv, 32 entrées, 64 sorties
    'wconv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # FCL 7*7*64 entrées, 1024 sorties
    'wfcl': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # Prédiction : 1024 entrées, 10 sorties (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biais = {
    'bconv1': tf.Variable(tf.random_normal([32])),
    'bconv2': tf.Variable(tf.random_normal([64])),
    'bfcl': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Construction du modèle
pred = conv_net(x, poids, biais, keep_prob)


# Fonction de perte et procédure d'optimisation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


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

    # Visualisation des 5 premiers filtres de chaque couche
    filters = poids['wconv1'].eval()
    filters2 = poids['wconv2'].eval()
    fig = plt . figure ()
    for i in range(10):
    	ax = fig.add_subplot(4, 5, i+1)
    	ax.matshow(filters[: ,: ,0 , i ] , cmap='gray')
    	ax = fig.add_subplot(4, 5, 10+i+1)
    	ax.matshow(filters2[: ,: ,0 , i ] , cmap='gray')
    plt.suptitle('Dix premiers filtres (conv1 ligne 1, conv2 ligne 2)')
    fig.savefig('filters.png')
    

