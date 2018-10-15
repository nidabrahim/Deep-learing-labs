import numpy as np
import tensorflow as tf

import plotResults

# Nombre de classes
num_labels = 2    
# Taille des batchs par apprentissage
batch_size = 100  

tf.app.flags.DEFINE_string('train', None,'Fichier des données d''entraînement (données et labels).')
tf.app.flags.DEFINE_string('test', None,'Fichier des données de test (données et labels).')
tf.app.flags.DEFINE_integer('num_epochs', 1,'Nombre de passage des batchs la base d''apprentissage.')
tf.app.flags.DEFINE_integer('num_hidden', 1,'Nombre de neurones de la couche cachée.')
FLAGS = tf.app.flags.FLAGS

# Extraction des données à partir de lignes label, desc1... descn en un format compatible
def extract_data(filename):
    labels = []
    features = []

    for line in open(filename):
        row = line.split(",")
        # Les labels sont des entiers
        labels.append(int(row[0]))
        # les descripteurs sont des réels
        features.append([float(x) for x in row[1:]])
    # Conversion en des types  matrices numpy
    features_np = np.matrix(features).astype(np.float32)
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels_onehot = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)

    return features_np,labels_onehot

# Initialisation des poids : méthode de Xavier 
def init_weights(shape,  xavier_params = (None, None)):
    #TODO
    
def main(argv=None):
    num_epochs = FLAGS.num_epochs
    
    # Extraction des données dans des matruces numpy
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Taille de la base d'apprentissage
    train_size,num_features = train_data.shape

    # Taille de la couche cachée
    num_hidden = FLAGS.num_hidden
 
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, num_labels])
    
    # Données test dans un noeud constant du graphe TF
    test_data_node = tf.constant(test_data)

    # Initialisation des poids et biais entre rétine et couche cachée
    w_hidden = #TODO
    b_hidden = #TODO

    # Calcul des activations de la couche cachée
    hidden = #TODO

    # Initialisation des poids et biais entre  couche cachée et couche de sortie
    w_out = #TODO
    b_out = #TODO

    # Calcul de la sortie du réseau
    #TODO
    
    # Optimisation.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    #Descente de Gradient sur l'entropie croisée
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    # Evaluation.
    predicted_class = tf.argmax(y,1);
    success = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(success, "float"))

    # Création d'une session TF pour exécuter le programme
    with tf.Session() as session:
        # Initialisation des variables
        tf.global_variables_initializer().run()
    	    
        # Entraînement sur batchs
        for step in range (num_epochs * train_size // batch_size):
    	    offset = (step * batch_size) % train_size
    	    batch_data = train_data[offset:(offset + batch_size), :]
    	    batch_labels = train_labels[offset:(offset + batch_size)]
    	    train_step.run(feed_dict={x: batch_data, y_: batch_labels})
    	    if offset >= train_size-batch_size:
            	print ("Précision:", accuracy.eval(feed_dict={x: test_data, y_: test_labels}))

        eval_fun = lambda X: predicted_class.eval(feed_dict={x:X}); 
        plotResults.plot(test_data, test_labels, eval_fun)
            
if __name__ == '__main__':
    tf.app.run()
