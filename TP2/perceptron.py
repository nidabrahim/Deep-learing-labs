import numpy as np
import tensorflow as tf
import plotResults

# Nombre de classes
num_labels = 2    
# Taille des batchs par apprentissage
batch_size = 100  

'''usage python3 perceptron.py --train ./simdata/train_file.csv 
                        --test ./simdata/eval_file.csv --trace True --num_epochs 10
'''
tf.app.flags.DEFINE_string('train', None,
                           'Fichier des données d''entraînement (données et labels).')
tf.app.flags.DEFINE_string('test', None,
                           'Fichier des données de test (données et labels).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Nombre de passage des batchs la base d''apprentissage.')
tf.app.flags.DEFINE_boolean('trace', False, 'trace de l''exécution.')
tf.app.flags.DEFINE_boolean('plot', True, 'Tracé de la frontière de décision.')
FLAGS = tf.app.flags.FLAGS

# Extraction des données à partir de lignes label, descripteur_1... descripteur_n en un format compatible
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

def main(argv=None):
    trace = FLAGS.trace
    plot = FLAGS.plot
    num_epochs = FLAGS.num_epochs

    # Extraction des données dans des matruces numpy
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Taille de la base d'apprentissage
    train_size,num_features = train_data.shape

    # Définition des tableaux TF permettant de passer un batch
    # d'image à chaque itération(en utilisant l'argument {feed_dict} le la méthode run())
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, num_labels])
    
    # Définition du modèle
    W = tf.Variable(tf.zeros([num_features,num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # Optimisation.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    #Descente de Gradient sur l'entropie croisée
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
   # Fonction de perte et procédure d'optimisation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.AdamOptimizer(learning_rate=0.15).minimize(cost)

    # Données text dans un noeud constant du graphe TF
    test_data_node = tf.constant(test_data)

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

            # Création d'un batch d'images
            batch_data = train_data[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # Les données d'entraînement sont données au modèle
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})           

        # Sortie 
        if trace:
            print ('Matrice des poids : ',session.run(W))
            print ('Biais :',session.run(b))
            point = test_data[:1]
            print ("Exemple de prédiction sur le point ",point,' :')
            print ("Wx+b = ", session.run(tf.matmul(point,W)+b))
            print ("softmax(Wx+b) = ", session.run(tf.nn.softmax(tf.matmul(point,W)+b)))
            print
            
        print ("Taux de bonne classification:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})*100,'%')

        if plot:
            eval_fun = lambda X: predicted_class.eval(feed_dict={x:X}); 
            plotResults.plot(test_data, test_labels, eval_fun)
    
if __name__ == '__main__':
    tf.app.run()
