import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def plot(X,Y,pred_func):
    # rectangle de tracé
    mins = np.amin(X,0); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,0); 
    maxs = maxs + 0.1*maxs;

    ## Génération d'une grille
    xs,ys = np.meshgrid(np.linspace(mins[0,0],maxs[0,0],300),np.linspace(mins[0,1], maxs[0,1], 300));

    # Modèle sur la grille
    Z = pred_func(np.c_[xs.flatten(), ys.flatten()]);
    Z = Z.reshape(xs.shape)

    # Contour et exemples d'apprentissate
    plt.contourf(xs, ys, Z, cmap=plt.cm.magma)
    toto = Y[:,1].reshape(X[:, 0].shape)
    print(X[:,0].shape,X[:,1].shape,toto.shape)
    plt.scatter(X[:, 0], X[:, 1], c=toto, s=50,cmap=colors.ListedColormap(['red', 'green']))
#    plt.scatter([X[:, 0]], [X[:, 1]],c=toto,s=50,cmap=colors.ListedColormap(['red', 'green']))
    plt.show()
