# This file implements a t-SNE visualization of the training set and testing set on the 2D plane.
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(data)

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    
    x_list = [[],[],[],[]]
    y_list = [[],[],[],[]]
    x_test = [[],[],[],[]]
    y_test = [[],[],[],[]]
    
    for i in range(data.shape[0]):
        if label[i] < 4:
            x_list[label[i]].append(data[i,0])
            y_list[label[i]].append(data[i,1])
        else:
            x_test[label[i]-4].append(data[i,0])
            y_test[label[i]-4].append(data[i,1])
        
    label_list = ['calm', 'sad', 'terrified', 'happy']
    
    for i in range(4):
        plt.scatter(x_list[i], y_list[i], marker = 'o', s = 30, c='',
                    edgecolors=plt.cm.Set1((i + 1) / 10), label = label_list[i]+'(S)')
    
    for i in range(4):
        plt.scatter(x_test[i], y_test[i], marker = '^', s = 30, c='',
                    edgecolors=plt.cm.Set1((i + 1) / 10), label = label_list[i] + '(T)')
        
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    return fig

X_train = np.array([])
y_train = np.array([])
X_test = np.array([])
y_test = np.array([])


for i in range(10):
    data = scipy.io.loadmat('train/%d.mat'%(i+1))['de_feature']
    label = scipy.io.loadmat('train/%d.mat'%(i+1))['label']
    
    if i == 0:
        X_train = data
        y_train = label
    else:
        X_train = np.vstack((X_train, data))
        y_train = np.vstack((y_train, label))
        
for i in range(13, 14):
    data = scipy.io.loadmat('test/%d.mat'%(i))['de_feature']
    label = scipy.io.loadmat('test/%d.mat'%(i))['label']
    
    X_test = data
    y_test = label + 4

X_train = np.concatenate((X_train, X_test))
y_train = np.concatenate((y_train, y_test))    
    
X_train = (X_train - np.min(X_train, axis = 0)) / (np.max(X_train, axis = 0) - np.min(X_train, axis=0))

row_rand_array = np.arange(X_train.shape[0])
np.random.shuffle(row_rand_array)

X_sample = X_train[row_rand_array[0:3000]]
y_sample = y_train[row_rand_array[0:3000]]

fig = plot_embedding(X_sample, y_sample.squeeze(1),
                     't-SNE embedding of the data in training set and testing set')

plt.savefig('t-SNE-3.png', bbox_inches = 'tight', dpi = 200)

plt.show(fig)