import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

print(len(fashion_mnist.load_data()[0][0][0][0]))

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

train_images=train_images/255
test_images=test_images/255

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i in range(20):
  plt.subplot(4,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i],cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()  

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(1000,activation=tf.nn.relu),
    keras.layers.Dense(1000,activation=tf.nn.relu),
    keras.layers.Dense(500,activation=tf.nn.relu),
    keras.layers.Dense(200,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=6)

test_loss,test_acc=model.evaluate(test_images,test_labels)

print(test_loss,test_acc)

weights=[layer.get_weights() for layer in model.layers]
weights_cp=weights.copy()
print(weights_cp)

import math
def sort(weight):
  weight_full=[]
  for w in range(1,5):
    if(w==1):
      for i in range(784):
        for l in weight[w][0][i]:
          weight_full.append(l)
    if(w==2 or w==3):
      for j in range(1000):
        for l in weight[w][0][j]:
          weight_full.append(l)
    if(w==4):
      for x in range(500):
        for l in weight[w][0][x]:
          weight_full.append(l)
  weight_full.sort()
  return weight_full

  def wt_prune(weight,k):
  weight_full=[]
  weight_full=sort(weights_cp)
  count=0
  k=math.floor((k*3284/100)+1)
  print('weight_full',weight_full[k])
  for w in range(1,5):
    if(w==1):
      for i in range(784):
        for l in range(1000):
          if(weight[w][0][i][l]<weight_full[k]):
            weight[w][0][i][l]=0
            count=count+1
            #print('heyy',weight[w][0][i][l])
    if(w==2):
      for j in range(1000):
        for l in range(1000):
          if(weight[w][0][j][l]<weight_full[k]):
            weight[w][0][j][l]=0
            count=count+1
            #print('heyy',weight[w][0][j][l])
    if(w==3):
      for j in range(1000):
        for l in range(500):
          if(weight[w][0][j][l]<weight_full[k]):
            weight[w][0][j][l]=0
            count=count+1
            #print('heyy',weight[w][0][j][l])
    if(w==4):
      for x in range(500):
        for l in range(200):
          if(weight[w][0][x][l]<weight_full[k]):
            weight[w][0][x][l]=0
            count=count+1
            #print('heyy',weight[w][0][x][l])
            #print(4)
  for i in range(6):
    model.layers[i].set_weights(weight[i])
    #print('yo')
  #model.fit(train_images,train_labels,epochs=6)
  test_loss,test_acc=model.evaluate(test_images,test_labels)
  print(test_loss,test_acc,count/len(weight_full))
  return (count/len(weight_full),test_acc)


def wt_graph():
  lst=[0,25,50,60,70,80,90,95,97,99]
  X_sparse=[]
  Y_acc=[]
  for i in lst:
    weights_cp=weights.copy()
    tup=wt_prune(weights_cp,i)
    X_sparse.append(tup[0])
    Y_acc.append(tup[1])
  plt.plot(X_sparse,Y_acc)
  plt.show()

wt_garaph()

def unt_prune(weight,k):
  l2norm=[]
  l2norm.append([])
  count=0
  k=math.floor((k*3284/100)+1)
  for w in range(1,5):
    summ=0
    if(w==1):
      l2norm.append([math.sqrt(sum(weight[w][0][i]**2)) for i in range(784)])
    if(w==2):
      l2norm.append([math.sqrt(sum(weight[w][0][i]**2)) for i in range(1000)])
    if(w==3):
      l2norm.append([math.sqrt(sum(weight[w][0][i]**2)) for i in range(1000)])
    if(w==4):
      l2norm.append([math.sqrt(sum(weight[w][0][i]**2)) for i in range(500)])
    
  norm_mat=[]
  for i in l2norm:
    for j in i:
      norm_mat.append(j)
  print(len(norm_mat))
  norm_mat.sort()
  print(k,norm_mat[k])
  prune_nrns=[]
  for i in range(len(l2norm)):
    for j in range(len(l2norm[i])):
      if(l2norm[i][j]<norm_mat[k]):
        prune_nrns.append([i,j])
  print(prune_nrns)
  for x in prune_nrns:
    for y in range(len(weight[x[0]][0][x[1]])):
      weight[x[0]][0][x[1]][y]=0
  for i in range(6):
    model.layers[i].set_weights(weight[i])
    #print('yo')
  #model.fit(train_images,train_labels,epochs=6)
  test_loss,test_acc=model.evaluate(test_images,test_labels)       
  return(len(prune_nrns)/len(norm_mat),test_acc)
  
 def unt_graph():
  lst=[0,25,50,60,70,80,90,95,97,99]
  X_sparse=[]
  Y_acc=[]
  for i in lst:
    weights_cp=weights.copy()
    tup=unt_prune(weights_cp,i)
    X_sparse.append(tup[0])
    Y_acc.append(tup[1])
  plt.plot(X_sparse,Y_acc)
  plt.show()

unt_graph()

