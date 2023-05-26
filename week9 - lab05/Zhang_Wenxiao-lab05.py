#!/usr/bin/env python
# coding: utf-8

# # CITS5508 - lab05
# 
# ### Wenxiao Zhang

# In[1]:


from data_loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import random
import time
import os
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ### Data Download and Preparation

# In[2]:


# step 1. Call the `load_batch` function
# According to the CIFAR-10 website, the training set is split into five batches
# stored in fives files. Each colour image has dimensions equal to 32 x 32 x 3. There
# are 10 classes.
image_width, image_height, image_Nchannels = 32, 32, 3
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# The default values of all the arguments of the load_batch function have been
# set for the CIFAR-10 dataset.
X_train, y_train = DataLoader.load_batch('data_batch')
X_test, y_test = DataLoader.load_batch('test_batch', Nbatches=1)

# Step 2. A quick inspection of the outputs from the `load_batch` function
# You need to split the training set to form a validation set. The original
# training set would become smaller.
print('X_test.shape =', X_test.shape, 'data type:', X_test.dtype)
print('y_test.shape =', y_test.shape, 'data type:', y_test.dtype)


# ### Tasks

# ### 1. perform 85/15 random split on training set

# In[3]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)


# ### 2. write a function to display 20 randomly sampled images

# In[4]:


def display_images(img_data, label, n_rows=2, n_cols=10):
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    # get 20 indices of images randomly 
    indices = random.sample(range(len(img_data)), n_rows*n_cols)
    random.shuffle(indices)
    # image plotting
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(img_data[indices[index]], cmap="binary",
                       interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[label[indices[index]]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


# In[5]:


print('Images from training set:')
display_images(X_train, y_train)
print('Images from validation set:')
display_images(X_val, y_val)
print('Images from testing set:')
display_images(X_test, y_test)


# ### 3. Implementation of an MLP

# #### Create a function that will build and compile a Keras model

# In[6]:


# create a function that will build and compile a Keras model
def build_model(n_hidden=2, n_neurons=500, dropout_rate=0.2, init_mode='he_normal', lr_schedule='exponetial'):
   model = keras.models.Sequential()

   # input layer
   model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))

   # hidden layers
   for layer in range(n_hidden):
      model.add(keras.layers.Dropout(dropout_rate))
      model.add(keras.layers.Dense(n_neurons, kernel_initializer=init_mode, activation="relu"))
      
   # output layer
   model.add(keras.layers.Dense(10, activation="softmax"))

   learning_rate = ''
   # exponetial decay scheduling
   if lr_schedule == 'exponetial':
      # number of steps in 20 epochs (batch size = 32)
      s = 20 * len(X_train) // 32
      learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
      

   # piecewise constant decay scheduling
   elif lr_schedule == 'piecewise':
      boundaries = [20 * len(X_train) // 32, 50 * len(X_train) // 32]
      values = [0.01, 0.005, 0.001]
      learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries, values)

   optimizer = keras.optimizers.SGD(learning_rate)
   model.compile(loss="sparse_categorical_crossentropy",
               optimizer=optimizer,
               metrics=["accuracy"])

   return model

# create a KerasClassifier based on this build_model() function:
keras_cls = keras.wrappers.scikit_learn.KerasClassifier(build_model, batch_size=32)


# #### Fine-tuning hyperparameter

# ```
# earlyStop = keras.callbacks.EarlyStopping(
#     monitor='val_loss', mode='min', min_delta=0.01, patience=10)
# grid_search_param = {
#     "lr_schedule": ['exponetial', 'piecewise'],
#     "init_mode": ['he_normal',  'he_uniform'],
#     "dropout_rate" : [0.2, 0.5],
# }
# grid_search_cv = GridSearchCV(keras_cls, grid_search_param, cv=3)
# grid_search_cv.fit(X_train, y_train, epochs=100,
#                   validation_data=(X_val, y_val),
#                    callbacks=[earlyStop])
# print(grid_search_cv.best_params_)
# ```
# ##### {'dropout_rate': 0.2, 'init_mode': 'he_uniform', 'lr_schedule': 'piecewise'}
# 

# #### Model training 

# In[7]:


if os.path.exists('Zhang_Wenxiao-MLP'):
    mlp_model = keras.models.load_model('Zhang_Wenxiao-MLP')
    print(mlp_model.summary())
    mlp_model.fit(X_train, y_train, epochs=1,
                  validation_data=(X_val, y_val))
else:
    os.makedirs('Zhang_Wenxiao-MLP')
    mlp_model = build_model(n_hidden=2, n_neurons=500, dropout_rate=0.2,
                            init_mode='he_uniform', lr_schedule='piecewise')
    mlp_model.fit(X_train, y_train, epochs=100,
                   validation_data=(X_val, y_val))
    mlp_model.save('Zhang_Wenxiao-MLP')


# #### Compare accuracies and F1 scores for both training and testing sets

# In[8]:


# training set
mlp_pred_train = mlp_model.predict(X_train)
mlp_pred_train_bool = np.argmax(mlp_pred_train, axis=1)
print(classification_report(y_train, mlp_pred_train_bool))


# In[9]:


# testing set
mlp_pred = mlp_model.predict(X_test)
mlp_pred_bool = np.argmax(mlp_pred, axis=1)
print(classification_report(y_test, mlp_pred_bool))


# #### Compare confusion matrices for both training and testing sets

# In[10]:


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

mlp_disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
    y_train, mlp_pred_train_bool))
mlp_disp_train.plot(ax = ax[0])
ax[0].set_title('MLP confusion matrix on training set')

mlp_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
    y_test, mlp_pred_bool))
mlp_disp.plot(ax=ax[1])
ax[1].set_title('MLP confusion matrix on testing set')

plt.show()


# In[ ]:





# ### 4. Implementation of a CNN

# #### Create a function that will build and compile a Keras model

# In[11]:


def build_model_cnn(kernel_size=(3,3), filters=32, activation="relu"):
    # create model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(kernel_size=kernel_size, filters=filters,
            activation=activation, padding="same", input_shape=[32, 32, 3]))
    model.add(keras.layers.BatchNormalization()),
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(kernel_size=kernel_size, filters=filters,
              activation=activation, padding="same", input_shape=[32, 32, 3]))

    # add MLP
    # input layer
    model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
   # hidden layers
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(
        500, kernel_initializer='he_uniform', activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(
        500, kernel_initializer='he_uniform', activation="relu"))
    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))
    # piecewise constant decay scheduling
    boundaries = [20 * len(X_train) // 32, 50 * len(X_train) // 32]
    values = [0.01, 0.005, 0.001]
    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])
    return model


# create a KerasClassifier based on this build_model_cnn() function:
keras_cls_cnn = keras.wrappers.scikit_learn.KerasClassifier(
    build_model_cnn, batch_size=32)


# In[12]:


print(tf.__version__)


# #### Fine-tuning hyperparameter

# In[13]:


grid_search_param_cnn = {
    "kernel_size": [(3,3), (4,4)],
    "filters": [32,  64],
    "activation": ['relu', 'selu'],
}
grid_search_cnn = GridSearchCV(keras_cls_cnn, grid_search_param_cnn, cv=3)
grid_search_cnn.fit(X_train, y_train, epochs=100,
                   validation_data=(X_val, y_val))

print(grid_search_cnn.best_params_)

# {'dropout_rate': 0.2, 'init_mode': 'he_uniform', 'lr_schedule': 'piecewise'}


# #### Model training 

# In[ ]:





# #### Compare accuracies and F1 scores for both training and testing sets

# In[ ]:





# #### Compare confusion matrices for both training and testing sets

# In[ ]:





# ### 5. Compare MLP and CNN models on the test set

# #### Write a function to display classified images
# 
# This function takes predicted results as input, display 5 images respectively for correctly and incorrectly classified images.

# In[ ]:


def display_pred_images(y_pred_bool):
    right_index = 0
    wrong_index = 0
    plt.figure(figsize=(7.5,  1.5))
    print("5 correctly classified images:")
    for i in range(len(y_pred_bool)):
        if y_test[i] == y_pred_bool[i]:
            right_index += 1
            plt.subplot(1, 5, right_index)
            plt.imshow(X_test[i], cmap="binary",
                       interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_pred_bool[i]], fontsize=12)
        if right_index == 5:
            break

    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
    print("")
    plt.figure(figsize=(7.5,  1.5))
    print("5 incorrectly classified images:")
    for j in range(len(y_pred_bool)):
        if y_test[j] != y_pred_bool[j]:
            wrong_index += 1
            plt.subplot(1, 5, wrong_index)
            plt.imshow(X_test[j], cmap="binary",
                       interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_pred_bool[j]], fontsize=12)
        if wrong_index == 5:
            break
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


# #### Display images classified by MLP

# In[ ]:


display_pred_images(mlp_pred_bool)


# In[ ]:




