import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Download Reuters Dataset for Multiclass classification problem
(x_trainData,y_trainLabels), (test_data, test_labels) = tf.keras.datasets.reuters.load_data(num_words=10000)

# Print both dataset length
print("Train Dataset Length: {}".format(len(x_trainData)))
print("Test Dataset Length: {}".format(len(test_data)))

#Print dataset 
print("Dataset: {}".format(x_trainData[10]))

#Dictionary Mapping
word_index = tf.keras.datasets.reuters.get_word_index()
reverse_word_index = dict([(value,key) for(value,key) in word_index.items()])
decoded_newswire = ''.join([reverse_word_index.get(i-3,'?') for i in x_trainData[0]])


#Length of Train Labels
print("Length of Train Labels: {}".format(y_trainLabels[10]))

#Vectorize Both Train & Test Dataset
def vectorize_Dataset(sequence,dimension=10000):
  result = np.zeros((len(sequence),dimension))
  for i, sequences in enumerate(sequence):
       result[i, sequences] = 1.
  return result

x_train = vectorize_Dataset(x_trainData)
x_test = vectorize_Dataset(test_data)

def to_one_hot(labels, dimensions=46):
   results = np.zeros((len(labels),dimensions))
   for i, label in enumerate(labels):
     results[i, label] = 1.
   return results

y_train = to_one_hot(y_trainLabels)
y_test = to_one_hot(test_labels)

#Building a Network
model = tf.keras.Sequential([
tf.keras.layers.Dense(64,activation='relu',input_shape=(10000,)),
tf.keras.layers.Dense(64,activation='relu'),
tf.keras.layers.Dense(46,activation='softmax')
])

#Objective Function, Lose and Optimizer
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#Validation of Data X
x_valid = x_train[:1000]
partial_x = x_train[1000:]

#Validation of Data Y
y_valid = y_train[:1000]
partial_y = y_train[1000:]

#Model Training
history = model.fit(
partial_x,
partial_y,
epochs=20,
batch_size=512,
validation_data=(x_valid,y_valid))

#Model Performance
loss = history.history['loss']
val_loss = history.history['loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs, loss, 'bo',label='Training Loss')
plt.plot(epochs, val_loss, 'b',label ='Validation Loss')
plt.title('Model Performance Training & Validation Loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.show()
 
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label= 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Model Performance Training & Validation Data')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Model appear in test
model.fit(
partial_x,
partial_y,
epochs=9,
batch_size=512,
validation_data=(x_valid,y_valid))

#Model evaluation
results = model.evaluate(
x_test,
y_test)

#Results
print("Results: {}".format(results))

#19% closer or not
import copy
test_copy_labels = copy.copy(test_labels)
np.random.shuffle(test_copy_labels)
hits = np.array(test_labels) == np.array(test_copy_labels)
model_results = float(np.sum(hits)/len(test_labels))
print("Results: {}".format(model_results))

#Predicition Results
predict = model.predict(x_test)
print("Vector Length:{}".format(predict[0].shape))

