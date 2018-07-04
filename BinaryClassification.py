import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Step : 1 I/O CODE:
	# Dataset download from imdb 
(x_features, y_samples), (x_testFeatures, y_testSamples) = tf.keras.datasets.imdb.load_data(num_words= 10000)

print("x_features {}".format(x_features[0]))
print("y_samples {}".format(y_samples[0]))
	# Data restricted 
print("{}".format(max([max(sequence) for sequence in x_features])))
	# Dataset getword_imdb
word_index = tf.keras.datasets.imdb.get_word_index()
	# classfication 0, 1, 2
reverse_word_index = dict([(value,key) for value, key in word_index.items()])   # map according to dictinonary
decode_review =' '.join([reverse_word_index.get(i-3 ,'?') for i in x_features[0]])  # decode review

#Convert Features List into Neural Net

def sequence_for_vectors(sequences, dimension=10000):
	results = np.zeros((len(sequences),dimension))	
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

train_features = sequence_for_vectors(x_features)
test_features = sequence_for_vectors(x_testFeatures)

print("Features_train_dataset {}".format(train_features[0]))

#Labels vectorize labels
train_samples = np.asarray(y_samples).astype('float32')
test_samples = np.asarray(y_testSamples).astype('float32')


#Now Model feed my data
model = tf.keras.Sequential([
tf.keras.layers.Dense(16,activation='relu',input_shape=(10000,)),
tf.keras.layers.Dense(16,activation='relu',),
tf.keras.layers.Dense(1,activation='sigmoid')
])

#Model Optimizer, lose and metrics 
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

#Dataset divide into two 
x_val = train_features[:10000]
partial_x_train = train_features[10000:]   # Features

y_val = train_samples[:10000]
partial_y_train = train_samples[10000:]     #Labels

#Training on Dataset
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

#During training
history_dic = history.history
print("{}".format(history_dic.keys()))

#Graph of Training Dataset

loss_values = history_dic['loss']
val_loss_values = history_dic['val_loss']

epochs = range(1,len(history_dic['acc'])+1)
plt.plot(epochs, loss_values, 'bo', label='Training_Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation_Loss')
plt.title("Training and Validation Loss of Data")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Accuracy Graph
plt.clf()
acc = history_dic['acc']
val_acc = history_dic['val_acc']

plt.plot(epochs,acc,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
plt.title("Dataset Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Test Dataset 
model.fit(test_features,test_samples,epochs=4,batch_size=512)
results = model.evaluate(test_features,test_samples)

print("Results ={}".format(results))
print("Predicition On Reviews = {}".format(model.predict(test_features)))