#1
!pip install -q git+https://github.com/tensorflow/docs

#2
!wget -N https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
  
#3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#4
dataset = pd.read_csv('insurance.csv')
len(dataset)

#5
dataset.head()

#6
df = dataset
df["sex"] = pd.factorize(df["sex"])[0]
df["region"] = pd.factorize(df["region"])[0]
df["smoker"] = pd.factorize(df["smoker"])[0]
dataset = df
dataset.head()

#7
test_dataset = dataset.sample(frac=0.2)
len(test_dataset)

#8
train_dataset = dataset[~dataset.isin(test_dataset)].dropna()
len(train_dataset)

#9
train_dataset.head()

#10
train_labels = train_dataset.pop("expenses")
train_labels.head()

#11
train_dataset.head()

#12
test_labels = test_dataset.pop("expenses")
test_labels.head()

#13
train_dataset.head()

#14
normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt(np.array(train_dataset))

#15
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)
model.build()
model.summary()

#16
history = model.fit(
    train_dataset,
    train_labels,
    epochs=100,
    validation_split=0.5,
    verbose=0, 
)

print(history)

#17
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
