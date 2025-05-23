# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Load the dataset
Load the Iris dataset using a suitable library.
### STEP 2: Preprocess the data
Preprocess the data by handling missing values and normalizing features.
### STEP 3: Split the dataset
Split the dataset into training and testing sets.
### STEP 4: Train the model
Train a classification model using the training data.
### STEP 5: Evaluate the model
Evaluate the model on the test data and calculate accuracy.
### STEP 6:  Display results
Display the test accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: Sai Vishal D

### Register Number: 212223230180

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```
```
class Model(nn.Module):
    def __init__(self, in_features=4, h1=10, h2=11, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

```
```
torch.manual_seed(32)
model = Model()
```

```
df = pd.read_csv('iris.csv')
df.head()
```
```
X = df.drop('target',axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
     
```
```
torch.manual_seed(4)
model = Model()
```
```

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
```
epochs = 100
losses = []

for i in range(epochs):
    i+=1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
```
import numpy as np
import matplotlib.pyplot as plt

# Convert each tensor in the list to a NumPy array
losses_np = np.array([loss.detach().cpu().numpy() if hasattr(loss, "detach") else loss for loss in losses])

plt.plot(range(epochs), losses_np)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

```
```
with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'{loss:.8f}')
```
```

torch.save(model.state_dict(), 'IrisDatasetModel.pt')
```
```
new_model = Model()
new_model.load_state_dict(torch.load('IrisDatasetModel.pt'))
new_model.eval()
```
```
with torch.no_grad():
    y_val = new_model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'{loss:.8f}')

```
```
mystery_iris = torch.tensor([5.6,3.7,2.2,0.5])
```

```
with torch.no_grad():
    print(new_model(mystery_iris))
    print()
    print(labels[new_model(mystery_iris).argmax()])
```

### Dataset Information
![image](https://github.com/user-attachments/assets/d0ab1554-91d7-4261-8de2-411c452355fb)

### OUTPUT

![image](https://github.com/user-attachments/assets/1bb7abe5-5abf-4795-ba0f-452b2f14afb3)


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
