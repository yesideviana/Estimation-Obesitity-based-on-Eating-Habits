import streamlit as st
st.title('Estimasi Kegemukan pada Kebiasaan makan')
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('obesity_dataset.csv')

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the neural network
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(zip(X_train, y_train)):
        inputs, labels = data
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch+1, running_loss / len(y_train)))

# Evaluate the neural network
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_test, y_test)):
        inputs, labels = data
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the neural network on the test data: %.2f %%' % (100 * correct / total))
