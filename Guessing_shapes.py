import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#Defining a set of transformations for the pre-processing of the images.
my_tranform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

#Images saved in a folder called "Shapes".
dataset = torchvision.datasets.ImageFolder(root='Shapes', transform=my_tranform)

#We split the set of images into a set to train the model and a set to test it.
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, \
                                             batch_size=64,
                                             shuffle=True)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, \
                                             batch_size=64)


#We create a class with our convolutional neural network model.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 8, 5, padding = 2)
        self.layer2 = nn.Conv2d(8, 16, 5, padding = 2)
        self.layer3 = nn.Conv2d(16, 32, 5, padding = 2)
        self.layer4 = nn.Conv2d(32, 64, 5, padding = 2)
        self.layer5 = nn.Linear(64*64*64, 3)
        
        self.relu = nn.ReLU()
    def forward(self, x):
        l = self.relu(self.layer1(x))
        l = self.relu(self.layer2(l))
        l = self.relu(self.layer3(l))
        l = self.relu(self.layer4(l))
        l = l.view(-1,64*64*64)
        return F.softmax(self.layer5(l), dim=1)

#Creating an instance of our model
net = Net()

#Creating optimizer (Adam)
opt = optim.Adam(net.parameters(), 0.0001)

#Mean Squared Error
loss_fn = nn.MSELoss()

#We will keep the losses and the accuracies in a list
losses = []
accuracies = [0.1]

#Training the model
for epoch in range(100):
    total_correct = 0
    total = train_size
    for images, labels in train_dataset_loader:
        print(images.max(), images.min(), images.mean(), images.std())
        hot_predictions = net(images)
        predictions = hot_predictions.argmax(axis=1)
        nb_correct = sum(predictions == labels)
        total_correct += nb_correct
        hot_labels = F.one_hot(labels, 3).to(torch.float32)
        loss = loss_fn(hot_labels, hot_predictions)
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()
    acc = total_correct/total
    accuracies.append(acc.item())
    print(f"Epoch: {epoch}\t Accuracy: {acc}")

#Plotting graphs with losses and accuracies
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(losses)
plt.subplot(1,2,2)
plt.plot(accuracies, "*r")
plt.ylim(0,1)
plt.show()

#Testing the model
total_correct = 0
total = test_size
for images, labels in test_dataset_loader:
    hot_predictions = net(images)
    predictions = hot_predictions.argmax(axis=1)
    nb_correct = sum(predictions == labels)
    total_correct += nb_correct
    hot_labels = F.one_hot(labels, 3).to(torch.float32)
    loss = loss_fn(hot_labels, hot_predictions)
    loss.backward()
    opt.step()
    opt.zero_grad()
acc = total_correct/total
print(f"Accuracy: {acc}")

#Extracting the model into a onnx file
dummy_input = torch.ones(1, 3, 64, 64)
torch.onnx.export(net, dummy_input, 'Test_model.onnx', export_params = True, verbose = True)