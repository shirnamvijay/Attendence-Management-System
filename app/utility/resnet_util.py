import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from PIL import Image
from torchvision.models import resnet50
import os

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        # Input shape= (256,3,150,150)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,75,75)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,75,75)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)
        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)
        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        # Above output will be in matrix form, with shape (256,32,75,75)
        output = output.view(-1, 32 * 75 * 75)
        output = self.fc(output)
        return output

class ResNetRunner():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.train_path = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
        self.test_path = os.path.join(os.path.dirname(__file__), "..", "..", "test")
        self.root=pathlib.Path(self.train_path)
        self.classes=None
        #Transforms
        self.transformer=transforms.Compose([
            transforms.Resize((150,150)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
            transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                                [0.5,0.5,0.5])
        ])
    def load_data(self):
        self.train_loader=DataLoader(
            torchvision.datasets.ImageFolder(self.train_path,transform=self.transformer),
            batch_size=64, shuffle=True
        )
        self.test_loader=DataLoader(
            torchvision.datasets.ImageFolder(self.train_path,transform=self.transformer),
            batch_size=32, shuffle=True
        )

    def train(self):
        #categories
        #self.classes=sorted([j.name.split('/')[-1] for j in self.root.iterdir()])

        print(self.classes)


        # CNN Network
        model=ResNet(num_classes=self.num_classes).to(self.device)
        #Optmizer and loss function
        optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
        loss_function=nn.CrossEntropyLoss()
        num_epochs=10
        #calculating the size of training and testing images
        train_count=len(glob.glob(self.train_path+'/**/*.jpg'))
        test_count=len(glob.glob(self.train_path+'/**/*.jpg'))
        print(train_count,test_count)
        # Model training and saving best model

        best_accuracy = 0.0

        for epoch in range(num_epochs):

            # Evaluation and training on training dataset
            model.train()
            train_accuracy = 0.0
            train_loss = 0.0

            for i, (images, labels) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                print(labels)
                optimizer.zero_grad()

                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_accuracy += int(torch.sum(prediction == labels.data))

            train_accuracy = train_accuracy / train_count
            train_loss = train_loss / train_count

            # Evaluation on testing dataset
            model.eval()

            test_accuracy = 0.0
            for i, (images, labels) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())

                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)
                test_accuracy += int(torch.sum(prediction == labels.data))

            test_accuracy = test_accuracy / test_count

            print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
                train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

            # Save the best model
            if test_accuracy > best_accuracy:
                torch.save(model.state_dict(),
                           os.path.join(os.path.dirname(__file__), "..", "..", "model", "best_checkpoint.model"))
                best_accuracy = test_accuracy

    # prediction function
    def prediction(self, image):
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "..", "..", "model", "best_checkpoint.model"))
        model = ResNet(num_classes=self.num_classes)
        model.load_state_dict(checkpoint)
        model.eval()
        image_tensor = self.transformer(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        if torch.cuda.is_available():
            image_tensor.cuda()
        input = Variable(image_tensor)
        output = model(input)

        # Apply softmax to get probabilities
        probabilities = nn.functional.softmax(output, dim=1)
        confidence, index = torch.max(probabilities, 1)
        print(confidence.item(), index.item())
        index = output.data.numpy().argmax()
        pred = self.classes[index]
        return pred, confidence.item()

    def test_prediction(self):
        images_path=glob.glob(self.test_path+'/*.jpg')
        if(not images_path):
            print("No Images are there to Test the predictions")
            return
        pred_dict={}
        for i in images_path:
            pred_dict[i[i.rfind('/')+1:]]=self.prediction(Image.open(i))
        print(pred_dict)