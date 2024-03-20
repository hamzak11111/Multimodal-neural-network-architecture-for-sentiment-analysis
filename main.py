import pandas as pd
import nltk
import re
from torchvision.datasets import ImageFolder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from torch import tensor
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim
import numpy as np
import ast
from torch.utils.data import TensorDataset, DataLoader
import gensim.downloader as api
import os
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df=pd.read_csv("sentiment.csv")

def clean(x):
    return nltk.word_tokenize(re.sub(r'[^\w\s]', '', x.lower()))

df["raw"]=df['raw'].apply(clean)

glove_vectors = api.load("glove-wiki-gigaword-300")

def embed(x):
    tokens = x
    embedded_tokens = []
    for token in tokens:
        if token in glove_vectors:
            embedded_tokens.append(glove_vectors[token])
    return np.mean(embedded_tokens, axis=0)

# Output the embedded sentences
df["raw"]=df["raw"].apply(embed)

os.chdir("sentiment_images")

files = [f for f in os.listdir(os.getcwd()) if f]

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])

resnet = torchvision.models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

images = []
labels = []

for x in files:
    img = Image.open(x)
    if img.mode == 'L': 
        img = TF.to_tensor(img).expand(3, -1, -1)
    elif img.mode == 'RGB': 
        img = TF.to_tensor(img)
    else:
        raise ValueError("Unsupported color mode: %s" % img.mode)
    img = img.unsqueeze(0)
    features = resnet(img)
    features = features.squeeze().detach().numpy()
    images.append(features)
    labels.append(x)

images = np.array(images)
labels = np.array(labels)

os.chdir("..")

image_features=[]
notPresent=[]
for j in df["filename"]:
    bol=True
    for i in range(0,len(images)):
        if labels[i]==j:
            image_features.append(images[i])
            bol=False
            break
    if bol==True:
        if j not in notPresent:
            notPresent.append(j)

df = df[~df['filename'].isin(notPresent)]
df["image_features"]=image_features

df=df.reset_index(drop=True)

train_caption=df[df["split"]=="train"]

train_caption_image=torch.tensor(train_caption["image_features"].tolist())
train_caption_word= torch.tensor(train_caption["raw"].tolist())
train_caption_label=torch.tensor(train_caption["sentiment"].values)


test_caption=df[df["split"]=="test"]

test_caption_image=torch.tensor(test_caption["image_features"].tolist())
test_caption_word= torch.tensor(test_caption["raw"].tolist())
test_caption_label=torch.tensor(test_caption["sentiment"].values)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim):
        super(MultimodalModel, self).__init__()

        # image branch
        self.image_fc = nn.Linear(image_dim, hidden_dim)

        # text branch
        self.text_fc = nn.Linear(text_dim, hidden_dim)

        # multimodal fusion
        self.fusion_fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, image, text):
        # image branch
        image = F.relu(self.image_fc(image))

        # text branch
        text = F.relu(self.text_fc(text))

        # multimodal fusion
        fusion = torch.cat((image, text), dim=1)
        fusion = F.relu(self.fusion_fc1(fusion))
        fusion = self.fusion_fc2(fusion)
        output = torch.sigmoid(fusion)

        return output.squeeze()
    
# define model
model = MultimodalModel(image_dim=512, text_dim=300, hidden_dim=256)

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

dataset = TensorDataset(train_caption_image,train_caption_word,train_caption_label)
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size)
num_epochs=15

# train loop
for epoch in range(num_epochs):
    for image, text, label in dataloader:
        optimizer.zero_grad()

        # forward pass
        output = model(image, text)

        # calculate loss
        loss = criterion(output, label.float())

        # backward pass and optimization
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}")
torch.save(model.state_dict(), "model.pt")

model = MultimodalModel(image_dim=512, text_dim=300, hidden_dim=256)
model.load_state_dict(torch.load("model.pt"))

dataset = TensorDataset(test_caption_image, test_caption_word, torch.tensor(test_caption["sentiment"].values))
batch_size = 100
val_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# evaluate on validation set
with torch.no_grad():
        val_loss = 0
        num_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        for image, text, label in val_data_loader:
            output = model(image, text)
            val_loss += criterion(output, label.float()).item()
            predictions = (output > 0.5).long()
            num_correct += (predictions == label).sum().item()
            total_samples += label.size(0)
            all_predictions += predictions.tolist()
            all_labels += label.tolist()
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)

print("accuracy: ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("F1 score: ",f1)
print("Predictions:", all_predictions)
print("Labels:", all_labels)

