from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)


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


model = MultimodalModel(image_dim=512, text_dim=300, hidden_dim=256)
model.load_state_dict(torch.load('model.pt'))


model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    image = torch.tensor(data['image'])
    text = torch.tensor(data['text'])

    with torch.no_grad():
        output = model(image, text)
    prediction = output.numpy().tolist()

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
