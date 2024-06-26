import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalModel(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim):
        super(MultimodalModel, self).__init__()

        self.image_fc = nn.Linear(image_dim, hidden_dim)

        self.text_fc = nn.Linear(text_dim, hidden_dim)

        self.fusion_fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, image, text):
        image = F.relu(self.image_fc(image))

        text = F.relu(self.text_fc(text))

        fusion = torch.cat((image, text), dim=1)
        fusion = F.relu(self.fusion_fc1(fusion))
        fusion = self.fusion_fc2(fusion)
        output = torch.sigmoid(fusion)

        return output.squeeze()


class TestMultimodalModel(unittest.TestCase):
    def setUp(self):
        self.model = MultimodalModel(image_dim=512, text_dim=300,
                                     hidden_dim=256)
        self.model.load_state_dict(torch.load("model.pt"))
        self.model.eval()

    def test_forward(self):
        image = torch.randn(32, 512)
        text = torch.randn(32, 300)
        output = self.model(image, text)
        output = self.model(image, text)

        self.assertEqual(output.shape, torch.Size([32]))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))


if __name__ == '__main__':
    unittest.main()
