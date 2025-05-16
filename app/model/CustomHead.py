from torch import nn

# Custom classification head to fix issues with label smoothing
class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout_prob=0.4):
        super(CustomHead, self).__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)