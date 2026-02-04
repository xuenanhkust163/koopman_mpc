from dlkoopman.nets import AutoEncoder
import torch.nn as nn


class CACCKoopmanNet(AutoEncoder):
    def __init__(self, state_dim=5, koopman_dim=32):
        super().__init__()
        # 编码器：根据你的状态维度定制
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, koopman_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        # Koopman算子：一个线性层
        self.koopman_matrix = nn.Linear(koopman_dim, koopman_dim, bias=False)
        # 关键：必须注册，训练器才能识别和优化它
        self._register_koopman_matrix(self.koopman_matrix)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def koopman_operation(self, z):
        return self.koopman_matrix(z)
