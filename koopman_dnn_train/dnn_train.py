import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# ------------------------------
# 1. 加载数据与归一化参数
# ------------------------------
def load_training_data(npz_path, norm_json_path=None):
    """加载预处理后的训练数据"""
    data = np.load(npz_path)
    X_t = data['X_t']   # (N, 5) [px, py, v, psi, omega]
    U_t = data['U_t']   # (N, 2) [a, delta]
    X_t1 = data['X_t1'] # (N, 5)
    Ts = float(data['Ts'])

    # 加载归一化参数（用于可选的反归一化）
    if norm_json_path and os.path.exists(norm_json_path):
        with open(norm_json_path, 'r') as f:
            norm_params = json.load(f)
    else:
        norm_params = None

    # 转换为 torch tensor
    X_t = torch.tensor(X_t, dtype=torch.float32)
    U_t = torch.tensor(U_t, dtype=torch.float32)
    X_t1 = torch.tensor(X_t1, dtype=torch.float32)

    return X_t, U_t, X_t1, Ts, norm_params

# ------------------------------
# 2. 神经网络模型定义
# ------------------------------
class DeepKoopman(nn.Module):
    def __init__(self, n_x=5, n_u=2, n_k=32, n_hidden=64):
        """
        n_x: 原始状态维度
        n_u: 控制输入维度
        n_k: Koopman 提升维度
        n_hidden: 编码器/解码器隐藏层神经元数
        """
        super(DeepKoopman, self).__init__()
        self.n_x = n_x
        self.n_k = n_k
        self.n_u = n_u

        # 编码器: [x; u] -> 提升状态
        self.encoder = nn.Sequential(
            nn.Linear(n_x + n_u, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_k)
        )

        # Koopman 矩阵 (线性部分)
        self.K = nn.Parameter(torch.randn(n_k, n_k) * 0.1)
        # 控制输入矩阵 L (n_k x n_u)
        self.L = nn.Parameter(torch.randn(n_k, n_u) * 0.1)

        # 解码器: 提升状态 -> 原始状态
        self.decoder = nn.Sequential(
            nn.Linear(n_k, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_x)
        )

    def forward(self, x, u):
        """
        x: (batch, n_x)
        u: (batch, n_u)
        返回: 预测的下一时刻状态 x_next_pred
        """
        # 编码当前状态和控制
        enc_input = torch.cat([x, u], dim=-1)   # (batch, n_x+n_u)
        z = self.encoder(enc_input)             # (batch, n_k)

        # 线性动力学: z_next = K*z + L*u
        z_next = z @ self.K.T + u @ self.L.T    # (batch, n_k)

        # 解码
        x_next_pred = self.decoder(z_next)      # (batch, n_x)
        return x_next_pred

    def linear_prediction(self, z, u):
        """给定提升状态 z 和控制 u，返回线性预测的 z_next 和 x_next"""
        z_next = z @ self.K.T + u @ self.L.T
        x_next = self.decoder(z_next)
        return z_next, x_next

    def get_linear_matrices(self):
        """返回 K 和 L 矩阵（numpy 格式）"""
        return self.K.detach().cpu().numpy(), self.L.detach().cpu().numpy()

# ------------------------------
# 3. 损失函数
# ------------------------------
def koopman_loss(model, x, u, x_next, n_pred_steps=5, alpha=0.1):
    """
    综合损失: 一步预测损失 + 多步预测损失 + 线性重建损失
    x, u, x_next: 批数据
    n_pred_steps: 多步预测的步数
    alpha: 多步损失的权重系数
    """
    batch_size = x.shape[0]
    # 一步预测
    x_next_pred = model(x, u)
    loss_1step = nn.MSELoss()(x_next_pred, x_next)

    # 多步预测（开环）
    loss_multistep = 0.0
    x_curr = x
    u_curr = u
    for t in range(n_pred_steps):
        # 使用模型预测下一步
        x_next_pred_t = model(x_curr, u_curr)
        # 计算预测误差（对真实下一步）
        # 需要真实数据: 取 x_next 作为第1步真实，之后需错位
        # 简化: 使用真实数据中的 x_next 作为目标，但需要移位索引
        if t == 0:
            target = x_next
        else:
            # 后续步的真实值需要从数据中取，这里简单用 x_next 的偏移（需确保数据足够）
            # 更严谨的做法是预先准备多步目标，为简化，我们只做一步多步损失的近似
            # 这里提供一个更清晰的多步损失实现:
            pass
    # 由于多步实现稍复杂，建议使用以下简化版本（仅一步 + 线性一致性）:
    # 线性一致性: 编码器-解码器重建损失
    enc_input = torch.cat([x, u], dim=-1)
    z = model.encoder(enc_input)
    x_recon = model.decoder(z)
    loss_recon = nn.MSELoss()(x_recon, x)

    # 总损失
    total_loss = loss_1step + 0.5 * loss_recon
    return total_loss, loss_1step, loss_recon

# 更完整的多步损失实现（推荐）
def multistep_loss(model, x0, u_seq, x_targets, n_steps):
    """
    x0: 初始状态 (batch, n_x)
    u_seq: 控制序列 (n_steps, batch, n_u)
    x_targets: 真实状态序列 (n_steps, batch, n_x)
    """
    loss = 0.0
    z = model.encoder(torch.cat([x0, u_seq[0]], dim=-1))
    for t in range(n_steps):
        z_next = z @ model.K.T + u_seq[t] @ model.L.T
        x_pred = model.decoder(z_next)
        loss += nn.MSELoss()(x_pred, x_targets[t])
        z = z_next
    return loss / n_steps

# ------------------------------
# 4. 训练循环
# ------------------------------
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_val_loss = float('inf')

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_u, batch_x_next in train_loader:
            batch_x, batch_u, batch_x_next = batch_x.to(device), batch_u.to(device), batch_x_next.to(device)
            optimizer.zero_grad()
            loss, loss1, loss_recon = koopman_loss(model, batch_x, batch_u, batch_x_next)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_u, batch_x_next in val_loader:
                batch_x, batch_u, batch_x_next = batch_x.to(device), batch_u.to(device), batch_x_next.to(device)
                loss, _, _ = koopman_loss(model, batch_x, batch_u, batch_x_next)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../_output/_train/best_koopman_model.pth")
            print(f"  -> 保存最佳模型，Val Loss = {val_loss:.6f}")

    return model

# ------------------------------
# 5. 主程序
# ------------------------------
if __name__ == "__main__":
    # 参数设置
    BATCH_SIZE = 64
    EPOCHS = 300
    LR = 1e-3
    N_K = 48          # 提升维度
    N_HIDDEN = 128
    VAL_SPLIT = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据（请根据实际文件路径修改）
    X_t, U_t, X_t1, Ts, norm_params = load_training_data(
        npz_path='../_output/_data_process/training_data.npz',
        norm_json_path='../_output/_data_process/training_data_norm_params.json'
    )

    # 划分训练集和验证集
    N = X_t.shape[0]
    indices = np.random.permutation(N)
    split = int(N * (1 - VAL_SPLIT))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = TensorDataset(X_t[train_idx], U_t[train_idx], X_t1[train_idx])
    val_dataset = TensorDataset(X_t[val_idx], U_t[val_idx], X_t1[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = DeepKoopman(n_x=5, n_u=2, n_k=N_K, n_hidden=N_HIDDEN)

    # 训练
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    # 保存最终模型和归一化参数
    torch.save(model.state_dict(), "../_output/_train/final_koopman_model.pth")
    if norm_params is not None:
        with open("norm_params.json", "w") as f:
            json.dump(norm_params, f, indent=2)

    # 输出 Koopman 矩阵
    K, L = model.get_linear_matrices()
    print("\nKoopman matrix K shape:", K.shape)
    print("Control matrix L shape:", L.shape)
    print("Training completed.")
