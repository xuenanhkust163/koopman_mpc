import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==================== 1. 定义深度 Koopman 模型 ====================
class DeepKoopmanModel(nn.Module):
    """
    用于商用车CACC的深度Koopman模型。
    架构: 编码器 - Koopman算子 - 解码器
    """

    def __init__(self, state_dim, koopman_dim, hidden_dims=[128, 256, 128]):
        """
        Args:
            state_dim: 原始状态维度 (例如: [自车速度, 加速度, 相对距离, 相对速度...])
            koopman_dim: 升维后的Koopman状态维度 (通常 > state_dim)
            hidden_dims: 编码器/解码器网络的隐藏层维度列表
        """
        super(DeepKoopmanModel, self).__init__()
        self.state_dim = state_dim
        self.koopman_dim = koopman_dim

        # --- 编码器网络 (非线性观测函数 φ: R^state_dim -> R^koopman_dim) ---
        encoder_layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, koopman_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Koopman 算子 (线性变换矩阵 K) ---
        # 这是核心，一个可学习的线性层（无偏置），代表 z_{t+1} = K * z_t
        self.koopman_matrix = nn.Linear(koopman_dim, koopman_dim, bias=False)

        # --- 解码器网络 (逆映射 ψ: R^koopman_dim -> R^state_dim) ---
        decoder_layers = []
        prev_dim = koopman_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, state_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """将原始状态映射到Koopman空间: z = φ(x)"""
        return self.encoder(x)

    def decode(self, z):
        """将Koopman状态映射回原始空间: x̂ = ψ(z)"""
        return self.decoder(z)

    def koopman_operation(self, z):
        """在Koopman空间中应用线性动力学: z_next = K * z"""
        return self.koopman_matrix(z)

    def forward(self, x_current, x_next=None):
        """
        前向传播，支持训练和推理两种模式。
        Args:
            x_current: 当前时刻状态
            x_next: 下一时刻真实状态 (训练时提供)
        Returns:
            包含各种输出和损失的字典
        """
        # 编码
        z_current = self.encode(x_current)

        # 在Koopman空间中进行线性预测
        z_next_pred = self.koopman_operation(z_current)

        # 解码重构和预测
        x_current_recon = self.decode(z_current)
        x_next_pred = self.decode(z_next_pred)

        outputs = {
            'z_current': z_current,
            'z_next_pred': z_next_pred,
            'x_current_recon': x_current_recon,
            'x_next_pred': x_next_pred
        }

        # 如果提供了下一时刻真值，计算所有损失
        if x_next is not None:
            # 重构损失 (确保自编码器有效)
            recon_loss = nn.functional.mse_loss(x_current_recon, x_current)
            # 预测损失 (确保模型能预测未来)
            pred_loss = nn.functional.mse_loss(x_next_pred, x_next)
            # 线性动力学损失 (强制Koopman空间中的线性关系)
            z_next_true = self.encode(x_next)
            linear_loss = nn.functional.mse_loss(z_next_pred, z_next_true)

            outputs['recon_loss'] = recon_loss
            outputs['pred_loss'] = pred_loss
            outputs['linear_loss'] = linear_loss
            # 总损失 (可加权)
            outputs['total_loss'] = recon_loss + pred_loss + 0.1 * linear_loss

        return outputs


# ==================== 2. 数据准备与训练循环 ====================
def prepare_simulated_data(num_samples=10000, state_dim=5):
    """
    生成模拟的商用车CACC数据用于演示。
    实际应用中应替换为真实或高保真仿真数据。
    """
    # 模拟状态: [自车速度, 自车加速度, 相对距离, 相对速度, 前车加速度]
    time = np.linspace(0, 10, num_samples)
    ego_speed = 20 + 2 * np.sin(0.5 * time)  # 自车速度波动
    ego_acc = np.gradient(ego_speed)  # 数值微分得加速度
    rel_distance = 30 + 5 * np.sin(0.3 * time + 1)  # 相对距离波动
    rel_speed = np.gradient(rel_distance)  # 相对速度
    lead_acc = 0.5 * np.sin(0.7 * time)  # 前车加速度

    # 堆叠状态并添加噪声
    states = np.column_stack([ego_speed, ego_acc, rel_distance, rel_speed, lead_acc])
    states += np.random.normal(0, 0.01, states.shape)  # 添加轻微噪声

    # 创建当前状态-下一时刻状态对
    X_current = states[:-1]
    X_next = states[1:]

    return torch.FloatTensor(X_current), torch.FloatTensor(X_next)


def train_koopman_model(model, train_data, val_data, epochs=500, lr=1e-3):
    """训练深度Koopman模型"""
    X_train, Y_train = train_data
    X_val, Y_val = val_data

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播并计算损失
        outputs = model(X_train, Y_train)
        loss = outputs['total_loss']

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, Y_val)
            val_loss = val_outputs['total_loss']

        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return train_losses, val_losses


# ==================== 3. 主程序与模型集成示例 ====================
def main():
    print("=== 商用车CACC深度Koopman模型训练 ===")

    # 参数配置
    STATE_DIM = 5  # 根据你的状态定义调整
    KOOPMAN_DIM = 32  # 升维后的维度
    HIDDEN_DIMS = [64, 128, 64]  # 编码器/解码器隐藏层

    # 1. 准备数据 (这里使用模拟数据)
    print("1. 准备模拟数据...")
    X, Y = prepare_simulated_data(num_samples=5000)

    # 分割训练集和验证集
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    # 2. 初始化模型
    print("2. 初始化深度Koopman模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepKoopmanModel(STATE_DIM, KOOPMAN_DIM, HIDDEN_DIMS).to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Koopman矩阵维度: {model.koopman_matrix.weight.shape}")

    # 3. 训练模型
    print("\n3. 开始训练...")
    train_losses, val_losses = train_koopman_model(
        model, (X_train, Y_train), (X_val, Y_val),
        epochs=300, lr=1e-3
    )

    # 4. 可视化训练过程
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Deep Koopman Model Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. 演示模型在MPC中的使用 (伪代码)
    print("\n4. 模型使用示例: 集成到MPC预测器中")
    model.eval()
    with torch.no_grad():
        # 假设当前状态
        current_state = X_val[0:1]  # shape: (1, state_dim)

        # 编码到Koopman空间
        z = model.encode(current_state)

        # 在MPC的预测时域内进行高效的线性预测
        prediction_horizon = 10
        predicted_states = []
        for _ in range(prediction_horizon):
            z = model.koopman_operation(z)  # 线性变换!
            x_pred = model.decode(z)
            predicted_states.append(x_pred.cpu().numpy())

        print(f"  完成 {prediction_horizon} 步预测，用于MPC优化。")

    # 6. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': STATE_DIM,
        'koopman_dim': KOOPMAN_DIM,
        'hidden_dims': HIDDEN_DIMS
    }, 'deep_koopman_cacc_model.pth')
    print("模型已保存为 'deep_koopman_cacc_model.pth'")

    return model


if __name__ == "__main__":
    model = main()