import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ==============================
# 1. Koopman 模型定义
# ==============================
class DeepKoopman(nn.Module):
    def __init__(self, n_x=5, n_u=2, n_k=48, n_hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_x+n_u, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_k)
        )
        self.K = nn.Parameter(torch.randn(n_k,n_k)*0.1)
        self.L = nn.Parameter(torch.randn(n_k,n_u)*0.1)
        self.decoder = nn.Sequential(
            nn.Linear(n_k,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_hidden), nn.ReLU(),
            nn.Linear(n_hidden,n_x)
        )

    def forward(self, x, u):
        z = self.encoder(torch.cat([x,u], dim=-1))
        z_next = z @ self.K.T + u @ self.L.T
        return self.decoder(z_next)

# ==============================
# 2. 加载模型（请改成你的路径）
# ==============================
DEVICE = "cpu"
model = DeepKoopman()
model.load_state_dict(torch.load("../_output/_train/best_koopman_model.pth", map_location=DEVICE))
model.eval()

# ==============================
# 3. 归一化参数
# ==============================
norm = {"px_mean": -23702.4, "px_std":14555.7, "py_mean":3109.56, "py_std":1841.54}

def normalize(x):
    x = x.copy()
    x[0] = (x[0]-norm["px_mean"])/norm["px_std"]
    x[1] = (x[1]-norm["py_mean"])/norm["py_std"]
    return x

def denormalize(x):
    x = x.copy()
    x[0] = x[0]*norm["px_std"] + norm["px_mean"]
    x[1] = x[1]*norm["py_std"] + norm["py_mean"]
    return x

# ==============================
# 4. Koopman 单步预测
# ==============================
def koopman_step(x, u):
    x_t = torch.tensor(normalize(x), dtype=torch.float32).unsqueeze(0)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        x_next = model(x_t, u_t).squeeze(0).numpy()
    x_next = np.clip(x_next, -1e5, 1e5)  # 防止溢出
    return denormalize(x_next)

# ==============================
# 5. 椭圆赛道
# ==============================
def generate_ellipse(a=20000,b=3000,N=800):
    t = np.linspace(0, 2*np.pi, N)
    x = a*np.cos(t)
    y = b*np.sin(t)
    psi = np.arctan2(np.gradient(y), np.gradient(x))
    return x, y, psi

ref_x, ref_y, ref_psi = generate_ellipse()

# ==============================
# 6. 简单控制器（平滑）
# ==============================
def controller(x, ref):
    best_u = np.array([0.0, 0.0])
    min_cost = 1e9
    for a in np.linspace(-0.05,0.05,3):
        for delta in np.linspace(-0.01,0.01,5):
            u = np.array([a, delta])
            try: x_pred = koopman_step(x, u)
            except: continue
            pos_err = np.linalg.norm(x_pred[:2]-ref[:2])
            psi_err = abs(x_pred[3]-ref[2])
            cost = pos_err + 0.3*psi_err
            if cost < min_cost:
                min_cost = cost
                best_u = u
    return best_u

# ==============================
# 7. 初始化状态
# ==============================
x = np.array([norm["px_mean"], norm["py_mean"], 31.6, 0.0, 0.0])
traj = []

# ==============================
# 8. 可视化缩放
# ==============================
VISUAL_SCALE = 1/100  # 缩小到可视范围

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(ref_x*VISUAL_SCALE, ref_y*VISUAL_SCALE, '--', color='gray', label='Reference')
traj_line, = ax.plot([], [], 'b-', lw=2)
car_patch = Rectangle((0,0), 500*VISUAL_SCALE, 200*VISUAL_SCALE, fc='red')
ax.add_patch(car_patch)
ax.set_xlim((ref_x.min()-1000)*VISUAL_SCALE, (ref_x.max()+1000)*VISUAL_SCALE)
ax.set_ylim((ref_y.min()-500)*VISUAL_SCALE, (ref_y.max()+500)*VISUAL_SCALE)
ax.set_aspect('equal')
ax.set_title("Koopman Racing Simulation")
ax.legend()

# ==============================
# 9. 动画更新函数
# ==============================
def update(frame):
    global x
    idx = frame % len(ref_x)
    ref = np.array([ref_x[idx], ref_y[idx], ref_psi[idx]])
    u = controller(x, ref)
    x = koopman_step(x, u)
    traj.append(x.copy())
    traj_np = np.array(traj)
    traj_line.set_data(traj_np[:,0]*VISUAL_SCALE, traj_np[:,1]*VISUAL_SCALE)

    car_patch.set_width(500*VISUAL_SCALE)
    car_patch.set_height(200*VISUAL_SCALE)
    car_patch.set_xy((x[0]-250)*VISUAL_SCALE, (x[1]-100)*VISUAL_SCALE)
    car_patch.angle = np.degrees(x[3])
    return traj_line, car_patch

ani = FuncAnimation(fig, update, frames=500, interval=50)
plt.show()
