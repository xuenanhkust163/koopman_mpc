import numpy as np
import scipy.io as sio
import pandas as pd
import json
import os

def convert_3d_to_5d(mat_file_path, normalize=True):
    """
    将萨格勒布数据集的 3 维状态转换为 5 维状态，并可选择对 px, py 进行标准化。
    状态顺序: [px, py, v, psi, omega]
    控制顺序: [a, delta]

    参数:
        mat_file_path: .mat 文件路径
        normalize: bool, 是否对 px, py 做标准化 (减去均值除以标准差)

    返回:
        X_t: 当前时刻状态 (N-1, 5)
        U_t: 当前时刻控制 (N-1, 2)
        X_t1: 下一时刻状态 (N-1, 5)
        Ts: 采样时间
        norm_params: dict or None, 包含 'px_mean', 'px_std', 'py_mean', 'py_std'
    """
    # 车辆参数
    m = 1752.0
    Cyf = 51488.0

    data = sio.loadmat(mat_file_path)
    x = data['x']      # (3, N)  [vx; vy; omega]
    u = data['u']      # (3, N)  [Fxf; Fyf; Fxr]
    Ts = data['Ts'][0][0]

    vx = x[0, :]
    vy = x[1, :]
    omega = x[2, :]

    Fxf = u[0, :]
    Fyf = u[1, :]
    Fxr = u[2, :]

    N = len(vx)
    print(f"数据加载完成: {N} 个时间步, Ts = {Ts} s")

    # 计算航向角 psi（积分 omega）
    psi = np.cumsum(omega) * Ts

    # 计算位置 px, py（绝对坐标）
    px = np.zeros(N)
    py = np.zeros(N)
    for i in range(1, N):
        px[i] = px[i-1] + (vx[i-1] * np.cos(psi[i-1]) - vy[i-1] * np.sin(psi[i-1])) * Ts
        py[i] = py[i-1] + (vx[i-1] * np.sin(psi[i-1]) + vy[i-1] * np.cos(psi[i-1])) * Ts

    # 控制输入转换
    a = (Fxf + Fxr) / m                     # 加速度
    delta = np.arctan(Fyf / Cyf)            # 转向角
    delta = np.clip(delta, -np.pi/4, np.pi/4)

    # 构造状态矩阵 X: [px, py, v, psi, omega]
    v = np.sqrt(vx**2 + vy**2)              # 合速度
    X = np.column_stack([px, py, v, psi, omega])
    U = np.column_stack([a, delta])

    # 归一化处理（仅对 px, py）
    norm_params = None
    if normalize:
        px_mean = np.mean(px)
        px_std = np.std(px)
        py_mean = np.mean(py)
        py_std = np.std(py)
        # 避免除零
        px_std = px_std if px_std > 1e-8 else 1.0
        py_std = py_std if py_std > 1e-8 else 1.0
        norm_params = {
            'px_mean': px_mean, 'px_std': px_std,
            'py_mean': py_mean, 'py_std': py_std
        }
        # 修改 X 中的 px, py 为标准化后的值
        X[:, 0] = (X[:, 0] - px_mean) / px_std
        X[:, 1] = (X[:, 1] - py_mean) / py_std
        print(f"归一化参数: px_mean={px_mean:.2f}, px_std={px_std:.2f}, py_mean={py_mean:.2f}, py_std={py_std:.2f}")
    else:
        print("未启用归一化，直接使用原始坐标")

    # 生成训练样本
    X_t = X[:-1]
    U_t = U[:-1]
    X_t1 = X[1:]

    print(f"\n生成 {len(X_t)} 个训练样本")
    print(f"  X_t shape: {X_t.shape} = [px, py, v, psi, omega]")
    print(f"  U_t shape: {U_t.shape} = [a, delta]")

    return X_t, U_t, X_t1, Ts, norm_params


def save_to_readable_formats(X_t, U_t, X_t1, Ts, norm_params=None, prefix='training_data'):
    """
    保存数据为 CSV, NPZ, Excel，以及归一化参数（若提供）
    """
    n_samples = min(len(X_t), 1000)   # CSV 只保存前1000行

    # 构造 DataFrame（使用归一化后的 px, py）
    df = pd.DataFrame({
        'px_t': X_t[:n_samples, 0],
        'py_t': X_t[:n_samples, 1],
        'v_t': X_t[:n_samples, 2],
        'psi_t': X_t[:n_samples, 3],
        'omega_t': X_t[:n_samples, 4],
        'a_t': U_t[:n_samples, 0],
        'delta_t': U_t[:n_samples, 1],
        'px_t1': X_t1[:n_samples, 0],
        'py_t1': X_t1[:n_samples, 1],
        'v_t1': X_t1[:n_samples, 2],
        'psi_t1': X_t1[:n_samples, 3],
        'omega_t1': X_t1[:n_samples, 4],
    })

    # 保存 CSV
    csv_file = f'{prefix}.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✅ CSV 已保存: {csv_file} (前 {n_samples} 行)")

    # 保存完整 NPZ（包含原始数组，未裁剪）
    npz_file = f'{prefix}.npz'
    np.savez(npz_file, X_t=X_t, U_t=U_t, X_t1=X_t1, Ts=Ts)
    print(f"✅ NPZ 已保存: {npz_file}")

    # 保存归一化参数（如果提供）
    if norm_params is not None:
        # 保存为 JSON
        json_file = f'{prefix}_norm_params.json'
        with open(json_file, 'w') as f:
            json.dump(norm_params, f, indent=2)
        print(f"✅ 归一化参数 (JSON) 已保存: {json_file}")

        # 也保存为 NPZ（方便 Python 加载）
        npz_params_file = f'{prefix}_norm_params.npz'
        np.savez(npz_params_file, **norm_params)
        print(f"✅ 归一化参数 (NPZ) 已保存: {npz_params_file}")

    # 可选：保存 Excel
    try:
        excel_file = f'{prefix}.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"✅ Excel 已保存: {excel_file}")
    except ImportError:
        print("⚠️ 未安装 openpyxl，跳过 Excel 保存。")

    # 打印预览
    print(f"\n数据预览（前5行，归一化后的坐标）:")
    print(df.head())

    return df


if __name__ == "__main__":
    # 转换数据（默认启用归一化）
    X_t, U_t, X_t1, Ts, norm_params = convert_3d_to_5d('dataset.mat', normalize=True)

    # 保存为可读格式（同时保存归一化参数）
    df = save_to_readable_formats(X_t, U_t, X_t1, Ts, norm_params, '../_output/_data_process/training_data')

    # 打印统计信息（注意：这里打印的是归一化后的统计量）
    print(f"\n📊 统计信息 (归一化后):")
    print(f"  px: [{X_t[:, 0].min():.3f}, {X_t[:, 0].max():.3f}]")
    print(f"  py: [{X_t[:, 1].min():.3f}, {X_t[:, 1].max():.3f}]")
    print(f"  v:  [{X_t[:, 2].min():.1f}, {X_t[:, 2].max():.1f}] m/s")
    print(f"  psi: [{X_t[:, 3].min():.2f}, {X_t[:, 3].max():.2f}] rad")
    print(f"  omega: [{X_t[:, 4].min():.2f}, {X_t[:, 4].max():.2f}] rad/s")

    # 如需使用原始坐标（非归一化），可设置 normalize=False 重新运行
    # X_t_raw, U_t_raw, X_t1_raw, Ts_raw, _ = convert_3d_to_5d('dataset.mat', normalize=False)