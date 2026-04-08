# koopman_mpc

本项目包含神经网络训练、车动力模型仿真。

## Dataset
来自萨格勒布大学论文的汽车动力原始数据集合（Matlab格式）
- dataset.mat
- errData.mat

### Installation:
```python
conda create -n koopman_dnn python=3.10
conda activate koopman_dnn
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### How to run
```数据预处理
python data_preprocess/data_preprocessor.py
```

```DNN训练
python koopman_dnn_train/dnn_train.py
```

```MPC仿真
未完成
```
