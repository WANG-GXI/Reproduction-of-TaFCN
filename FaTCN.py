import numpy as np
from tensorflow.keras import layers, models, activations, Input
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
scaler_feature = MinMaxScaler(feature_range=(0, 1))

# 绘制训练和验证的损失与准确性
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确性
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练准确性')
    plt.plot(history.history['val_mae'], label='验证准确性')
    plt.title('训练和验证准确性')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.show()

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(scaler_feature.inverse_transform(np.array(y_test).reshape(-1, 1)), label='实际值')
    plt.plot(scaler_feature.inverse_transform(np.array(y_pred).reshape(-1, 1)), label='预测值', linestyle='--')
    plt.title('实际值和预测值对比')
    plt.xlabel('样本')
    plt.ylabel('值')
    plt.legend()
    plt.show()

# 读取数据
data = pd.read_csv('D:\My pythoin\FaTCN_Microsoft Stock_forecast\Microsoft Dataset.csv')
data_result1 = np.array(data['Volume']).reshape(-1, 1)

# 检查数据形状
print("Data shape:", data_result1.shape)

# 归一化
data_result1 = scaler_feature.fit_transform(data_result1)

# 参数定义
num_filter1 = 32
num_filter2 = 64
num_filter3 = 128
kernel1_size = 3
kernel2_size = 3
kernel3_size = 3
sequence_length = 10  # 每个样本的时间步长度，根据实际调整
num_features = 1  # 1维 特征数量为1

# 创建训练数据和目标数据
X = []
y = []
for i in range(len(data_result1) - sequence_length):
    X.append(data_result1[i:i + sequence_length])
    y.append(data_result1[i + sequence_length])

# 将列表转换为 numpy 数组
X = np.array(X)
y = np.array(y)

# # 划分数据集、测试集、验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 检查数据形状
print("X shape:", X.shape)
print("y shape:", y.shape)

# 定义一维卷积神经网络模型

def FCN_model_1D(sequence_length, num_features, num_filter1, num_filter2, num_filter3, kernel1_size, kernel2_size,
                 kernel3_size):
    in0 = Input(shape=(sequence_length, num_features))  # 输入形状为 (sequence_length, num_features)

    # SE块
    # x = layers.GlobalAveragePooling1D()(in0)
    x = layers.GlobalAveragePooling1D()(in0)
    x = layers.Dense(int(x.shape[-1]), use_bias=False, activation=activations.relu)(x)
    kernel = layers.Dense(int(in0.shape[-1]), use_bias=False, activation=activations.hard_sigmoid)(x)
    begin_senet = layers.Multiply()([in0, kernel])  # 给通道加权重

    # 一维卷积层
    conv0 = layers.Conv1D(num_filter1, kernel1_size, strides=1, padding='same', name='layer_9')(begin_senet)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu', name='layer_8')(conv0)

    conv0 = layers.Conv1D(num_filter2, kernel2_size, strides=1, padding='same', name='layer_7')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu', name='layer_6')(conv0)

    conv0 = layers.Conv1D(num_filter3, kernel3_size, strides=1, padding='same', name='layer_5')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Activation('relu', name='layer_4')(conv0)

    conv0 = layers.GlobalAveragePooling1D(name='layer_3')(conv0)
    conv0 = layers.Dense(num_filter1, activation='relu', name='layer_2')(conv0)
    out = layers.Dense(1, activation='sigmoid', name='layer_1')(conv0)  # 使用 sigmoid 激活函数进行二分类

    model = models.Model(inputs=in0, outputs=[out])

    return model




# 以下代码用于出图
# 创建模型
model = FCN_model_1D(sequence_length, num_features, num_filter1, num_filter2, num_filter3, kernel1_size, kernel2_size, kernel3_size)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
# 预测
y_pred = model.predict(X_test)

plot_predictions(y_test, y_pred)
print("rmse:", sqrt(mean_squared_error(y_test, y_pred)))
# 调用绘图函数
plot_history(history)

