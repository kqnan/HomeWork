import tensorflow as tf
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# 下载并加载USPS数据集
usps = fetch_openml('usps')
X = np.array(usps.data)
y = np.array(usps.target)

# 将像素值缩放到[0,1]范围内
X = X / 255.0

# 将数据集分为训练集和测试集
train_size = 5000
test_size = 1000
train_X = X[:train_size].reshape(-1, 16, 16, 1)
train_y = y[:train_size].astype(np.int32)
test_X = X[-test_size:].reshape(-1, 16, 16, 1)
test_y = y[-test_size:].astype(np.int32)

# 定义LeNet模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(16, 16, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation='relu'),
    tf.keras.layers.Dense(units=84, activation='relu'),
    tf.keras.layers.Dense(units=11, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y))

# 绘制训练过程中的loss和accuracy曲线
plt.plot(history.history['loss'], label='train_loss')
i,loss=list(enumerate(history.history['loss']))[-1]
plt.text(i, loss, str(round(loss, 4)), fontsize=8, ha='center', va='bottom')
plt.plot(history.history['val_loss'], label='val_loss')
i,loss=list(enumerate(history.history['val_loss']))[-1]
plt.text(i, loss, str(round(loss, 4)), fontsize=8, ha='center', va='bottom')
plt.plot(history.history['accuracy'], label='train_accuracy')
i,loss=list(enumerate(history.history['accuracy']))[-1]
plt.text(i, loss, str(round(loss, 4)), fontsize=8, ha='center', va='bottom')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
i,loss=list(enumerate(history.history['val_accuracy']))[-1]
plt.text(i, loss, str(round(loss, 4)), fontsize=8, ha='center', va='bottom')


plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
