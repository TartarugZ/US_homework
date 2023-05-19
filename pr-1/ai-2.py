import pandas as pd
import numpy as np
import matplotlib.pyplot as graph
from joblib.numpy_pickle_utils import xrange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Берем датасет
dataset = pd.read_csv('https://raw.githubusercontent.com/sdukshis/ml-intro/master/datasets/Davis.csv', index_col=0)
# Удаляем строки, где есть nan
dataset = pd.DataFrame.dropna(dataset)
dataset.head()

# Отбираем х(вес) - признак, у(рост) - целевая переменная
X = dataset.iloc[:, 1:-3].values
y = dataset.iloc[:, 2:-2].values

# Делим датасет 80 тренировочных 20 тестировочных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Линейная регресия
model = LinearRegression()
model.fit(X_train, y_train.reshape(-1, 1))

# Изображение исходных данных
dataset.head()
dataset.describe()
dataset.plot(x='weight', y='height', style='yo')
graph.title('Dataset')
graph.xlabel('weight')
graph.ylabel('height')
graph.show()

# Предсказываем значения и выводим среднюю ошибку
y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)

# Выводим прямую регрессии
graph.plot(X_train, y_train, 'bo', label='Train', alpha=0.2)
graph.plot(X_test, y_test, 'rx', label='Test', alpha=1)
line = np.arange(40, 120).reshape(-1, 1)
graph.plot(line, model.predict(line), 'g-')
graph.title('Result regression')
graph.xlabel('weight')
graph.ylabel('height')
graph.legend()
graph.show()

# Отбираем х(пол, вес, repwt) - признаки, у(рост) - целевая переменная
y = dataset.iloc[:, 2:-2].values
dataset = dataset.drop(['height', 'repht'], axis=1)
X = dataset.iloc[:, :].values
# Заменяем категориальные признаки на числовые
for i in X:
    if i[0] == 'M':
        i[0] = 1
    elif i[0] == 'F':
        i[0] = 0

# Делим датасет 80 тренировочных 20 тестировочных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train.reshape(-1, 1))

# Выводим среднюю ошибку
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('\nDifference:')
mae_2 = metrics.mean_absolute_error(y_test, y_pred)
mse_2 = metrics.mean_squared_error(y_test, y_pred)

# Выводим разницу в ошибке между первой и второй моделью
dif_mae = mae - mae_2
dif_mse = mse - mse_2
print('Mean Absolute Error difference:', dif_mae)
print('Mean Squared Error difference:', dif_mse)
