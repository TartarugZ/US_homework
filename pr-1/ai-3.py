from numpy import savetxt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Берем датасет
dataset = pd.read_csv('mnist.csv')
dataset = pd.DataFrame.dropna(dataset)
dataset.head()

# Берем х - признаки(цвета пикселей), у - ответ(цифра)
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, :1].values

# Разбиваем датасет на 70 тренировочных и 30 тестировочных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Модель Дерево решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказываем значение и выводим среднюю ошибку, матрицу ошибок, точность, полноту и f-меру
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

savetxt('tree_answer.csv', model.predict(X_test), delimiter=',', fmt='%d')
print('accuracy score:', metrics.accuracy_score(y_test, y_pred))
