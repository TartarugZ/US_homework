from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as graph
import matplotlib.patches as mpatches
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=(0.15, 0.85),
    class_sep=6.0,
    hypercube=False,
    random_state=2,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Графика датасета
graph.plot(X_train, y_train, 'bo', label='Train', alpha=0.5)
graph.plot(X_test, y_test, 'rx', label='Test', alpha=1)
graph.legend()
graph.show()

# Логистическая регрессия
model = LogisticRegression()
model.fit(X_train, y_train)

# К-ближайших соседей
model2 = KNeighborsRegressor()
model2.fit(X_train, y_train)

# Предсказываем результаты для обеих моделей
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)

# Вывод: средняя ошибка, матрица ошибок, точность, полнота и F-мера для Логистической регрессии
mae_1 = metrics.mean_absolute_error(y_test, y_pred)
mse_1 = metrics.mean_squared_error(y_test, y_pred)
print('Mean Absolute Error:', mae_1)
print('Mean Squared Error:', mse_1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Вывод: средняя ошибка, матрица ошибок, точность, полнота и F-мера для К-ближайших соседей
mae_2 = metrics.mean_absolute_error(y_test, y_pred2)
mse_2 = metrics.mean_squared_error(y_test, y_pred2)
print('Mean Absolute Error:', mae_2)
print('Mean Squared Error:', mse_2)

# Предсказание дает нам дробные результаты, поэтому приводим их к 0 или 1
for j in range(len(y_pred2)):
    if y_pred2[j] < 0.5:
        y_pred2[j] = 0
    else:
        y_pred2[j] = 1

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

# Графики PR кривых
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
area = round(auc(recall, precision), 3)
_, ax = graph.subplots()
ax.plot(recall, precision, color='green')
green = mpatches.Patch(color='green', label=f'PR LogisticRegression: Average Precision = {area}')
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

precision, recall, _ = precision_recall_curve(y_test, y_pred2)
area = auc(recall, precision)
ax.plot(recall, precision, color='red')
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
red = mpatches.Patch(color='red', label=f'PR KNearestNeighbors: Average Precision = {area}')
ax.legend(handles=[red, green])
graph.show()


# Графики ROC кривых
y_pred_roc = model.predict(X_test)
fp, tp, _ = metrics.roc_curve(y_test, y_pred_roc)
roc_auc = round(metrics.roc_auc_score(y_test, y_pred), 3)
graph.plot(fp, tp, color='green')
graph.ylabel('True Positive Rate')
graph.xlabel('False Positive Rate')
green = mpatches.Patch(color='green', label=f'ROC LogisticRegression (ROC AUC = {roc_auc})')

y_pred_roc = model2.predict(X_test)
fp, tp, _ = metrics.roc_curve(y_test, y_pred_roc)
roc_auc = round(metrics.roc_auc_score(y_test, y_pred2), 3)
graph.plot(fp, tp, color='red')
graph.ylabel('True Positive Rate')
red = mpatches.Patch(color='red', label=f'ROC KNearestNeighbors (ROC AUC = {roc_auc})')

graph.plot([0, 1], [0, 1], color='b', linestyle='--')
graph.xlim([0.0, 1.0])
graph.ylim([0.0, 1.05])
graph.legend(handles=[red, green])
graph.show()

# Выводим разницу в ошибке между первой и второй моделью
dif_mae = mae_1 - mae_2
dif_mse = mse_1 - mse_2
print('Mean Absolute Error difference:', dif_mae)
print('Mean Squared Error difference:', dif_mse)
