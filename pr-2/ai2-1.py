import scipy as sc
import numpy as np


# Функция для выбоки n строк из данной матрицы
def get_n_str(matrix, n_str):
    new_matrix = []
    for p in range(n_str):
        new_matrix.append(matrix[p])
    return np.array(new_matrix)


# Берем датасет 'пользователь-фильм'
user_movie_matrix = np.array(((1, 5, 0, 5, 4), (5, 4, 4, 3, 2), (0, 4, 0, 0, 5), (4, 4, 1, 4, 0),
                              (0, 4, 3, 5, 0), (2, 4, 3, 5, 3)))

# Сингулярное разложение матрицы на u - матрицу-представление пользователей, s диагональ сингулярных чисел, м - матрица-представление фильмов
u, s, v = sc.linalg.svd(user_movie_matrix)

# Извлечение первых двух строк из матричных представлений
n = 2
u_new = get_n_str(u, n)
s_new = get_n_str(s, n)
v_new = get_n_str(v, n)

print('n: ', n)
print('U: \n', u_new)
print('s: \n', s_new)
print('V: \n', v_new)

vt = v_new.T
print('v trans: \n', vt)

# Представление пользователя сниженной размерности
low_dimension = np.matmul(user_movie_matrix[2], vt)
# Обратная трансформация ветора в вектор оценок фильмов
inversed_transformation = np.matmul(low_dimension, v_new)

non_watched_max_value = -277353
index = -277353
for i in range(len(inversed_transformation)):
    if user_movie_matrix[2][i] == 0 and inversed_transformation[i] > non_watched_max_value:
        non_watched_max_value = inversed_transformation[i]
        index = i

rate = inversed_transformation[index]
max_index = index  # Индекс непросмотренного фильма с наибольшей оценкой
print('\nrate: ', inversed_transformation, '\nmax_rate: ', rate, '\nfilm_index: ', max_index)

# Та же самая ситуация, только с новым пользователем
new_user = np.array((0, 0, 3, 4, 0))
vt = v_new.T

print('v trans: \n', vt)
print('n: ', n)
print('u: \n', u_new)
print('s: \n', s_new)
print('v: \n', v_new)

low_dimension = np.matmul(new_user, vt)
inversed_transformation = np.matmul(low_dimension, v_new)
for i in range(len(new_user)):
    if new_user[i] == 0 and inversed_transformation[i] > non_watched_max_value:
        non_watched_max_value = new_user[i]
        index = i

rate = inversed_transformation[index]
max_index = index
print('\nrate: ', inversed_transformation, '\nmax_rate: ', rate, '\nfilm_index: ', max_index)

# Та же самая ситуация, только с использованием трех компонент
n = 3
u_new = get_n_str(u, n)
s_new = get_n_str(s, n)
v_new = get_n_str(v, n)

print('n: ', n)
print('u: \n', u_new)
print('s: \n', s_new)
print('v: \n', v_new)
vt = v_new.T
print('v trans: \n', vt)

low_dimension = np.matmul(user_movie_matrix[2], vt)
inversed_transformation = np.matmul(low_dimension, v_new)

for i in range(len(user_movie_matrix[2])):
    if user_movie_matrix[2][i] == 0 and inversed_transformation[i] > non_watched_max_value:
        non_watched_max_value = user_movie_matrix[2][i]
        index = i

rate = inversed_transformation[index]
max_index = index

print('\nrate: ', inversed_transformation, '\nmax_rate: ', rate, '\nfilm_index: ', max_index)
