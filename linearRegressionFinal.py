# -*- coding: utf-8 -*-
"""
HISTORY:
    Created on Mon Jun 15 10:33:44 2020

Project: FINAL PROJECT FOR OTUS COURSE

Author: Shustov Aleksei (SemperAnte), semperante@mail.ru        
 
TODO:

DESCRIPTION:
    Проектная работа "Реализация линейной регрессии методом наименьших квадратов"
"""
import sys
sys.path.append('src')

# разработанные библиотеки линейной регресии и матричной алгебры
import matrixNoLib as mnl
import LinearRegressionNoLib as lrnl

import matplotlib.pyplot as plt
import random
import utils

# Пример 1. Сгенерируем случайные данные на основе полиномиальной функции 3-й степени с добавлением шума.
in_length = 400 # количество точек
in_coef = [-0.7, -0.6, 0.9, 0.6]
in_x = [x / in_length for x in range(in_length)]
in_y = [x**3 * in_coef[0] + x**2 * in_coef[1] + x * in_coef[2] + in_coef[3] + 0.1 * random.gauss(0, 1) for x in in_x]
plt.plot(in_x, in_y, '.r')
in_x = mnl.T([in_x])
in_y = mnl.T([in_y])
train_x = in_x[::2]
train_y = in_y[::2]

# Построение линейной регресии с помощью разработанных библиотек
degree = 5
poly_own = lrnl.PolynomialFeatures(degree = degree)
xnew_own = poly_own.fit_transform(train_x)
linReg_own = lrnl.LinearRegression()
linReg_own.fit(xnew_own, train_y)

# Построение линейной регресии на основе библиотеки scikit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_scikit = PolynomialFeatures(degree = degree)
xnew_scikit = poly_scikit.fit_transform(train_x)
linReg_scikit = LinearRegression() 
linReg_scikit.fit(xnew_scikit, train_y)

# Сравнение коэффициентов
coefs_own = linReg_own.coefs
coefs_own = [i[0] for i in coefs_own]
print('Задающий полином на основе собственной библиотеки:')
utils.print_coefs(coefs_own)
coefs_scikit = linReg_scikit.intercept_.tolist() + linReg_scikit.coef_[0][1:].tolist()
coefs_scikit = coefs_scikit[::-1]
print('Задающий полином на основе библиотеки scikit:')
utils.print_coefs(coefs_scikit)

# тестовая последовательность
test_x = in_x[1::2]
check_y = in_y[1::2]
# # предсказание на основе собственной библиотеки
x = poly_own.transform(test_x)
test_y = linReg_own.predict(x)
mse = utils.mse_error(test_y, check_y)
print(f'Ошибка MSE = {mse}')

# plot both methods
plt.plot(mnl.T(test_x)[0], mnl.T(test_y)[0])
plt.xlabel('Значения X')
plt.ylabel('Значения Y')
plt.title(f'Линейная регрессия - степень полинома {degree}')
plt.show()