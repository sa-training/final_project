# -*- coding: utf-8 -*-
"""
HISTORY:
    Created on Mon Jun 15 10:33:44 2020

Project: FINAL PROJECT FOR OTUS COURSE

Author: Shustov Aleksei (SemperAnte), semperante@mail.ru        
 
TODO:

DESCRIPTION:
    Проектная работа "Реализация линейной регрессии методом наименьших квадратов"
    
    - реализуем предсказание стоимости квартиры по общей площади квартиры
"""
import sys
sys.path.append('src')

# разработанные библиотеки линейной регресии и матричной алгебры
import matrixNoLib as mnl
import LinearRegressionNoLib as lrnl

import matplotlib.pyplot as plt
import utils
import csv

# Пример 1. Сгенерируем случайные данные на основе полиномиальной функции 3-й степени с добавлением шума
square = []
price = []
with open('appartments.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            square.append(float(row[0]))
            price.append(float(row[1]))

square = mnl.T([square])
price = mnl.T([price])
train_square = square[::10]
train_price = price[::10]

# Построение линейной регресии с помощью разработанных библиотек
degree = 2
plt.plot(square, price, '.r')

poly_own = lrnl.PolynomialFeatures(degree = degree)
xnew_own = poly_own.fit_transform(train_square)
linReg_own = lrnl.LinearRegression()
linReg_own.fit(xnew_own, train_price)

# Сравнение коэффициентов
coefs_own = linReg_own.coefs
coefs_own = [i[0] for i in coefs_own]
print('Задающий полином на основе собственной библиотеки:')
utils.print_coefs(coefs_own)

# тестовая последовательность
test_square = square
check_price = price
# # предсказание на основе собственной библиотеки
x = poly_own.transform(test_square)
test_price = linReg_own.predict(x)
mse = utils.mse_error(test_price, check_price)
print(f'Ошибка MSE = {mse}')

plt.plot(mnl.T(test_square)[0], mnl.T(test_price)[0])
plt.xlabel('Общая площадь квартиры, м2')
plt.ylabel('Стоимость квартиры, руб.')
plt.title(f'Предсказание стоимости квартиры по общей площади')
plt.grid('on')
plt.show()