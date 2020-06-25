# -*- coding: utf-8 -*-
"""
HISTORY:
    Created on Mon Jun 16 14:21:06 2020

Project: FINAL PROJECT FOR OTUS COURSE

Author: Shustov Aleksei (SemperAnte), semperante@mail.ru        
 
TODO:

DESCRIPTION:
    Базовая библиотека для работы с матрицами:
        - генерация типовых матриц;
        - транспонирование матрицы;
        - суммирование матриц;
        - перемножение матриц;
        - нахождение определителя;
        - решение линейных уравнений вида A * X = B.
"""
def zeros(rows, cols):
    """ Генерация нулевой матрицы """
    
    y = list()
    for row in range(rows):
        y.append([0.0] * cols)
    return y

def ones(rows, cols):
    """ Генерация матрицы, состоящей из единиц """
    
    y = list()
    for row in range(rows):
        y.append([1.0] * cols)
    return y

def identity(size):
    """ Генерация единичной матрицы """
    
    y = zeros(size, size)
    for idx in range(size):
        y[idx][idx] = 1.0
    return y

def T(x):
    """ Транспонирование матрицы """
    
    rows = len(x)
    cols = len(x[0])
    y = zeros(cols, rows)
    for row in range(rows):
        for col in range(cols):
            y[col][row] = x[row][col]
    return y

def add(a, b):
    """ Суммирование матриц """
    
    aRows = len(a)
    aCols = len(a[0])
    bRows = len(b)
    bCols = len(b[0])

    assert (aRows == bRows and aCols == bCols), 'Number columns and rows are not equal for addition.'

    y = zeros(aRows, bCols)
    for row in range(aRows):
        for col in range(bCols):
            y[row][col] = a[row][col] + b[row][col]
    return y

def multiply(a, b):
    """ Перемножение матриц """
    
    aRows = len(a)
    aCols = len(a[0])
    bRows = len(b)
    bCols = len(b[0])

    assert aCols == bRows, 'Number columns and rows are not equal for multiplication.'

    y = zeros(aRows, bCols)
    for row in range(aRows):
        for col in range(bCols):
            sum = 0.0
            for col2 in range(aCols):
                sum += a[row][col2] * b[col2][col]
            y[row][col] = sum
    return y

def determinant(x, det = 0):
    """ Нахождение определителя матрицы """
    
    rows = len(x)
    cols = len(x[0])

    if rows == 2 and cols == 2:
        det = x[0][0] * x[1][1] - x[1][0] * x[0][1]
    else:
        for col in range(len(x)):
            xcopy = x.copy()
            xcopy = xcopy[1:] # remove first row
            h = len(xcopy)    
            for i in range(h):
                xcopy[i] = xcopy[i][0:col] + xcopy[i][col+1:] # remove column
    
            sign = (-1) ** (col % 2) # find sign for submatrix
            subDet = determinant(xcopy)
            det += sign * x[0][col] * subDet
    return det

def solve(a, b):
    """Решение линейных уравнений вида A * X = B"""
    
    rows = len(a)
    cols = len(a[0])
    assert rows == cols, 'Number columns and rows are not equal for solver.'

    acopy = a.copy()
    bcopy = b.copy()

    idx = list(range(rows))
    for diag in range(rows):
        if acopy[diag][diag] == 0:
            acopy[diag][diag] = 1.0e-17
        scale = 1.0 / acopy[diag][diag]
        # scale diag row for diag inverse
        for j in range(rows): # column loop
            acopy[diag][j] *= scale
        bcopy[diag][0] *= scale
        # calculate all rows except diag
        for i in idx[0:diag] + idx[diag+1:]: # skip
            crScaler = acopy[i][diag]
            for j in range(rows):
                acopy[i][j] = acopy[i][j] - crScaler * acopy[diag][j]
            bcopy[i][0] = bcopy[i][0] - crScaler * bcopy[diag][0]

    return bcopy