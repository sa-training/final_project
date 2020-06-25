# -*- coding: utf-8 -*-
"""
HISTORY:
    Created on Mon Jun 21 11:10:16 2020

Project: FINAL PROJECT FOR OTUS COURSE

Author: Shustov Aleksei (SemperAnte), semperante@mail.ru        
 
TODO:

DESCRIPTION:
"""

def print_coefs(coefs):
    print('y(x) = ', end = '')
    n = len(coefs)
    for i in range(n):        
        print(f'{coefs[i]:.6f} * x^{n - i - 1}', end = '')        
        if i < n - 1:
            print(' + ', end = '')
        else:
            print()
            
def mse_error(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i][0] - y[i][0]) ** 2
    return sum / len(x)