"""
HISTORY:
    Created on Mon Jun 18 17:48:12 2020

Project: FINAL PROJECT FOR OTUS COURSE

Author: Shustov Aleksei (SemperAnte), semperante@mail.ru        
 
TODO:

DESCRIPTION:
    Базовая библиотека для построения линейной регрессии методом наименьших квадратов
    
"""

import matrixNoLib as mnl

class LinearRegression():
    """ Построение линейной регрессии """
    
    def __init__(self):
        self.coefs = None

    def fit(self, x, y):
        """ Вычисление коэффициентов """

        xTranspose = mnl.T(x)
        a = mnl.multiply(xTranspose, x)
        b = mnl.multiply(xTranspose, y)
        self.coefs = mnl.solve(a, b)

    def predict(self, x):
        """ Построение прогноза для входных данных x """

        return mnl.multiply(x, self.coefs)
    
    def normalize(self, x):
        """ Нормирование входных данных """
        
        nums = len(x)
        vars = len(x[0])

        xNorm = []
        for i in range(nums):
            xNorm.append([])
            for j in range(vars):
                xNorm[-1].append(x[i][j])

        for var in range(vars):
            total = 0
            sumOfSqrt = 0

            for num in range(nums):
                total += xNorm[num][var]

            if total == nums:
                mean = 0
            else:
                mean = total/nums

            for num in range(nums):
                xNorm[num][var] -= mean
                sumOfSqrt += xNorm[num][var]**2

            if sumOfSqrt == nums:
                lNorm = 1.0
            else:
                lNorm = sumOfSqrt**0.5

            for num in range(nums):
                xNorm[num][var] /= lNorm

        return xNorm

class PolynomialFeatures():
    """ Вычисление полниомиальных признаков """
    
    def __init__(self, degree, bias = True):
        """ degree - степень полинома, bias - использование смещения """       
        self.degree = degree
        self.bias = bias
        
    def fit(self, x):
        """ Вычисление набора степеней для входных значений """
        
        self.vars = len(x[0])
        self.powers = [0]*self.vars

        self._getPowersList(degree = self.degree, 
                                  var = 1, 
                                  powers = self.powers, 
                                  powerList = set())

        self.powerList.sort(reverse = True)

        self._modifyPowers()

    def getFeatureNames(self):

        defaultNames = []
        featureNames = []
        for powers in self.powerList:
            prod = []
            for i in range(len(defaultNames)):
                if powers[i] == 0:
                    continue
                elif powers[i] == 1:
                    val = defaultNames[i]
                else: 
                    val = defaultNames[i] + '^' + str(powers[i])
                prod.append(val)
            if prod == []:
                prod = ['1']
            featureNames.append(' '.join(prod))
        
        return featureNames

    def transform(self, x):
        """ Упорядочивание найденных степеней и входных значений """
        
        xout = []
        for row in x:
            temp = []
            for powers in self.powerList:
                prod = 1
                for i in range(len(row)):
                    prod *= row[i] ** powers[i]
                temp.append(prod)
            xout.append(temp)

        return xout
        
    def fit_transform(self, x):
        """ Операции в один шаг для соответствия питоновской библиотеки """   

        self.fit(x)
        return self.transform(x)

    def _getPowersList(self, degree, var = 1, powers = [0, 0], powerList = set()):
        
        for pow in range(degree + 1):
            powers[var - 1] = pow
            if sum(powers) <= degree:
                powerList.add(tuple(powers))
            if var < self.vars:
                self._getPowersList(degree = degree, 
                                    var = var + 1, 
                                    powers = powers, 
                                    powerList = powerList)
        self.powerList = [list(x) for x in powerList]

    def _modifyPowers(self):
        """ Удалить степень, если не требуется учитывать смещение """
    
        if self.bias == False:
            try:
                self.powerList.remove([0] * self.vars)
            except:
                pass