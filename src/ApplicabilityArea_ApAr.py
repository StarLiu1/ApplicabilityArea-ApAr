import numpy as np
import sympy as sy
import matplotlib as plt
def treatNone(x, uFN, uTN):
    return x * uFN + (1-x) * uTN
def test(x, sensitivity, specificity, uTN, uTP, uFN, uFP, u):
    return x * sensitivity * uTP + x * (1-sensitivity) * uFN + (1-x) * (1-specificity) * uFP + (1-x) * specificity * uTN + u
def treatAll(x, uFP, uTP):
    return x * uTP + (1-x) * uFP

def pLpStarpUThresholds(s1, s2, uTN, uTP, uFN, uFP, u):
    """
    Returns the pL, pStar, and pU thresholds
    """
    x = sy.symbols('x')
    pU = sy.solve(treatAll(x, uFP, uTP) - test(x, s1, s2, uTN, uTP, uFN, uFP, u), x)
    pStar = sy.solve(treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN), x)
    pL = sy.solve(treatNone(x, uFN, uTN) - test(x, s1, s2, uTN, uTP, uFN, uFP, u), x)
    pU = -999 if (len(pU) == 0) else float(pU[0])
    pU = 1 if (pU > 1) else pU
    pU = 0 if ((pU < 0) & (pU != -999)) else pU
    pStar = -999 if (len(pStar) == 0) else float(pStar[0])
    pStar = 1 if (pStar > 1) else pStar
    pStar = 0 if ((pStar < 0) & (pStar != -999)) else pStar
    pL = -999 if (len(pL) == 0) else float(pL[0])
    pL = 1 if (pL > 1) else pL
    pL = 0 if ((pL < 0) & (pL != -999)) else pL

    return [pL, pStar, pU]

def expectedGainUtil(s1, s2, uTN, uTP, uFN, uFP, u, graph):
    x = sy.symbols('x')
    pU = sy.solve(treatAll(x, uFP, uTP) - test(x, s1, s2, uTN, uTP, uFN, uFP, u), x)
    pStar = sy.solve(treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN), x)
    pL = sy.solve(treatNone(x, uFN, uTN) - test(x, s1, s2, uTN, uTP, uFN, uFP, u), x)
    if(pStar > pU):
        result = 0
    else:
        result = sy.integrate(x * (test(x, s1, s2, uTN, uTP, uFN, uFP, u) - treatNone(x, uFN, uTN)), (x, pL, pStar)) + \
                sy.integrate(x * (test(x, s1, s2, uTN, uTP, uFN, uFP, u) - treatAll(x, uFP, uTP)), (x, pStar, pU))
    
    if(graph == True):
        X = np.array(range(2))
        plt.plot(X,treatNone(X, uFN, uTN), label = "Treat None")
        plt.plot(X,test(X, s1, s2, uTN, uTP, uFN, uFP, u), label = "Test")
        plt.plot(X,treatAll(X, uFP, uTP), label = "Treat All")     
        plt.legend(prop={'size':13}, loc='lower right')
        plt.show()

    pU = -999 if (len(pU) == 0) else float(pU[0])
    pU = 1 if (pU > 1) else pU
    pU = 0 if ((pU < 0) & (pU != -999)) else pU
    pStar = -999 if (len(pStar) == 0) else float(pStar[0])
    pStar = 1 if (pStar > 1) else pStar
    pStar = 0 if ((pStar < 0) & (pStar != -999)) else pStar
    pL = -999 if (len(pL) == 0) else float(pL[0])
    pL = 1 if (pL > 1) else pL
    pL = 0 if ((pL < 0) & (pL != -999)) else pL

    return [result, pL, pStar, pU]

def modelPriorsOverRoc(modelChosen, uTN, uTP, uFN, uFP, u):
    pLs = []
    pStars = []
    pUs = []
    #for each pair of tpr, fpr
    if(type(np.array(modelChosen['tpr'])[0]) == list):
        tprArray = np.array(np.array(modelChosen['tpr'])[0])
        fprArray = np.array(np.array(modelChosen['fpr'])[0])
    else:
        tprArray = np.array(modelChosen['tpr'])
        fprArray = np.array(modelChosen['fpr'])
    if(tprArray.size > 1):
        for cutoffIndex in range(0, tprArray.size):
            tpr = tprArray[cutoffIndex]
            fpr = fprArray[cutoffIndex]
            pL, pStar, pU = pLpStarpUThresholds(tpr, 1 - fpr, uTN, uTP, uFN, uFP, u)
            pLs.append(pL)
            pStars.append(pStar)
            pUs.append(pU)
        return [pLs, pStars, pUs]
    else:
        return [[0], [0], [0]]
    
def priorFiller(priorList, lower: bool):
    """
    some priors are not defined. For those -999, change to 1 or 0 depending on pL or pU
    """
    if(lower == True):
        for index, item in enumerate(priorList):
            lenList = len(priorList)
            midPoint = lenList/2
            if((index < midPoint) & (lenList > 1)):
                if(item == -999):
                    priorList[index] = 1
            if((index > midPoint) & (lenList > 1)):
                if(item == -999):
                    priorList[index] = 0
    else:
        for index, item in enumerate(priorList):
            lenList = len(priorList)
            midPoint = lenList/2
            if((index < midPoint) & (lenList > 1)):
                if(item == -999):
                    priorList[index] = 0
            if((index > midPoint) & (lenList > 1)):
                if(item == -999):
                    priorList[index] = 0

def priorModifier(priorList):
    """
    some priors are not defined. For those -999, change to 1 or 0 depending on pL or pU
    """
#     if(lower == True):
    for index, item in enumerate(priorList):
        lenList = len(priorList)
        midPoint = lenList/2
        if((index < midPoint) & (lenList > 1)):
            if((item == 1) & (priorList[index + 2] > priorList[index + 1]) & (priorList[index + 3] > priorList[index + 2])):
                priorList[index] = 0
            elif((item == 0) & (priorList[index + 2] < priorList[index + 1]) & (priorList[index + 3] < priorList[index + 2])):
                priorList[index] = 1
        if((index > midPoint) & (lenList > 1)):
            if((item == 1) & (priorList[index - 2] > priorList[index - 1]) & (priorList[index - 3] > priorList[index - 2])):
                priorList[index] = 0
            elif((item == 0) & (priorList[index - 2] < priorList[index - 1]) & (priorList[index - 3] < priorList[index - 2])):
                priorList[index] = 1
        if(index == lenList -1):
            if((priorList[index - 1] != 0) & (priorList[index] == 0)):
                priorList[index] = priorList[index - 1]

def applicableArea(modelRow, thresholds, utils, p):
    uTN, uTP, uFN, uFP, u = utils
    #calculate pLs, pStars, and pUs
#     print(modelRow)
    uFP = uTN - (uTP - uFN) * (1/modelRow['costRatio'])
    pLs, pStars, pUs = modelPriorsOverRoc(modelRow, uTN, uTP, uFN, uFP, u)
    #extract thresholds
    thresholds = np.array(thresholds)
    thresholds = np.where(thresholds > 1, 1, thresholds)
    #fill in undefined priors
    priorFiller(pLs, True)
    priorModifier(pLs)
    priorFiller(pUs, False)
    priorModifier(pUs)
    area = 0
    largestRangePrior = 0
    largestRangePriorThresholdIndex = -999
    withinRange = False
    leastViable = 1
#     if((pLs is None) == False):
    for i, prior in enumerate(pLs): 
        if(i < len(pLs) - 1):
            if((pLs[i] < pUs[i]) | (pLs[i + 1] < pUs[i + 1])):
                if((leastViable > pLs[i]) & (pLs[i] > 0)):
                    leastViable = pLs[i]
                ##check if input prior is within this range of priors
                if((p > pLs[i]) & (p < pUs[i])):
                    withinRange = True
                ### To add
                #first diff + second diff / 2 if the 2nd difference is still positive and its not the last one: trapezoidal rule
                rangePrior = pUs[i] - pLs[i]
                if(rangePrior > largestRangePrior):
                    largestRangePrior = rangePrior
                    largestRangePriorThresholdIndex = i
                avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2 #trapezoidal rule (upper + lower base)/2
                area = area + abs(avgRangePrior) * abs(thresholds[i + 1] - thresholds[i])
                area = np.round(area, 3)
    if(area > 1):
        area = 1
    return [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]