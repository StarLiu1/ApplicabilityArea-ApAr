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
    return priorList
                    
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
    return priorList

def extractThresholds(row):
    """
    Forgot to save the thresholds as a single column. 
    Thus, this extracts the thresholds and adjusts those outside the [0,1] range.
    """
    thresholds = row['thresholds']
    for i, cutoff in enumerate(thresholds):
        if(cutoff > 1):
            thresholds[i] = 1
    return thresholds

def adjustpLpUClassificationThreshold(thresholds, pLs, pUs):
    pLs = priorFiller(pLs, True)
    pLs = priorModifier(pLs)
    pUs = priorFiller(pUs, False)
    pUs = priorModifier(pUs)
    thresholds = np.array(thresholds)
    thresholds = np.where(thresholds > 1, 1, thresholds)
    if thresholds[-1] == 0:
        thresholds[-1] == 0.0001
        thresholds = np.append(thresholds, 0)
        pLs[0] = pLs[1]
        pUs[0] = pUs[1]
        pLs = np.append([0], pLs)
        pUs = np.append([0], pUs)
    thresholds = thresholds[::-1]
    return [thresholds, pLs, pUs]

def applicableArea(modelRow, thresholds, utils, p):
    uTN, uTP, uFN, uFP, u = utils
    area = 0
    largestRangePrior = 0
    largestRangePriorThresholdIndex = -999
    withinRange = False
    priorDistributionArray = []
    leastViable = 1
    minPrior = 0
    maxPrior = 0
    meanPrior = 0
    
    #calculate pLs, pStars, and pUs
    uFP = uTN - (uTP - uFN) * (1 / modelRow['costRatio'])
    pLs, pStars, pUs = modelPriorsOverRoc(modelRow, uTN, uTP, uFN, uFP, u)
    thresholds = np.array(thresholds)
    thresholds = np.where(thresholds > 1, 1, thresholds)
    thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
    
    for i, prior in enumerate(pLs):
        if i < len(pLs) - 1:
            if pLs[i] < pUs[i] or pLs[i + 1] < pUs[i + 1]:
                rangePrior = pUs[i] - pLs[i]
                #incomplete: where pL and pU cross.
                #incomplete: where pL and pU cross.
                #incomplete: where pL and pU cross.
                if rangePrior > largestRangePrior:
                    largestRangePrior = rangePrior
                    largestRangePriorThresholdIndex = i
                avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2 # trapezoidal rule (upper + lower base)/2
                area += abs(avgRangePrior) * abs(thresholds[i + 1] - thresholds[i])
                area = np.round(area, 3)

    if(area > 1):
        area = 1           
    if((p > minPrior) & (p < maxPrior)):
        withinRange = True
                
    return [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]