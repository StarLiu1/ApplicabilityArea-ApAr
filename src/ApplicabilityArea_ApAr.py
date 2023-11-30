import numpy as np
import sympy as sy
import matplotlib as plt

""" Context

Binary classification problem with 3 options
    - Option treat all: treat all patients as if they have the target condition
    - Option treat none: treat none as if everyone is free of the target condition
    - Option test: test (ML model) everyone to identify those with and without the target condition accurately
    
    *theoretical basis can be found in Chapter 9 in the Medical Decision Making by Sox et al.
    
""" 
def treatAll(x, uFP, uTP):
    """
    Expected value calculation for the option treat all
    
    Args: 
        x (float): probability
        uFP (float): utility of false positive
        uTP (float): utility of true positive
        
    Returns: 
        expected utility (float)
    
    """
    return x * uTP + (1-x) * uFP

def treatNone(x, uFN, uTN):
    """
    Expected value calculation for the option treat none
    
    Args: 
        x (float): probability
        uFN (float): utility of false negative
        uTN (float): utility of true negative
        
    Returns: 
        expected utility (float)
    
    """
    return x * uFN + (1-x) * uTN
def test(x, sensitivity, specificity, uTN, uTP, uFN, uFP, u):
    """
    Expected value calculation for the option test
    
    Args: 
        x (float): probability
        sensitivity (float): sensitivity of the test
        specificity (float): specificity of the test
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u: utility of the test itself
        
    Returns: 
        expected utility (float)
    
    """
    return x * sensitivity * uTP + x * (1-sensitivity) * uFN + (1-x) * (1-specificity) * uFP + (1-x) * specificity * uTN + u


def pLpStarpUThresholds(sens, spec, uTN, uTP, uFN, uFP, u):
    """
    Identifies the three thresholds formed by the three utility lines
    
    Args: 
        sens (float): sensitivity of the test
        spec (float): specificity of the test
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u: utility of the test itself
        
    Returns: 
        a list of three thresholds (pL, pStar, and pU): [pL, pStar, pU]
    
    """
    #initate a variable called x (prior)
    x = sy.symbols('x')
    
    #solve for upper threshold formed by test and treat all
    pU = sy.solve(treatAll(x, uFP, uTP) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    #solve for treatment threshold formed by treat all and treat none
    pStar = sy.solve(treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN), x)
    
    #solve for lower threshold formed by treat none and test
    pL = sy.solve(treatNone(x, uFN, uTN) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    #placeholder values when there are not two thresholds formed
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

def modelPriorsOverRoc(modelChosen, uTN, uTP, uFN, uFP, u):
    """
    Collects all the lower, pStar, and upper thresholds for every point on the ROC curve
    
    Args: 
        modelChosen (model): chosen ml model
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u: utility of the test itself
        
    Returns: 
        a list of lists of thresholds (pL, pStar, and pU): [[pL], [pStar], [pU]]
    
    """
    pLs = []
    pStars = []
    pUs = []
    
    # get TPRs and FPRs from the model
    if(type(np.array(modelChosen['tpr'])) == list):
        tprArray = np.array(np.array(modelChosen['tpr'])[0])
        fprArray = np.array(np.array(modelChosen['fpr'])[0])
    elif type(np.array(modelChosen['tpr'])[0]) == list:
        tprArray = np.array(np.array(modelChosen['tpr'])[0])
        fprArray = np.array(np.array(modelChosen['fpr'])[0])
    elif (np.array(modelChosen['tpr'])).size > 1:
        tprArray = np.array(modelChosen['tpr'])
        fprArray = np.array(modelChosen['fpr'])
    else:
        tprArray = np.array(modelChosen['tpr'])[0]
        fprArray = np.array(modelChosen['fpr'])[0]
        
    #for each pair of tpr, fpr
    if(tprArray.size > 1):
        for cutoffIndex in range(0, tprArray.size):
            
            #assign tpr and fpr
            tpr = tprArray[cutoffIndex]
            fpr = fprArray[cutoffIndex]
            
            #find pL, pStar, and pU thresholds
            pL, pStar, pU = pLpStarpUThresholds(tpr, 1 - fpr, uTN, uTP, uFN, uFP, u)
            
            #append results
            pLs.append(pL)
            pStars.append(pStar)
            pUs.append(pU)
            
        return [pLs, pStars, pUs]
    else:
        return [[0], [0], [0]]
    
def priorFiller(priorList, lower: bool):
    """
    Some priors are not defined. For those -999, change to 1 or 0 depending on pL or pU
    
    Args: 
        priorList (list): list of lower or upper thresholds (priors)
        lower (bool): specifies list of lower or upper thresholds. True for lower, false for upper.
        
    Returns: 
        A modified list of priors that fills in the "NA"(-999) values at the head and tail of the list
    
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
    The previous prior filler function did not take care of all the problems. 
    This function provides additional modifications. 
        - for example, "1, 0, 1" the 0 should be a 1. 
        - another example, "0, 0, 1, 0" the 1 should be a 0.
    Will refine and merge the two functions in the future
    
    Args: 
        priorList (list): list of lower or upper thresholds (priors)
        
    Returns: 
        A modified list of priors
    
    """
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
    Based on https://github.com/scikit-learn/scikit-learn/issues/3097, by default 1 is added to the last number to 
    compute the entire ROC curve. Thus, this extracts the thresholds and adjusts those outside the [0,1] range. 
    
    Args:
        row (list): a row in the dataframe with the column "thresholds" obtained from the model
        
    Returns: 
        a modified list of thresholds. 
    """
    thresholds = row['thresholds']
    if thresholds is not None:
        for i, cutoff in enumerate(thresholds):
            if(cutoff > 1):
                thresholds[i] = 1
        return thresholds
    else:
        return None

def adjustpLpUClassificationThreshold(thresholds, pLs, pUs):
    """
    Modifies the prior thresholds as well as the predicted probability cutoff thresholds 
    
    Args:
        thresholds (list): a row in the dataframe with the column "thresholds" obtained from the model
        
    Returns: 
        a modified list of thresholds. 
    """
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

def eqLine(x, x0, x1, y0, y1):
    """
    Find the value of f(x) given x and the two points that make up the line of interest.
    
    Args:
        x (float): desired x
        x0 (float): x coordinate of the first point
        x1 (float): y coordinate of the first point
        y0 (float): x coordinate of the second point
        y1 (float): y coordinate of the second point
        
    Returns: 
        f(x)
    """
    slope = (y1 - y0) / (x1 - x0)
    y = slope * (x - x0) + y0
    return y

def applicableArea(modelRow, thresholds, utils, p):
    """
    Find the applicability area (ApAr) of the model. 
    Interpretation of the result: ranges of prior probability in the target population for which the model has value (utility)
    over the alternatives of treat all and treat none. ApAr is calculated by integrating the range of applicable prior over
    the entired ROC. 
    
    Args:
        modelRow (row of a dataframe): a row of the dataframe with the model parameters and results:
            - asymmetric cost
            - TPRs
            - FPRs
            - predicted probability cutoff thresholds
            - utilities
                - uTN > uTP > uFP > uFN
                - uFN should be 0 or has the least utility
                - uTN should be 1 or has the highest utility
                - uTP should have the second highest utility
                - uFP should have the third highest utility
        thresholds (list): list of thresholds used for classifying the predicted probabilities
        utils (list): list of utility parameters
            - utilities of true negative, true positive, false negative, false positive, and the uility of the test itself
        p (float): the specific prior probability of interest. See if the specified prior fits in the range of applicable priors
        
    Returns: 
        a list of 5 results
            area (float): the ApAr value
            largestRangePriorThresholdIndex (int): index of the threshold that gives us the largest range of applicable priors
            withinRange (bool): a boolean indicating if the specified prior of interest falls within the range of the model
            leastViable (float): minimum applicable prior for the target population
            uFP (float): returns the uFP 
    """
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
    
    #modify the classification thresholds
    thresholds = np.array(thresholds)
    thresholds = np.where(thresholds > 1, 1, thresholds)
    thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
    
    #calculate applicability area
    for i, prior in enumerate(pLs):
        if i < len(pLs) - 1:
            if pLs[i] < pUs[i] and pLs[i + 1] < pUs[i + 1]:
                
                #find the range of priors
                rangePrior = pUs[i] - pLs[i]
                
                #check if it is the largest range of priors
                if rangePrior > largestRangePrior:
                    largestRangePrior = rangePrior
                    largestRangePriorThresholdIndex = i
                    
                # trapezoidal rule (upper + lower base)/2
                avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2 
                
                #accumulate areas
                area += abs(avgRangePrior) * abs(thresholds[i + 1] - thresholds[i])
                
            #where pL and pU cross into pU > pL
            elif pLs[i] > pUs[i] and pLs[i + 1] < pUs[i + 1]:                
                x0 = thresholds[i]
                x1 = thresholds[i+1]
                if x0 != x1:
                    pL0 = pLs[i]
                    pL1 = pLs[i+1]
                    pU0 = pUs[i]
                    pU1 = pUs[i+1]
                    x = sy.symbols('x')
                    
                    #solve for x and y at the intersection
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                    
                    # trapezoidal rule (upper + lower base)/2
                    avgRangePrior = (0 + (pUs[i + 1] - pLs[i + 1])) / 2
                    
                    #accumulate areas
                    area += abs(avgRangePrior) * abs(thresholds[i + 1] - xIntersect[0])
                
            elif (pLs[i] < pUs[i] and pLs[i + 1] > pUs[i + 1]):
                x0 = thresholds[i]
                x1 = thresholds[i+1]
                if x0 != x1:
                    pL0 = pLs[i]
                    pL1 = pLs[i+1]
                    pU0 = pUs[i]
                    pU1 = pUs[i+1]
                    x = sy.symbols('x')
                    
                    #solve for x and y at the intersection
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    
                    if len(xIntersect) == 0:
                        xIntersect = [0]
                        
                    yIntersect = eqLine(xIntersect[0], x0, x1, pL0, pL1)
                    
                    #accumulate areas
                    avgRangePrior = (0 + (pUs[i] - pLs[i])) / 2 # trapezoidal rule (upper + lower base)/2
                    area += abs(avgRangePrior) * abs(xIntersect[0] - thresholds[i + 1])
                
    #round the calculation
    area = np.round(float(area), 3)
    
    #due to minor calculation inaccuracies in the previous iterations of the function. This should no longer apply. All ApAr 
    #should be less than 1
    if(area > 1):
        area = 1           
    #check if the specified prior is within the ranges of priors
    if((p > minPrior) & (p < maxPrior)):
        withinRange = True
                
    return [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]