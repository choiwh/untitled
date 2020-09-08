import numpy as np

xData = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yData = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
conditionData = np.array([True, False, True, True,False])
result = np.where(conditionData, xData, yData)
result
