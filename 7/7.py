import numpy as np
from scipy.stats import chi2


def Simonov_Cay(data=None,data_pred=None):

    if data==None:
        data=np.array([
            [
                10,30
            ],
            [
                15,25
            ]
        ])
    if data_pred==None:
        data_pred=np.array([
            [20,20],
            [20,20]
        ])
    χ2_critical=chi2.ppf(0.95,df=(10-1)*(10-1))
    χ2_stat=(((data-0.5)**2)/0.5).sum()

    left=((χ2_critical**2)**(1/2))/((χ2_stat**2)**(3/2))
    right=((np.abs(data-data_pred)**3)/((data_pred)**2)).sum()

    S=left*right

    if S>0.25:
        print('С апроксимацией χ² могут быть потенциальные проблемы')

    else:
        print('Аппроксимация хи–квадрата допустима')

Simonov_Cay()