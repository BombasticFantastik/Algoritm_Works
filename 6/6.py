import numpy as np
from sklearn.linear_model import LinearRegression


def aic(x=None,y=None):
    if x==None:
        x=np.random.randn(100,1)

    if y==None:
        y=(1.5)*(x**2)
    model=LinearRegression().fit(x,y)
    pred=model.predict(x)
    rss=np.sum((y-pred)**2)
    n=len(y)
    k=model.coef_.size
    if model.intercept_:
        k+=1
    aic=n*np.log(rss/n)+2*k
    return aic
print(f'AIC: {aic()}')    