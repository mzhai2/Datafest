import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#details = pd.read_csv("sets123.csv")
bldg1 = pd.read_csv("/Users/Fei/Desktop/001-Restraurant Quick Serve 1.csv")
#test = bldg1['Building']
bldg1 = bldg1.dropna(axis=1, how='all')
del bldg1['Main.Load.Non.HVAC']
del bldg1['IntervalStart']
del bldg1['Building']
bldg1
bldg1.ix[:, 2:7]=bldg1.ix[:, 2:7].apply(pd.Series.interpolate)
RFR = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=3)
fitted = RFR.fit(bldg1.ix[:, 2:7], bldg1['Main.Load'])
#.head()
print fitted
fitted.oob_prediction_
Rsquare = RFR.score(bldg1.ix[:, 2:7], bldg1['Main.Load'])
print Rsquare