__author__ = 'kirk'

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
combined = pd.read_csv('sets123.csv')
bldg1 = combined.ix[combined['Building'] == 'BLDG001']
bldg1 = bldg1.dropna(axis=1, how='all')
del bldg1['Main.Load.Non.HVAC']
del bldg1['Building']


bldg1.ix[:,2:7]=bldg1.ix[:,2:7].apply(pd.Series.interpolate)
bldg1RF = RandomForestRegressor(n_estimators=10000, oob_score=True, n_jobs=32)
bldg1RFfitted = bldg1RF.fit(bldg1.ix[:,2:7], bldg1['Main.Load'])

print bldg1['Main.Load'] - bldg1RFfitted.oob_prediction_

print bldg1RFfitted.oob_score_
del bldg1['IntervalStart']
del combined

fits = {'fitted':bldg1RFfitted.oob_prediction_, 'actual': bldg1['Main.Load'], 'difference': bldg1['Main.Load'] - bldg1RFfitted.oob_prediction_}
fits = pd.DataFrame(fits)
fits.to_csv('fitresults.csv')
