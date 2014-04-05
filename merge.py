__author__ = 'kirk'
__author__ = 'kirk'
import pandas as pd
aafirst = pd.read_csv("/home/kirk/First37.csv")
second = pd.read_csv("/home/kirk/Second37.csv")
third = pd.read_csv("/home/kirk/Third36.csv")



combined = first.append(second)
combined = combined.append(third)
del first
del second
del third
combined.to_csv('/home/kirk/sets123.csv')

