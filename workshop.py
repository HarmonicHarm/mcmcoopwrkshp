import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn import metrics

modeld = pd.read_excel("modelling_data.xlsx")

Xm = modeld.drop(['claimcount'], axis = 1)
Ym = modeld['claimcount']

#print(Xm)
print("-----------------------------------------------")
print(Ym.describe())

evald = pd.read_excel("evaluation_data.xlsx")


Xe = evald.drop(['ROW_ID','claimcount'], axis = 1)
Ye = evald['claimcount']

#print(Xe)
#print(Ye.describe())

#print(Xm.dtypes)

'''
print(Xm.annual_mileage.value_counts())
print(Xm.winter_tires.value_counts())
print(Xm.gender.value_counts()) 
print(Xm.location.value_counts())
print(Xm.annual_income.value_counts())
print(Xm.ownership.value_counts()) 
print(Xm.occupation.value_counts())
print(Xm.credit_band.value_counts())
print(Xm.marital_status.value_counts()) 
print(Xm.vehicle_value.value_counts())
print(Xm.car_model.value_counts())
'''

Xm.replace({'annual_mileage':{'0-5000':0, '5000-10000':1,'10000-15000':2,'15000-20000':3,'20000-25000':4}},inplace=True)
Xm['annual_mileage'] = Xm['annual_mileage'].astype('int')
Xm.replace({'winter_tires':{'not provided':0, 'No':1, 'Yes':2}},inplace=True)
Xm['winter_tires'] = Xm['winter_tires'].astype('int')
Xm.replace({'gender':{'Female':0, 'Male':1}},inplace=True)
Xm['gender'] = Xm['gender'].astype('int')
Xm.replace({'location':{'Mississauga':0, 'Oakville':1,'Toronto':2,'Burlington':3, 'Hamilton':4,'Brampton':5}},inplace=True)
Xm['location'] = Xm['location'].astype('int')
Xm.replace({'annual_income':{'0-10000':0, '10000-30000':1, '30000-50000':2,'50000-75000':3, '75000-100000':4, '100000-150000':5, '150000-1000000':6}},inplace=True)
Xm['annual_income'] = Xm['annual_income'].astype('int')
Xm.replace({'ownership':{'not provided':0, 'financed':1,'leased':2, 'owned':3}},inplace=True)
Xm['ownership'] = Xm['ownership'].astype('int')
Xm.replace({'occupation':{'not provided':0, 'Military':1, 'Farming':2, 'Government':3, 'Healthcare':4, 'Businessowner':5, 'Unemployed':6, 'Trades':7, 'Industrial':8, 'Office':9, 'Education':10, 'Retail':11}},inplace=True)
Xm['occupation'] = Xm['occupation'].astype('int')
Xm.replace({'credit_band':{'not provided':0, 'K':1, 'I':2, 'J':3, 'F':4, 'E':5, 'D':6, 'C':7, 'B':8, 'A':9, 'AA':10, 'AAA':11}},inplace=True)
Xm['credit_band'] = Xm['credit_band'].astype('int')
Xm.replace({'marital_status':{'not provided':0, 'Widowed':1, 'Seperated':2, 'Single':3, 'Married':4}},inplace=True)
Xm['marital_status'] = Xm['marital_status'].astype('int')
Xm.replace({'vehicle_value':{'0-5000':0, '5000-10000':1, '10000-15000':2, '15000-20000':3, '20000-25000':4, '25000-30000':5, '30000-50000':6, '50000-75000':7, '75000-100000':8}},inplace=True)
Xm['vehicle_value'] = Xm['vehicle_value'].astype('int')
Xm.replace({'car_model':{'Kia':0, 'Jeep':1, 'Hyundai':2, 'Cadillac':3, 'Volvo':4, 'Dodge':5, 'Ford':6, 'Nissan':7, 'Audi':8, 'Subaru':9, 'BMW':10, 'Toyota':11, 'Mercedes':12, 'Honda':13}},inplace=True)
Xm['car_model'] = Xm['car_model'].astype('int')

Xe.replace({'annual_mileage':{'0-5000':0, '5000-10000':1,'10000-15000':2,'15000-20000':3,'20000-25000':4}},inplace=True)
Xm['annual_mileage'] = Xm['annual_mileage'].astype('int')
Xe.replace({'winter_tires':{'not provided':0, 'No':1, 'Yes':2}},inplace=True)
Xm['winter_tires'] = Xm['winter_tires'].astype('int')
Xe.replace({'gender':{'Female':0, 'Male':1}},inplace=True)
Xm['gender'] = Xm['gender'].astype('int')
Xe.replace({'location':{'Mississauga':0, 'Oakville':1,'Toronto':2,'Burlington':3, 'Hamilton':4,'Brampton':5}},inplace=True)
Xm['location'] = Xm['location'].astype('int')
Xe.replace({'annual_income':{'0-10000':0, '10000-30000':1, '30000-50000':2,'50000-75000':3, '75000-100000':4, '100000-150000':5, '150000-1000000':6}},inplace=True)
Xm['annual_income'] = Xm['annual_income'].astype('int')
Xe.replace({'ownership':{'not provided':0, 'financed':1,'leased':2, 'owned':3}},inplace=True)
Xm['ownership'] = Xm['ownership'].astype('int')
Xe.replace({'occupation':{'not provided':0, 'Military':1, 'Farming':2, 'Government':3, 'Healthcare':4, 'Businessowner':5, 'Unemployed':6, 'Trades':7, 'Industrial':8, 'Office':9, 'Education':10, 'Retail':11}},inplace=True)
Xm['occupation'] = Xm['occupation'].astype('int')
Xe.replace({'credit_band':{'not provided':0, 'K':1, 'I':2, 'J':3, 'F':4, 'E':5, 'D':6, 'C':7, 'B':8, 'A':9, 'AA':10, 'AAA':11}},inplace=True)
Xm['credit_band'] = Xm['credit_band'].astype('int')
Xe.replace({'marital_status':{'not provided':0, 'Widowed':1, 'Seperated':2, 'Single':3, 'Married':4}},inplace=True)
Xm['marital_status'] = Xm['marital_status'].astype('int')
Xe.replace({'vehicle_value':{'0-5000':0, '5000-10000':1, '10000-15000':2, '15000-20000':3, '20000-25000':4, '25000-30000':5, '30000-50000':6, '50000-75000':7, '75000-100000':8}},inplace=True)
Xm['vehicle_value'] = Xm['vehicle_value'].astype('int')
Xe.replace({'car_model':{'Kia':0, 'Jeep':1, 'Hyundai':2, 'Cadillac':3, 'Volvo':4, 'Dodge':5, 'Ford':6, 'Nissan':7, 'Audi':8, 'Subaru':9, 'BMW':10, 'Toyota':11, 'Mercedes':12, 'Honda':13}},inplace=True)
Xm['car_model'] = Xm['car_model'].astype('int')

model = PoissonRegressor()
model.fit(Xm,Ym)


pred = model.predict(Xm)
print(pred)

error = metrics.r2_score(Ym, pred)
print(error)


'''
model = XGBRegressor(enable)
model.fit(Xm,Ym)
pred = model.predict(Xe)
print(pred)
'''