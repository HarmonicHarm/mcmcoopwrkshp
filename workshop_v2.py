import pandas as pd
import seaborn as sb
import math
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_excel("modelling_data.xlsx")

ed = pd.read_excel("evaluation_data.xlsx")

Xm = d.drop(['claimcount'], axis = 1)
Ym = d['claimcount']
Xe = ed.drop(['ROW_ID','claimcount'], axis = 1)
Ye = ed['claimcount']

corrholder = pd.concat([Xm, Ym], axis=1)
corr = corrholder.corr(numeric_only=True).sort_values(by='claimcount', ascending=False)
corr2 = corr.drop('claimcount')
sb.heatmap(corr2[['claimcount']], cmap='coolwarm', annot=True)
plt.show()

x_ord = ['annual_mileage','annual_income','credit_band', 'vehicle_value']
x_bin = ['gender', 'winter_tires']
x_one_hot = ['location', 'ownership', 'occupation', 'marital_status', 'car_model']

am_r = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000']
ai_r = ['0-10000', '10000-30000', '30000-50000','50000-75000', '75000-100000', '100000-150000', '150000-1000000']
cb_r = ['K', 'I', 'J', 'F', 'E', 'D', 'C', 'B', 'A', 'AA', 'AAA']
vv_r = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000-30000', '30000-50000', '50000-75000', '75000-100000']

g_r = ['Male', 'Female']
wt_r = ['Yes', 'No']

l_r = ['Mississauga', 'Oakville','Toronto', 'Burlington', 'Hamilton', 'Brampton']
ow_r = ['financed', 'leased', 'owned']
oc_r = ['Military', 'Farming', 'Government', 'Healthcare', 'Businessowner', 'Unemployed', 'Trades', 'Industrial', 'Office', 'Education', 'Retail']
ms_r = ['Widowed', 'Seperated', 'Single', 'Married'] 
cm_r = ['Kia', 'Jeep', 'Hyundai', 'Cadillac', 'Volvo', 'Dodge', 'Ford', 'Nissan', 'Audi', 'Subaru', 'BMW', 'Toyota', 'Mercedes', 'Honda']

print(Xm.head())

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
cb_h = OrdinalEncoder(categories=[cb_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
cb_h2 = cb_h.fit_transform(Xm[['credit_band']])
cb_df = pd.DataFrame(cb_h2, columns=cb_h.get_feature_names_out())
am_h = OrdinalEncoder(categories=[am_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
am_h2 = am_h.fit_transform(Xm[['annual_mileage']])
am_df = pd.DataFrame(am_h2, columns=am_h.get_feature_names_out())
ai_h = OrdinalEncoder(categories=[ai_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
ai_h2 = ai_h.fit_transform(Xm[['annual_income']])
ai_df = pd.DataFrame(ai_h2, columns=ai_h.get_feature_names_out())
vv_h = OrdinalEncoder(categories=[vv_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
vv_h2 = vv_h.fit_transform(Xm[['vehicle_value']])
vv_df = pd.DataFrame(vv_h2, columns=vv_h.get_feature_names_out())

g_h = OrdinalEncoder(categories=[g_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
g_h2 = g_h.fit_transform(Xm[['gender']])
g_df = pd.DataFrame(g_h2, columns=g_h.get_feature_names_out())
wt_h = OrdinalEncoder(categories=[wt_r], handle_unknown='use_encoded_value', unknown_value=np.nan)
wt_h2 = wt_h.fit_transform(Xm[['winter_tires']])
wt_df = pd.DataFrame(wt_h2, columns=wt_h.get_feature_names_out())

l_h = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
l_h2 = l_h.fit_transform(Xm[['location']])
l_df = pd.DataFrame(l_h2, columns=l_h.get_feature_names_out())
ow_h = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ow_h2 = ow_h.fit_transform(Xm[['ownership']])
ow_df = pd.DataFrame(ow_h2, columns=ow_h.get_feature_names_out())
oc_h = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
oc_h2 = oc_h.fit_transform(Xm[['occupation']])
oc_df = pd.DataFrame(oc_h2, columns=oc_h.get_feature_names_out())
ms_h = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ms_h2 = ms_h.fit_transform(Xm[['marital_status']])
ms_df = pd.DataFrame(ms_h2, columns=ms_h.get_feature_names_out())
cm_h = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cm_h2 = cm_h.fit_transform(Xm[['car_model']])
cm_df = pd.DataFrame(cm_h2, columns=cm_h.get_feature_names_out())

Xm_n = Xm.drop(['annual_mileage', 'winter_tires', 'gender', 'annual_income', 'credit_band', 'vehicle_value', 'location', 'ownership', 'occupation', 'marital_status', 'car_model'], axis = 1)

Xm2 = pd.concat([Xm_n, cb_df, am_df, ai_df, vv_df, g_df, wt_df, l_df, ow_df, oc_df, ms_df, cm_df], axis=1)




"""from sklearn.linear_model import LogisticRegression, 

clfs = [
    ('Logistic Regression', LogisticRegression(solver='liblinear', max_iter=2000)),
    ('Poisson Regression', LogisticRegression(solver='liblinear', max_iter=2000)),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Linear SVM', LinearSVC(random_state=42, max_iter=1000, dual='auto')),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
    ('Gradient Boost', GradientBoostingClassifier(random_state=42)),
    ('Bagging', BaggingClassifier(random_state=42)),
    ('CatBoost', CatBoostClassifier(random_state=42, verbose=0)),
]"""

"""

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xs = scaler.fit_transform(Xm)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xm, Ym, test_size=0.9, random_state=10, stratify=Ym)

from sklearn.linear_model import PoissonRegressor
model = PoissonRegressor()
model.fit(Xtrain, Ytrain)

ypred = model.predict(Xtest)


#print(d.head())

from sklearn.metrics import mean_squared_error, r2_score
print("RMSE: ", math.sqrt(mean_squared_error(Ytest, ypred)))
print("R2: ", r2_score(Ytest, ypred))

Av = pd.DataFrame(Ytest)
Pv = pd.DataFrame(ypred)

Av = Av.iloc[1:].reset_index(drop=True)
Pv = Pv.iloc[1:].reset_index(drop=True)

Av.columns = ['Actual']
Pv.columns = ['Predicted']

AvP = pd.concat([Av, Pv], axis=1)

AvP.insert(0, 'Row Number', range(1, len(AvP) + 1))

#print(AvP.tail(20))

plt.plot(AvP['Row Number'], AvP['Actual'], label='Actual', linestyle='-')
plt.plot(AvP['Row Number'], AvP['Predicted'], label='Predicted', linestyle='--')

plt.xlabel("Row Number")  # X-axis label
plt.ylabel("Value")       # Y-axis label
plt.title("Actual vs. Predicted Line Graph")
plt.legend()              # Show legend

plt.show()


#AvP.to_csv('AvP.csv', index=False)

Av.to_csv('ActualValues.csv', index=False)
Pv.to_csv('PredValues.csv', index=False)

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

regr = OLS(y, add_constant(X)).fit()
print(regr.aic)
"""