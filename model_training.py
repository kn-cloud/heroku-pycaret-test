#!/usr/bin/env python3
#from pycaret.datasets import get_data
import numpy as np
import pandas as pd
import sys

dataset = pd.read_csv('data_elas_1.csv')
dataset = dataset[dataset.columns[6:11]]
print(dataset.columns)

data = dataset.sample(frac=0.8, random_state=123).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


from pycaret.regression import *

exp_reg101 = setup(data = data, target = 'elastic_anisotropy', session_id=123) 


#compare_models()


#create, tune, finalize, and save model
ada = create_model('ada')
tuned_ada = tune_model('ada')
final_ada = finalize_model(tuned_ada)
save_model(final_ada,'Final_AdaBoost_Model')

# test
cols = ['K_Reuss', 'K_VRH', 'K_Voigt', 'K_Voigt_Reuss_Hill']
final = np.array([1,2,3,4])
data_unseen = pd.DataFrame([final], columns = cols)
prediction = predict_model(final_ada, data=data_unseen)
print(prediction.Label[0])
