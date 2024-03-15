import pandas 
from statsmodels.tsa.ar_model import AutoReg

import matplotlib.pyplot as plt
import datetime
import math


dataset_df = pandas.read_csv("vente_maillots_de_bain.csv")


def prepare_data_autoreg(dataset_df, x_column_name, y_column_name):

	dataset_df[x_column_name] = pandas.to_datetime(dataset_df[x_column_name])

	dataset_df.set_index(x_column_name, inplace=True)
	dataset_df.index = pandas.DatetimeIndex(dataset_df.index.values, freq=dataset_df.index.inferred_freq)
	df_mean, df_std = dataset_df[y_column_name].mean(), dataset_df[y_column_name].std()

	dataset_df[y_column_name]= (dataset_df[y_column_name]-df_mean)/df_std

	split_index = int(len(dataset_df) * 0.7)

	train_df = dataset_df.iloc[ : split_index]
	test_df = dataset_df.iloc[split_index : ]

	return train_df, test_df, df_mean, df_std



def generate_auto_reg_fitted_model(train_df, lag):
	return AutoReg(train_df, lag).fit()


def test_model(model, test_df, start="2007-04-01", end="2009-12-01", y_column_name="Sales"):

	start = datetime.datetime.strptime(start, "%Y-%m-%d")
	end = datetime.datetime.strptime(end, "%Y-%m-%d")


	y_test_predicted = model.predict(start, end).values
	rmse = math.sqrt(sum((test_df[y_column_name].values - y_test_predicted)**2)/len(y_test_predicted))
	print(rmse)

	plt.figure(figsize=(15, 6))
	
	plt.plot(test_df.index, test_df[y_column_name], "b-") # données originales
	plt.plot(test_df.index, y_test_predicted, "r-")

	
	plt.ylabel(y_column_name)
	plt.legend(["Vérité terrain", "Prediction"])
	plt.show()





train_df, test_df, df_mean, df_std = prepare_data_autoreg(dataset_df, x_column_name="Years", y_column_name="Sales")
model = generate_auto_reg_fitted_model(train_df, lag=12)
test_model(model, test_df, start="2007-04-01", end="2009-12-01", y_column_name="Sales")