from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.regression import LinearRegressionWithSGD
import matplotlib.pyplot as plt

sc = SparkContext("local[3]", "Movie Linear Regression")
# read from file 
records=sc.textFile("result.txt").map(lambda line:line.split(" "))
records.cache()

data=records.map(lambda r:LabeledPoint(float(r[-1]),[float(field) for field in r[0:-1]]))
first_point = data.first()
print "Linear Model feature vector length: " + str(len(first_point.features))
# train the model
linear_model = LinearRegressionWithSGD.train(data, iterations=200, step=0.9, intercept=True)
# predict
true_vs_predicted = data.map(lambda p: (p.label,linear_model.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.take(5))
# calculate R^2
y_true=true_vs_predicted.map(lambda (t,p):t).collect()
y_pred=true_vs_predicted.map(lambda (t,p):p).collect()
y_pred=np.array([max(0,p) for p in y_pred])
y_true=np.array([y_true])
t_mean=y_true.mean()
s1=((y_true-y_pred)**2).sum()
s2=((y_true-t_mean)**2).sum()
#s1=true_vs_predicted.map(lambda (t,p):(t-p)**2).sum()
#s2=true_vs_predicted.map(lambda (t,p):(t-t_mean)**2).sum()
R2=1-s1/s2
print "Linear Model - R^2 :%.4f" %R2
# choose hyperparameters
def evaluate(data,iter,s):
	linear_model = LinearRegressionWithSGD.train(data, iterations=iter, step=s, intercept=True)
	true_vs_predicted = data.map(lambda p: (p.label,linear_model.predict(p.features)))
	y_true=true_vs_predicted.map(lambda (t,p):t).collect()
	y_pred=true_vs_predicted.map(lambda (t,p):p).collect()
	y_pred=np.array([max(0,p) for p in y_pred])
	y_true=np.array([y_true])
	t_mean=y_true.mean()
	s1=((y_true-y_pred)**2).sum()
	s2=((y_true-t_mean)**2).sum()
	R2=1-s1/s2
	return R2

params=[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
metrics=[]
for p in params:
	metrics.append(evaluate(data,200,p))
print params
print metrics
plt.plot(params,metrics)
plt.show()