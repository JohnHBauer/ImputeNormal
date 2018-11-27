# ImputeNormal

PySpark Estimator whose fit method finds the mean and standard deviation of a DataFrame column.  The Model which is emitted replaces missing values with a random variate drawn from the normal distribution with that mean and standard deviation.