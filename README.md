# ImputeNormal

PySpark Estimator whose fit method finds the mean and standard deviation of a DataFrame column.  The Model which is emitted replaces missing values with a random variate drawn from the normal distribution with that mean and standard deviation.  If the mean (or median) is imputed to replace missing values, the variance of the imputed column will have be smaller than the variance of the original column.
