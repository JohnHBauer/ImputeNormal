#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:25:10 2018

@author: JohnBauer
"""

import os
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_191.jdk/Contents/Home"
os.environ['SPARK_HOME'] = "/Users/john.h.bauer/spark"
os.environ['PYTHONPATH'] = "$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPATH"

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, randn

from pyspark import keyword_only
from pyspark.ml import Estimator, Model
#from pyspark.ml.feature import SQLTransformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

spark = SparkSession\
    .builder\
    .appName("ImputeNormal")\
    .getOrCreate()
    
class ImputeNormal(Estimator,
                   HasInputCol,
                   HasOutputCol,
                   DefaultParamsReadable,
                   DefaultParamsWritable,
                   ):
    @keyword_only
    def __init__(self, inputCol="inputCol", outputCol="outputCol"):
        super(ImputeNormal, self).__init__()
        
        self._setDefault(inputCol="inputCol", outputCol="outputCol")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, inputCol="inputCol", outputCol="outputCol"):
        """
        setParams(self, inputCol="inputCol", outputCol="outputCol")
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self
    
    def _fit(self, data):
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        stats = data.select(inputCol).describe()
        mean = stats.where(col("summary") == "mean").take(1)[0][inputCol]
        stddev = stats.where(col("summary") == "stddev").take(1)[0][inputCol]
        
        return ImputeNormalModel(mean=float(mean),
                                 stddev=float(stddev),
                                 inputCol=inputCol,
                                 outputCol=outputCol,
                                 )
# FOR A TRULY MINIMAL BUT LESS DIDACTICALLY EFFECTIVE DEMO, DO INSTEAD:        
#        sql_text = "SELECT *, IF({inputCol} IS NULL, {stddev} * randn() + {mean}, {inputCol}) AS {outputCol} FROM __THIS__"
#        
#        return SQLTransformer(statement=sql_text.format(stddev=stddev, mean=mean, inputCol=inputCol, outputCol=outputCol))
   
class ImputeNormalModel(Model,
                        HasInputCol,
                        HasOutputCol,
                        DefaultParamsReadable,
                        DefaultParamsWritable,
                        ):
    
    mean = Param(Params._dummy(), "mean", "Mean value of imputations. Calculated by fit method.",
                  typeConverter=TypeConverters.toFloat)

    stddev = Param(Params._dummy(), "stddev", "Standard deviation of imputations. Calculated by fit method.",
                  typeConverter=TypeConverters.toFloat)


    @keyword_only
    def __init__(self, mean=0.0, stddev=1.0, inputCol="inputCol", outputCol="outputCol"):
        super(ImputeNormalModel, self).__init__()
        
        self._setDefault(mean=0.0, stddev=1.0, inputCol="inputCol", outputCol="outputCol")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        
    @keyword_only
    def setParams(self, mean=0.0, stddev=1.0, inputCol="inputCol", outputCol="outputCol"):
        """
        setParams(self, mean=0.0, stddev=1.0, inputCol="inputCol", outputCol="outputCol")
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def getMean(self):
        return self.getOrDefault(self.mean)

    def setMean(self, mean):
        self._set(mean=mean)

    def getStddev(self):
        return self.getOrDefault(self.stddev)

    def setStddev(self, stddev):
        self._set(stddev=stddev)

    def _transform(self, data):
        mean = self.getMean()
        stddev = self.getStddev()
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        
        df = data.withColumn(outputCol,
                             when(col(inputCol).isNull(),
                                  stddev * randn() + mean).\
                                  otherwise(col(inputCol)))
        return df

if __name__ == "__main__":
    
    train = spark.createDataFrame([[0],[1],[2]] + [[None]]*100,['input'])
    impute = ImputeNormal(inputCol='input', outputCol='output')
    impute_model = impute.fit(train)
    print("Input column: {}".format(impute_model.getInputCol()))
    print("Output column: {}".format(impute_model.getOutputCol()))
    print("Mean: {}".format(impute_model.getMean()))
    print("Standard Deviation: {}".format(impute_model.getStddev()))
    test = impute_model.transform(train)
    test.show(10)
    test.describe().show()
    print("mean and stddev for outputCol should be close to those of inputCol")
 

    