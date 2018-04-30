# Batch-predict

Repository for batch predict

## Overview

Batch predict is useful when users have a large number of instances to get
predictions for and/or they don't need to get the prediction result in a
real-time fashion.

This [apache-beam](https://beam.apache.org/)-based implementation allows several
input file formats: JSON (text), TFRecord, and compressed TFRecord files. It
supports JSON and CSV output formats.  Batch predict supports models trained
using [TensorFlow](http://www.tensorflow.org)(in SavedModel format),
[xgboost](http://xgboost.readthedocs.io/en/latest/) and
[scikit-learn](http://scikit-learn.org/stable/).

Today, batch predict can run on a single node in a K8s cluster using beam local
runner. Alternatively, it can run on Google's Dataflow service using Dataflow
Runner. We expect as other runners on K8s mature, it can run on multiple nodes
in a k8s cluster.

Batch predict also supports running on GPU (k80/P100) in k8s if the cluster has
configured with GPU and proper nvidia drivers installed.
