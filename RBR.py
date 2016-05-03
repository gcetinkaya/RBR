""" Author: Gokhan Cetinkaya, gcastorrr@gmail.com

    This class implements a feature generator based on the paper:
      Random Bits Regression: a Strong General Predictor for Big Data
      Yi Wang, Yi Li, Momiao Xiong, Li Jin
      http://arxiv.org/abs/1501.02990

    The class persists feature schema that can later be used to apply the process to test data.

    Usage:

    # get n random bir features:
    rbr = RBR(train)
    features = rbr.generate_features(1000)

    # get schema:
    schema = rbr.feature_schema

    # dump schema:
    rbr.dump_schema('rbr_dump')

    # load schema and reset features:
    schema = rbr.load_schema('rbr_dump')

    # get features for specific schema and X:
    schema_features = rbr.generate_features_from_schema(X, schema)
"""

import time

import numpy as np
import pandas as pd
import joblib


class RBR(object):
  """ Base class for generating random bit features.

      params:
      -------
        X                   : a pandas DataFrame object
        n_attrs_per_feature : number of original features to generate a random bit feature. (int or list of ints w probas)

  """
  def __init__(self, X, n_attrs_per_feature=[(2, 0.25), (3, 0.50), (4, 0.25)],
                random_state=time.time()):

    if X.empty:
      raise ValueError("X is required")
    self.X = X

    self.n_attrs_per_feature = n_attrs_per_feature
    np.random.seed(int(random_state))
    self.random_state = random_state

    self.feature_schema = {}
    self.features = pd.DataFrame()
    self.last_feat_ix = 0

  def add_feature(self, attr_indexes, weights, threshold, feature):
    """ Adds generated feature to features DataFrame and features_schema dict.
    """
    ix = self.last_feat_ix
    self.features[ix] = pd.Series(feature)
    self.feature_schema[ix] = {"attr_indexes": attr_indexes, "weights": weights, "threshold": threshold}
    self.last_feat_ix += 1

  def _get_feature(self):
    """ Generates a random feature and populates it's values by given data.
    """
    while True:
      attrs, weights = self._get_random_attrs_weights()
      concept_data = self.X[attrs]
      features = np.dot(concept_data, weights)
      unique_features = np.unique(features)
      # if it's a binary feature, set threshold accordingly..
      if len(unique_features) == 2:
        threshold = unique_features[1]
      else:
        threshold = np.random.choice(unique_features)
      features[features>=threshold] = 1
      features[features<threshold] = 0
      # if there's no variance in created feature, discard it..
      if features.std() == 0.0:
        continue
      # if self.is_existing_feature(features):
      #   continue

      return attrs, weights, threshold, features

  def generate_features(self, n_features, warm_start=True):
    """ Generates n_features random bir features and persists them.
    """
    if not warm_start:
      self.features = pd.DataFrame()
      self.feature_schema = {}

    start = time.time()
    for i in range(n_features):
      attr_indexes, weights, threshold, feature = self._get_feature()
      self.add_feature(attr_indexes, weights, threshold, feature)

    print "Generated %d features in %.1f secs" % (n_features, time.time()-start)

    return self.features

  def _get_random_attrs_weights(self):
    """ Determines attributes (original features) and weights to use to generate a random bit feature.
    """
    if type(self.n_attrs_per_feature) == int:
      n_attrs_for_feat = self.n_attrs_per_feature
    else:
      n_attrs_candidates = [ nattr for nattr, prob in self.n_attrs_per_feature ]
      attr_probas = [ prob for nattr, prob in self.n_attrs_per_feature ]
      n_attrs_for_feat = np.random.choice(n_attrs_candidates, size=1, p=attr_probas, replace=False)

    #attr_indexes = np.random.choice(range(self.X.shape[1]), size=n_attrs_for_feat, replace=False)
    attrs = np.random.choice(range(self.X.shape[1]), size=n_attrs_for_feat, replace=False)
    weights = np.array( [ np.random.normal() for _ in range(n_attrs_for_feat) ] ) # weights are selected from normal distribution

    return attrs, weights

  def _get_feature_from_schema(self, data, attr_indexes, weights, threshold):

      concept_data = data[:,attr_indexes]
      features = np.dot(concept_data, weights)
      features[features>=threshold] = 1
      features[features<threshold] = 0

      return features

  def generate_features_from_schema(self, X, schema):
    """ Generates and returns features for given X and schema.
        If schema is not provided, uses self.feature_schema.
    """
    if X.empty:
      raise ValueError("X is required")

    if not schema:
      if not self.feature_schema:
        raise ValueError("schema is required")
      else:
        schema = self.feature_schema

    _features = pd.DataFrame()
    for ix, d in schema.items():
      feature = self._get_feature_from_schema(X.as_matrix(), d["attr_indexes"], d["weights"], d["threshold"])
      _features[ix] = pd.Series(feature)

    return _features

  def dump_schema(self, f):
    """ Dumps current schema to file w joblib
    """
    if not self.feature_schema:
      raise ValueError("schema is not present")

    joblib.dump(self.feature_schema, f)

  def load_schema(self, f):
    """ Loads schema from file.
        Resets features.
    """
    self.feature_schema = joblib.load(f)
    self.features = self.generate_features_from_schema(self.X, self.feature_schema)


