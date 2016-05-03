# RBR
A simple Random Bit Regression implementation

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
