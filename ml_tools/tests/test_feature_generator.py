import pytest

from ml_tools import FeatureGenerator, Hour, SimpleAggregator
import pandas as pd


def test_not_implemented():
    with pytest.raises(TypeError, match="Can't instantiate abstract class FeatureGenerator with abstract methods "
                                        "generate_feature"):
        FeatureGenerator()


# def test_feature_generator_init(single_dataset):
#     df = single_dataset
#
#     test_feature = GenericGenerator(df)
#
#     assert test_feature._data.equals(df)


def test_hour(timestamps):
    df = timestamps
    df_comp = df

    for idx, time in enumerate(df.timestamp):
        df_comp.timestamp[idx] = time.hour

    new_feature = Hour.generate_feature(df)

    assert new_feature.equals(df_comp)


def test_hour_columns(timestamps):
    df = timestamps
    df_comp = df.copy()

    for idx, time in enumerate(df.timestamp):
        df_comp.timestamp[idx] = time.hour

    new_feature = Hour.generate_feature(df, column='timestamp')

    assert new_feature.equals(df_comp)


def test_simple_aggregator_init(multi_dataset):
    df1, df2 = multi_dataset

    test_agg = SimpleAggregator(df1, df2)

    assert test_agg.keys() == {'data1', 'data2'}
    assert test_agg['data1'].equals(df1)
    assert test_agg['data2'].equals(df2)


def test_simple_agg_labels(multi_dataset):
    df1, df2 = multi_dataset

    test_agg = SimpleAggregator(df1, df2, label1='animals', label2='animal_types')

    assert test_agg['animals'].equals(df1)
    assert test_agg['animal_types'].equals(df2)


def test_simple_define_relationship(multi_dataset):
    df1, df2 = multi_dataset

    test_agg = SimpleAggregator(df1, df2, rkey1='type', rkey2='type')

    assert test_agg.relationships == "\n RELATIONSHIPS:\ndata1.type -> data2.type\n"


def test_new_relationship(multi_dataset):
    df1, df2 = multi_dataset

    test_agg = SimpleAggregator(df1, df2)
    test_agg.new_relationship('type', 'type')

    assert test_agg.relationships == "\n RELATIONSHIPS:\ndata1.type -> data2.type\n"


def test_simple_agg(multi_dataset):
    df1, df2 = multi_dataset

    df_comp = pd.DataFrame({'type': ['mammal', 'reptile', 'insect'],
                            'count': [3, 1, 0]})

    test_agg = SimpleAggregator(df1, df2, rkey1='type', rkey2='type')

    test_feauture = test_agg.aggregate()

    assert test_feauture.equals(df_comp)
