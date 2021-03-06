import pytest

from ml_tools import FeatureGenerator, Hour, SimpleAggregator, custom_generator, SingleAggregator, Average
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


def test_custom_feature_generator(user_test_func, single_dataset):
    usr_func = user_test_func
    df = single_dataset
    test_generator = custom_generator(usr_func)

    new_feature = test_generator.generate_feature(df)

    assert new_feature.equals(df)


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
    assert test_agg._rkey1 == "type"
    assert test_agg._rkey2 == "type"


def test_new_relationship(multi_dataset):
    df1, df2 = multi_dataset

    test_agg = SimpleAggregator(df1, df2)
    test_agg.new_relationship('type', 'type')

    assert test_agg.relationships == "\n RELATIONSHIPS:\ndata1.type -> data2.type\n"
    assert test_agg._rkey1 == "type"
    assert test_agg._rkey2 == "type"


def test_simple_agg(multi_dataset):
    df1, df2 = multi_dataset

    df_comp = pd.DataFrame({'type': ['mammal', 'reptile', 'insect'],
                            'count': [3, 1, 0]})

    test_agg = SimpleAggregator(df1, df2, rkey1='type', rkey2='type')

    test_feature = test_agg.aggregate()

    assert test_feature.equals(df_comp)


def test_single_agg(single_dataset):
    df = single_dataset

    df_comp = pd.DataFrame({'animal': ["dog", "cat", "monkey"],
                            'count': [1, 1, 1]})

    test_agg = SingleAggregator(df, rkey1='animal')

    test_feature = test_agg.aggregate().copy()

    test_sorted = test_feature.sort_values(by='animal', ignore_index=True)
    df_comp_sorted = df_comp.sort_values(by='animal', ignore_index=True)

    assert test_sorted.equals(df_comp_sorted)


def test_average(test_dataset):
    df = test_dataset

    df_comp = pd.DataFrame(df['species'].value_counts()).reset_index()
    cols = ['species', 'count']
    df_comp.columns = cols
    elements = df_comp['species']

    avgs = []
    for el in list(elements):
        agg_data = df.loc[df['species'] == el]
        x = [el]
        x.extend([agg_data[col].mean() for col in agg_data if col != 'species'])
        avgs.append(x)
    new_cols = ['species']
    new_cols.extend([col+'_avg' for col in df if col != 'species'])
    avgs = pd.DataFrame(avgs)
    avgs.columns = new_cols

    df_comp = pd.merge(df_comp, avgs, on='species')

    agg = Average(df, "species")

    df_avg = agg.aggregate()

    assert df_avg.equals(df_comp)
