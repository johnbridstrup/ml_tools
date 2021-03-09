import pytest
from uszipcode import SearchEngine

from ml_tools import FeatureGenerator, \
    Hour, \
    SimpleAggregator, \
    custom_generator, \
    SingleAggregator, \
    Average, \
    DateTimeInfo, \
    ZipCodeInfo
import pandas as pd


def test_not_implemented():
    with pytest.raises(TypeError, match="Can't instantiate abstract class FeatureGenerator with abstract methods "
                                        "generate_feature"):
        FeatureGenerator()


def test_hour(timestamps):
    df = timestamps
    df_comp = df

    for idx, time in enumerate(df.timestamp):
        df_comp.timestamp[idx] = time.hour

    new_feature = Hour.generate_feature(df)

    assert new_feature.equals(df_comp)
    assert Hour.name == 'Hour'
    assert Hour.feature_type == 'transformation'


def test_hour_columns(timestamps):
    df = timestamps
    df_comp = df.copy()

    for idx, time in enumerate(df.timestamp):
        df_comp.timestamp[idx] = time.hour

    new_feature = Hour.generate_feature(df, column='timestamp')

    assert new_feature.equals(df_comp)


def test_split_timestamp(timestamps):
    df = timestamps
    split_time = DateTimeInfo.generate_feature(df.copy(), 'timestamp')

    assert all([d1.year == d2 for d1, d2 in zip(df['timestamp'], split_time['timestamp_year'])])
    assert all([d1.month == d2 for d1, d2 in zip(df['timestamp'], split_time['timestamp_month'])])
    assert all([d1.day == d2 for d1, d2 in zip(df['timestamp'], split_time['timestamp_day'])])
    assert all([d1.weekday() == d2 for d1, d2 in zip(df['timestamp'], split_time['timestamp_weekday'])])
    assert all([d1 == d2 for d1, d2 in zip(df['timestamp'].dt.time, split_time['timestamp_time'])])
    assert DateTimeInfo.name == 'datetime_info'
    assert DateTimeInfo.feature_type == 'generation'


def test_split_timestamp_no_col(timestamps):
    df = timestamps

    with pytest.raises(ValueError, match='timestamp column must be given'):
        DateTimeInfo.generate_feature(df.copy())


def test_zipcode_info(zipcodes):
    df = zipcodes
    df_comp = df.copy()

    searcher = SearchEngine(simple_zipcode=True)
    df_comp['state'] = ''
    df_comp['county'] = ''
    df_comp['city'] = ''
    df_comp['lat'] = ''
    df_comp['lng'] = ''
    df_comp['timezone'] = ''

    for zipcode in df_comp['zip_code'].unique():
        zip_search = searcher.by_zipcode(zipcode)
        df_comp.loc[df_comp['zip_code'] == zipcode, 'city'] = zip_search.major_city
        df_comp.loc[df_comp['zip_code'] == zipcode, 'county'] = zip_search.county
        df_comp.loc[df_comp['zip_code'] == zipcode, 'lat'] = zip_search.lat
        df_comp.loc[df_comp['zip_code'] == zipcode, 'lng'] = zip_search.lng
        df_comp.loc[df_comp['zip_code'] == zipcode, 'state'] = zip_search.state
        df_comp.loc[df_comp['zip_code'] == zipcode, 'timezone'] = zip_search.timezone

    zip_info = ZipCodeInfo.generate_feature(df, 'zip_code')

    assert zip_info.equals(df_comp)
    assert ZipCodeInfo.name == 'zipcode_info'
    assert ZipCodeInfo.feature_type == 'generation'


def test_zipcode_info_no_col(zipcodes):
    df = zipcodes

    with pytest.raises(ValueError, match='zipcode column must be given'):
        ZipCodeInfo.generate_feature(df)


def test_custom_feature_generator(user_test_func, single_dataset):
    usr_func = user_test_func
    df = single_dataset
    name = 'test_func'
    feature_type = 'test_type'
    test_generator = custom_generator(usr_func, name, feature_type)

    assert test_generator.name == name
    assert test_generator.feature_type == feature_type

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


def test_simple_define_relationship_errors(multi_dataset):
    df1, df2 = multi_dataset
    with pytest.raises(KeyError, match="need both keys for a relationship"):
        SimpleAggregator(df1, df2, rkey1='type')
    with pytest.raises(KeyError, match="need multiple dataframes to define the relationship"):
        SimpleAggregator(df1, rkey1='type', rkey2='type')


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
    new_cols.extend([col + '_avg' for col in df if col != 'species'])
    avgs = pd.DataFrame(avgs)
    avgs.columns = new_cols

    df_comp = pd.merge(df_comp, avgs, on='species')

    agg = Average(df, "species")

    df_avg = agg.aggregate()

    assert df_avg.equals(df_comp)
