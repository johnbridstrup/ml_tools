import pandas as pd
import pytest

from ml_tools import DataClass, Relationship


def test_dataclass_init_blank():
    blank_data = DataClass()

    assert blank_data._names == "NO DATA"
    assert blank_data.data is None


def test_data_init_single(single_dataset):
    df = single_dataset
    data = DataClass(df, name='animals')

    assert data._names == {'animals'}
    assert data['animals'].equals(df)


def test_bad_kwargs(single_dataset):
    df = single_dataset

    with pytest.raises(TypeError, match='kwargs must be dataframes'):
        data = DataClass(df, name='animals', bad_kwargs="test")


def test_data_init_multiple(multi_dataset):
    df1, df2 = multi_dataset
    data = DataClass(df1, 'animals', animal_type_info=df2)

    assert data._names == {'animals', 'animal_type_info'}
    assert data['animals'].equals(df1)
    assert data['animal_type_info'].equals(df2)


def test_basic_aggregation(multi_dataset):
    df1, df2 = multi_dataset
    data = DataClass(df1, 'animals', animal_type_info=df2)


def test_relationship_init():
    rel = Relationship(('animals', 'type'), ('animal_type_info', 'type'))
