import pytest
import pandas as pd
import seaborn as sns

@pytest.fixture
def single_dataset():
    df = pd.DataFrame({"animal": ["dog", "cat", "monkey"],
                       "num_legs": [4, 4, 2],
                       "num_arms": [0, 0, 2]})
    return df


@pytest.fixture
def multi_dataset():
    df1 = pd.DataFrame({"animal": ["dog", "cat", "monkey", "lizard"],
                        "num_legs": [4, 4, 2, 4],
                        "num_arms": [0, 0, 2, 0],
                        "type": ["mammal", "mammal", "mammal", "reptile"]})
    df2 = pd.DataFrame({"type": ["mammal", "reptile", "insect"],
                        "blood_temp": ["warm", "cold", "idk"],
                        "birth_type": ["live", "eggs", "eggs"]})
    return df1, df2


@pytest.fixture
def timestamps():
    dt = pd.date_range("2018-01-01", periods=5, freq="H")
    df = pd.DataFrame({'timestamp': dt, "numbers": range(len(dt))})
    return df


@pytest.fixture
def zipcodes():
    df = pd.DataFrame({'zip_code': [90293, 20186]})
    return df


@pytest.fixture
def user_test_func():
    def t_func(data, column):
        return data
    return t_func


@pytest.fixture
def test_dataset():
    iris = sns.load_dataset('iris')
    return iris
