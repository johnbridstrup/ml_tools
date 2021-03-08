import pandas as pd
from abc import ABC, abstractmethod


"""Module for generating, transforming and aggregating features of datasets

  Typical usage examples:

  feature_generator = GeneratorClass()
  new_data = feature_generator.generate_feature(data)
  
  data_aggregator = AggregatorClass(data, relationship key)
  aggregated_data = data_aggregator.aggregate()
"""


class FeatureGenerator(ABC):
    """Abstract base class for all feature generators and transformers

    Attributes:
        name: name of transformation or generator (string)
        feature_type: type of feature being generated (string)
    """

    @property
    def name(self):
        raise NotImplementedError('Name property is not implemented')

    @property
    def feature_type(self):
        raise NotImplementedError('feature_type property is not implemented')

    @abstractmethod
    def generate_feature(self, data, column=None, **kwargs):
        """Abstract method for creating and transforming features



        :param data: Pandas dataframe
        :param column: Column on which to perform operation
        :param kwargs: additional parameters for specific implementations
        :return: Pandas dataframe with new or transformed features
        """
        pass


class Hour(FeatureGenerator):
    """Transforms datetime data to hour of the day"""
    @property
    def name(self):
        return 'Hour'

    @property
    def feature_type(self):
        return 'transformation'

    @classmethod
    def generate_feature(cls, data, column=None, **kwargs):
        """Implements generate feature

        Converts all datetime data to hours or, if given a column, converts a single column to hours
        """
        if column is None:
            types = data.dtypes
            ts_indices = [idx for idx, t in enumerate(list(types)) if t == "datetime64[ns]"]
            print(ts_indices)
            for idx in ts_indices:
                for index, time in data.iloc[:, idx].iteritems():
                    data.iloc[:, idx][index] = time.hour
            return data
        else:
            for index, time in data[column].iteritems():  # No idea why i need iteritems here but not in the other loop
                data[column][index] = time.hour
            return data


def custom_generator(user_func, name='custom_feature', feature_type='custom_feature_type'):
    """Function for creating feature generators from user implemented funtions

    :param user_func: transformation or generation function written by user
    :param name: name of operation
    :param feature_type: returned feature type
    :return: Instantiated FeatureGenerator implementation
    """
    class CustomGenerator(FeatureGenerator):
        @property
        def name(self):
            return name

        @property
        def feature_type(self):
            return feature_type

        def generate_feature(self, data, column=None, **kwargs):
            return user_func(data, column, **kwargs)

    return CustomGenerator()


class Aggregator(ABC):
    """Abstract base class for aggregators

    Attributes:
        name: name of aggregator (string)
        aggregation_type: type of aggregation performed (string)
        relationships: relationship or column aggregation is performed over (string)

    """
    def __init__(self, data1, data2=None, rkey1=None, rkey2=None, label1='data1', label2='data2'):
        """

        :param data1: dataframe to be aggregated
        :param data2: dataframe with a one to many relationship to data1
        :param rkey1: key of data1 with which to aggregate over
        :param rkey2: key of data2 that is related to data1.rkey1
        :param label1: name of data1
        :param label2: name of data2
        """
        self._data = {label1: data1}
        self._label1 = label1
        self._label2 = label2
        self._rkey1 = rkey1
        self._rkey2 = rkey2
        self._relationships = {}
        if rkey1 is not None:
            if rkey2 is None:
                raise KeyError("need both keys for a relationship")
            if data2 is None:
                raise KeyError("need multiple dataframes to define the relationship")
            self._relationships[rkey1] = rkey2

        if data2 is not None:
            self._data[label2] = data2

    @property
    def relationships(self):
        """Returns a description of the relationship between data1 and data2

        String will be of the form: data1.rkey1 -> data2.rkey2

        :return: Relationship between data1 and data2
        """
        relationships = "\n RELATIONSHIPS:\n"
        for key, rel in self._relationships.items():
            relationships = relationships + f"{self._label1}.{key} -> {self._label2}.{rel}\n"

        return relationships

    def __getitem__(self, item):
        """returns data labelled by item"""
        return self._data[item]

    def keys(self):
        """returns the labels of the data"""
        return set(self._data.keys())

    @abstractmethod
    def aggregate(self):
        """Aggregates data stored in class"""
        pass

    def new_relationship(self, rkey1, rkey2):
        """Defines the relationship between data1 and data2"""
        self._relationships[rkey1] = rkey2
        self._rkey1 = rkey1
        self._rkey2 = rkey2


class SingleAggregator(Aggregator):
    """Counts occurrences of each unique value in a given column"""
    def __init__(self, data, label=None, rkey1=None):
        """

        :param data: data to be aggregated
        :param label: name of the dataset
        :param rkey1: column to aggregate over
        """
        super().__init__(data, data2="NONE", label1=label, rkey1=rkey1, rkey2="NONE")

    def aggregate(self):
        """Counts occurrences of each unique value in given column"""
        out_df = pd.DataFrame(self._data[self._label1][self._rkey1].value_counts()).reset_index()
        out_df.columns = [self._rkey1, 'count']
        return out_df


class SimpleAggregator(Aggregator):
    """Counts occurrences of values

    Gets unique values in a column of one dataframe and counts their occurrences in another
    """

    def aggregate(self):
        """Counts occurrences by defined relationship

        Gets unique values from data2.rkey2 and counts their occurrences in data1.rkey1

        :return: dataframe with unique values from data2.rkey2 and their count in data1.rkey1
        """
        rel_values = self._data[self._label2][self._rkey2].unique()

        agg_df = pd.DataFrame(self._data[self._label1][self._rkey1].value_counts()).reset_index()
        agg_df.columns = [self._rkey1, 'count']

        for v in list(rel_values):
            if v not in list(agg_df[self._rkey1]):
                print(self._rkey1)
                print(v)
                s = pd.DataFrame({self._rkey1: [v], 'count': [0]})
                agg_df = agg_df.append(s, ignore_index=True)

        return agg_df


class Average(Aggregator):
    """Calculates average values of additional data for each unique value in a specified column"""

    def __init__(self, data, key, label=None):
        """Initializes aggregator with super()

        :param data: data to be aggregated
        :param key: column to aggregate data over
        :param label: name of dataset
        """
        super().__init__(data, data2="NONE", label1=label, rkey1=key, rkey2="NONE")

    def aggregate(self):
        """Performs aggregation

        :return: dataframe with the averages of each column over the repeated occurrences in key column
        """
        out_df = pd.DataFrame(self._data[self._label1][self._rkey1].value_counts()).reset_index()
        cols = [self._rkey1, 'count']
        out_df.columns = cols
        elements = out_df[self._rkey1]

        avgs = []
        for el in list(elements):
            agg_data = self._data[self._label1].loc[self._data[self._label1][self._rkey1] == el]
            x = [el]
            x.extend([agg_data[col].mean() for col in agg_data if col != self._rkey1])
            avgs.append(x)
        new_cols = [self._rkey1]
        new_cols.extend([col+'_avg' for col in self._data[self._label1] if col != self._rkey1])
        avgs = pd.DataFrame(avgs)
        avgs.columns = new_cols

        out_df = pd.merge(out_df, avgs, on=self._rkey1)
        return out_df
