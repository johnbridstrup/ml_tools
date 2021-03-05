import pandas as pd
from abc import ABC, abstractmethod


class FeatureGenerator(ABC):

    # def __init__(self, data):
    #     self._data = data.copy()

    @abstractmethod
    def generate_feature(self, data, column=None, **kwargs):
        raise NotImplementedError("generate_feature must be implemented")


# class GenericGenerator(FeatureGenerator):
#     def generate_feature(self, data, column=None):
#         pass


class Hour(FeatureGenerator):

    @classmethod
    def generate_feature(cls, data, column=None, **kwargs):
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


def custom_generator(user_func):
    class CustomGenerator(FeatureGenerator):
        def generate_feature(self, data, column=None, **kwargs):
            return user_func(data, column, **kwargs)

    return CustomGenerator()


class Aggregator(FeatureGenerator):

    def __init__(self, data1, data2=None, rkey1=None, rkey2=None, label1='data1', label2='data2'):
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
        relationships = "\n RELATIONSHIPS:\n"
        for key, rel in self._relationships.items():
            relationships = relationships + f"{self._label1}.{key} -> {self._label2}.{rel}\n"

        return relationships

    def __getitem__(self, item):
        return self._data[item]

    def keys(self):
        return set(self._data.keys())

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def generate_feature(self, data, column=None, **kwargs):
        pass

    def new_relationship(self, rkey1, rkey2):
        self._relationships[rkey1] = rkey2
        self._rkey1 = rkey1
        self._rkey2 = rkey2


class SingleAggregator(Aggregator):
    def __init__(self, data, label=None, rkey1=None):
        super().__init__(data, data2="NONE", label1=label, rkey1=rkey1, rkey2="NONE")

    def generate_feature(self, data, column=None, **kwargs):
        pass

    def aggregate(self):
        out_df = pd.DataFrame(self._data[self._label1][self._rkey1].value_counts()).reset_index()
        out_df.columns = [self._rkey1, 'count']
        return out_df


class SimpleAggregator(Aggregator):
    """
    just counts appearances
    """

    def aggregate(self):
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

    def generate_feature(self, data, column=None, **kwargs):
        raise NotImplementedError("This aggregator doesn't generate new features")
