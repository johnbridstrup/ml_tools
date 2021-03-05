import pandas as pd


class DataClass:

    def __init__(self, data=None, name=None, **kwargs):
        if name is None:
            if data is None:
                self._names = 'NO DATA'
                self.data = None
            else:
                self._names = {'data1'}
                self.data = {'data1': data}
        else:
            self._names = {name}
            self.data = {name: data}
        for key, kdata in kwargs.items():
            if not isinstance(kdata, pd.DataFrame):
                raise TypeError('kwargs must be dataframes')
            self._names.add(key)
            self.data.update({key: kdata})

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        for name, data in self.data.items():
            print("{}\n{}\n\n".format(name, data))


class Relationship:

    def __init__(self, many, one):
        pass
