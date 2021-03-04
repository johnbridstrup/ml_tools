import pandas as pd
from abc import ABC, abstractmethod


class FeatureGenerator(ABC):

    @abstractmethod
    def generate_feature(self):
        raise NotImplementedError("generate_feature must be implemented")