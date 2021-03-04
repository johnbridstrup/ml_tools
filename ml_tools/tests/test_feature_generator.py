import pytest

from ml_tools import FeatureGenerator


def test_not_implemented():
    with pytest.raises(TypeError, match="Can't instantiate abstract class FeatureGenerator with abstract methods "
                                        "generate_feature"):
        FeatureGenerator()
