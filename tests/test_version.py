import kinamax


def test_version_exposed():
    assert hasattr(kinamax, "__version__")


def test_version_value():
    assert kinamax.__version__ == "0.1.0"

