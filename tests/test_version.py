import kymatax


def test_version_exposed():
    assert hasattr(kymatax, "__version__")


def test_version_value():
    assert kymatax.__version__ == "0.1.0"

