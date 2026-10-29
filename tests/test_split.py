import pandas as pd 
import pytest
from time_series.evaluation.split import temporal_train_test_split

def test_temporal_split_preserves_order() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.date_range(
            start='2020-01-01',
            periods=5,
            freq='D',
        )
    )

    result = temporal_train_test_split(
        series=series,
        test_size=2,
    )

    assert result.train.tolist() == [1.0, 2.0, 3.0]
    assert result.test.tolist() == [4.0, 5.0]

def test_temporal_split_preserves_index() -> None:
    index = pd.date_range(
        start="2020-01-01",
        periods=5,
        freq='D',
    )
    series = pd.Series(
        range(5),
        index=index,
        dtype=float,
    )

    result = temporal_train_test_split(
        series=series,
        test_size=2,
    )

    assert result.train.index.equals(index[:3])
    assert result.test.index.equals(index[3:])

def test_temporal_split_rejects_unsorted_index() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-01",
                "2020-01-03",                
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="递增顺序",
    ):
        temporal_train_test_split(
            series=series,
            test_size=1,
        )

@pytest.mark.parametrize(
    'test_size',
    [0, -1, 5],
)
def test_temporal_split_rejects_invaid_size(
    test_size: int,
) -> None:
    series = pd.Series(
        range(5),
        dtype=float,
    )

    with pytest.raises(ValueError):
        temporal_train_test_split(
            series=series,
            test_size=test_size,
        )