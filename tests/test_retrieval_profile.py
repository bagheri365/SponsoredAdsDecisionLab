import json

import pytest

from sponsored_ads.behavioral_retrieval import BehavioralIndex, ObservedAd, ShopperHistory, retrieve_fused
from sponsored_ads.retrieval_profile import main, profile_phases


def example():
    index = BehavioralIndex([
        ObservedAd("1", "c1", "b1", 10),
        ObservedAd("2", "c2", "b2", 20),
        ObservedAd("3", "c1", "b3", 5),
    ])
    history = ShopperHistory.from_delimited("c1", "b1", "1")
    return index, history


def test_profile_preserves_retrieval_results():
    index, history = example()
    report = profile_phases(index, history, top_k=2, pool_size=3, repetitions=3)
    assert report["result_ids"] == [r.adgroup_id for r in retrieve_fused(index, history, 2, pool_size=3)]
    assert set(report["phase_latency_ms"]) == {"affinity", "popularity", "fusion"}
    assert "affinity" in report["affinity_cprofile_top20"]
    assert all(stats["median"] >= 0 for stats in report["phase_latency_ms"].values())


@pytest.mark.parametrize("kwargs", [
    {"repetitions": 0}, {"top_k": 0}, {"top_k": 4, "pool_size": 3},
])
def test_invalid_parameters(kwargs):
    index, history = example()
    with pytest.raises(ValueError):
        profile_phases(index, history, **kwargs)


def test_cli_writes_report(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([
        {"adgroup_id": "1", "customer": "x", "cate_id": "c1", "brand": "b1", "impressions": 10, "clicks": 1},
        {"adgroup_id": "2", "customer": "x", "cate_id": "c2", "brand": "b2", "impressions": 20, "clicks": 0},
    ]))
    output = tmp_path / "profile.json"
    main(["--catalog", str(catalog), "--cate-his", "c1", "--brand-his", "b1",
          "--btag-his", "1", "--top-k", "1", "--pool-size", "2",
          "--repetitions", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["catalog_ads"] == 2
    assert report["history_length"] == 1
    assert len(report["result_ids"]) == 1
