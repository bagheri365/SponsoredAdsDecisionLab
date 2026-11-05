import json

import pytest

from sponsored_ads.behavioral_retrieval import ShopperHistory, retrieve_fused
from sponsored_ads.retrieval_integration import catalog_record_to_ad, load_observed_catalog
from sponsored_ads.retrieval_integration_cli import main


def record(adgroup_id="1", cate_id="12", brand="34", impressions=10, clicks=1):
    return dict(adgroup_id=adgroup_id, customer="99", cate_id=cate_id,
                brand=brand, impressions=impressions, clicks=clicks)


def test_adapter_preserves_identifiers():
    ad = catalog_record_to_ad(record())
    assert (ad.adgroup_id, ad.cate_id, ad.brand, ad.impressions) == ("1", "12", "34", 10)


@pytest.mark.parametrize("change", [{"clicks": 11}, {"impressions": -1},
                                    {"impressions": True}, {"brand": None},
                                    {"customer": ""}, {"adgroup_id": 1}])
def test_rejects_bad_records(change):
    with pytest.raises(ValueError):
        catalog_record_to_ad({**record(), **change})


def test_missing_field():
    bad = record()
    del bad["clicks"]
    with pytest.raises(ValueError):
        catalog_record_to_ad(bad)


def test_load_and_retrieve(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([record(), record("2", "55", "66", 20, 0)]))
    index = load_observed_catalog(path)
    hits = retrieve_fused(index, ShopperHistory.from_delimited("12", "34", "1"), 2)
    assert hits[0].adgroup_id == "1"
    assert set(hits[0].sources) == {"affinity", "popularity"}


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([record(), record()]))
    with pytest.raises(ValueError, match="duplicate"):
        load_observed_catalog(path)


def test_rejects_non_list(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="JSON list"):
        load_observed_catalog(path)


def test_cli_writes_report(tmp_path):
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "smoke.json"
    catalog.write_text(json.dumps([record()]))
    main(["--catalog", str(catalog), "--cate-his", "12", "--brand-his", "34",
          "--btag-his", "1", "--queries", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["catalog_ads"] == 1
    assert report["results"][0]["adgroup_id"] == "1"
    assert report["query_count"] == 2
