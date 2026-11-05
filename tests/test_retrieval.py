"""M04A retrieval contracts and streaming catalog smoke tests."""
import csv
import json

import pytest

from sponsored_ads.retrieval import (
    CatalogAd, RetrievalQuery, build_catalog, load_catalog, retrieve_popular, save_catalog,
)
from sponsored_ads.taobao import COLUMNS


def ad(ad_id, impressions, category='10', brand='20'):
    return CatalogAd(ad_id, 'customer', category, brand, impressions, 0)


def test_order_and_tie_break():
    result = retrieve_popular([ad('z', 2), ad('b', 5), ad('a', 5)], RetrievalQuery(), k=2)
    assert [item.adgroup_id for item in result] == ['a', 'b']


def test_filters_and_exclusions():
    ads = [ad('a', 9, '1', '2'), ad('b', 8, '1', '3'), ad('c', 7, '2', '2')]
    assert [x.adgroup_id for x in retrieve_popular(ads, RetrievalQuery(category='1', brand='2'))] == ['a']
    assert [x.adgroup_id for x in retrieve_popular(ads, RetrievalQuery(excluded_ad_ids=frozenset({'a'})))] == ['b', 'c']


def test_empty_and_zero_k():
    assert retrieve_popular([], RetrievalQuery()) == []
    assert retrieve_popular([ad('a', 1)], RetrievalQuery(), k=0) == []


def test_invalid_k_and_duplicate():
    with pytest.raises(ValueError):
        retrieve_popular([], RetrievalQuery(), k=-1)
    with pytest.raises(ValueError):
        retrieve_popular([ad('a', 1), ad('a', 2)], RetrievalQuery())


def test_invalid_catalog_counts():
    with pytest.raises(ValueError):
        CatalogAd('a', 'c', '1', '2', 1, 2)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for ad_id, click, category in rows:
            row = {name: '1' for name in COLUMNS}
            row.update(clk=str(click), adgroup_id=ad_id, cate_id=category,
                       cate_his='1^2', brand_his='3^4', btag_his='1^1')
            writer.writerow(row)


def test_stream_catalog_and_roundtrip(tmp_path):
    source = tmp_path / 'train.csv'
    write_csv(source, [('a', 1, '9'), ('a', 0, '9'), ('b', 0, '8')])
    ads, manifest = build_catalog(source)
    assert (manifest['rows_scanned'], manifest['clicks_scanned'], manifest['catalog_ads']) == (3, 1, 2)
    assert [(a.adgroup_id, a.impressions, a.clicks) for a in ads] == [('a', 2, 1), ('b', 1, 0)]
    save_catalog(ads, manifest, tmp_path / 'out')
    assert load_catalog(tmp_path / 'out' / 'catalog.json') == ads
    assert json.loads((tmp_path / 'out' / 'manifest.json').read_text())['point_in_time_verified'] is False


def test_limit_and_conflicting_metadata(tmp_path):
    source = tmp_path / 'train.csv'
    write_csv(source, [('a', 1, '9'), ('a', 0, '8')])
    ads, manifest = build_catalog(source, limit=1)
    assert manifest['rows_scanned'] == 1 and len(ads) == 1
    with pytest.raises(ValueError, match='conflicting metadata'):
        build_catalog(source)


def test_reject_test_file(tmp_path):
    with pytest.raises(ValueError, match='train.csv'):
        build_catalog(tmp_path / 'test.csv')
