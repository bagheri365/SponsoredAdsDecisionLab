import csv
import json

import pytest

from sponsored_ads.retrieval_evaluation_runner import evaluate, stream_sample_clicked


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clk", "adgroup_id", "cate_his", "brand_his", "btag_his"])
        writer.writeheader()
        writer.writerows(rows)


def row(ad, clk="1", cat="cat", brand="brand", tag="1"):
    return dict(clk=clk, adgroup_id=ad, cate_his=cat, brand_his=brand, btag_his=tag)


def test_deterministic_sample_and_click_counts(tmp_path):
    path = tmp_path / "test.csv"
    write_csv(path, [row(str(i)) for i in range(30)] + [row("x", clk="0")])
    a, clicks, invalid = stream_sample_clicked(path, sample_size=7, seed=17)
    b, _, _ = stream_sample_clicked(path, sample_size=7, seed=17)
    assert a == b
    assert clicks == 30 and invalid == 0 and len(a) == 7
    assert len({n for n, _ in a}) == 7
    assert a != stream_sample_clicked(path, sample_size=7, seed=18)[0]


def test_malformed_clicked_history_is_counted(tmp_path):
    path = tmp_path / "test.csv"
    write_csv(path, [row("a"), row("b", cat="a^b"), row("c", clk="0")])
    sample, clicks, invalid = stream_sample_clicked(path, sample_size=10, seed=1)
    assert (clicks, invalid, len(sample)) == (2, 1, 1)


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("clk,adgroup_id\n1,a\n")
    with pytest.raises(ValueError, match="missing columns"):
        stream_sample_clicked(path, sample_size=1, seed=1)


def test_invalid_click_label_rejected(tmp_path):
    path = tmp_path / "test.csv"
    write_csv(path, [row("a", clk="yes")])
    with pytest.raises(ValueError, match="invalid clk"):
        stream_sample_clicked(path, sample_size=1, seed=1)


def test_invalid_sample_and_pool(tmp_path):
    with pytest.raises(ValueError, match="sample_size"):
        stream_sample_clicked(tmp_path / "missing", sample_size=0, seed=1)
    with pytest.raises(ValueError, match="pool_size"):
        evaluate("x", "y", pool_size=99)


def test_end_to_end_fixture_recall_and_coverage(tmp_path):
    test = tmp_path / "test.csv"
    catalog = tmp_path / "catalog.json"
    write_csv(test, [row("a"), row("missing", cat="other", brand="other"), row("a", clk="0")])
    catalog.write_text(json.dumps([
        {"adgroup_id": "a", "customer": "u", "cate_id": "cat", "brand": "brand", "impressions": 20, "clicks": 1},
        {"adgroup_id": "b", "customer": "u", "cate_id": "other", "brand": "other", "impressions": 10, "clicks": 0},
    ]))
    result = evaluate(test, catalog, sample_size=10)
    assert result["full_test_clicked_impressions"] == 2
    assert result["sampled_clicked_impressions"] == 2
    assert result["catalog_ads"] == 2
    for k in ("10", "50", "100"):
        summary = result["summaries"][k]
        assert summary["catalog_coverage"] == 0.5
        assert summary["methods"]["popularity"]["hits"] == 1
        assert summary["methods"]["fused"]["catalog_conditional_recall"] == 1.0


def test_empty_clicked_split_reports_null(tmp_path):
    test = tmp_path / "test.csv"
    catalog = tmp_path / "catalog.json"
    write_csv(test, [row("a", clk="0")])
    catalog.write_text("[]")
    result = evaluate(test, catalog)
    assert result["sampled_clicked_impressions"] == 0
    assert result["query_latency_ms"] is None
    assert result["summaries"]["10"]["catalog_coverage"] is None


def test_full_test_catalog_coverage_and_paired_metrics(tmp_path):
    test = tmp_path / "test.csv"
    catalog = tmp_path / "catalog.json"
    write_csv(test, [row("a"), row("a"), row("missing"), row("b", cat="x^y"), row("a", clk="0")])
    catalog.write_text(json.dumps([
        {"adgroup_id": "a", "customer": "u", "cate_id": "cat", "brand": "brand", "impressions": 5, "clicks": 1},
        {"adgroup_id": "b", "customer": "u", "cate_id": "x", "brand": "x", "impressions": 3, "clicks": 0},
    ]))
    result = evaluate(test, catalog, sample_size=2, seed=1)
    full = result["full_test_catalog_coverage"]
    assert full["valid_clicked_impressions"] == 3
    assert full["clicked_ads_in_catalog"] == 2
    assert full["coverage"] == pytest.approx(2 / 3)
    for k in ("10", "50", "100"):
        assert result["paired_recall_differences"][k]["fused_minus_popularity"]["n"] == 2
        assert result["paired_recall_differences"][k]["fused_minus_affinity"]["n"] == 2
