"""PCR 扩增产物预测接口与核心算法测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pcr import find_bindings, predict_pcr_products


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, sequence, topology="linear"):
    resp = client.post(
        "/digests",
        json={
            "sequence": sequence,
            "topology": topology,
            "enzymes": [
                {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _predict(client, digest_id, forward, reverse, max_product_length):
    payload = {
        "forward_primer": forward,
        "reverse_primer": reverse,
        "max_product_length": max_product_length,
    }
    return client.post(f"/digests/{digest_id}/pcr-products", json=payload)


# ---------- 线性模板基本产物 ----------


def test_linear_basic_product(client):
    # 正引物 AAAA 锚于 0；反引物 CCCC 在 top 链以 GGGG 比对，锚于 12
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 16)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["id"] == digest_id
    assert data["topology"] == "linear"
    assert data["sequence_length"] == 16
    assert data["forward_primer"] == "AAAA"
    assert data["reverse_primer"] == "CCCC"
    assert data["max_product_length"] == 16
    assert data["product_count"] == 1
    product = data["products"][0]
    assert product == {
        "start": 0,
        "end": 16,
        "length": 16,
        "sequence": "AAAATTTTCCCCGGGG",
        "wraps_origin": False,
        "forward_binding": {
            "strand": "+", "start": 0, "end": 4, "length": 4,
            "wraps_origin": False, "mismatches": 0,
        },
        "reverse_binding": {
            "strand": "-", "start": 12, "end": 16, "length": 4,
            "wraps_origin": False, "mismatches": 0,
        },
        "mismatches": 0,
    }


def test_products_sorted_by_start_then_length(client):
    # 正引物 AA 在全 A 背景上多个锚点；反引物结合区固定在右端
    digest_id = _create(client, "AATAATTTTGGGG")  # L=13
    resp = _predict(client, digest_id, "AA", "CCCC", 13)
    assert resp.status_code == 200
    starts_lengths = [(p["start"], p["length"]) for p in resp.json()["products"]]
    assert starts_lengths == sorted(starts_lengths)
    # 正引物精确命中 0、3（1 处错配的位点仍允许：1 处 T vs A 非锚点错配）
    starts = {p["start"] for p in resp.json()["products"]}
    assert 0 in starts and 3 in starts


def test_max_product_length_filters_long_products(client):
    digest_id = _create(client, "AAAATTGGGG")  # 产物全长 10
    assert _predict(client, digest_id, "AAAA", "CCCC", 10).json()["product_count"] == 1
    resp = _predict(client, digest_id, "AAAA", "CCCC", 9)
    assert resp.status_code == 200
    assert resp.json()["products"] == []
    assert resp.json()["product_count"] == 0


def test_divergent_primers_produce_nothing(client):
    # 反引物结合区在左、正引物在右：延伸方向相背，不产生产物
    digest_id = _create(client, "GGGGTTAAAA")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 10)
    assert resp.status_code == 200
    assert resp.json()["products"] == []


def test_overlapping_binding_regions_produce_nothing(client):
    # 正引物锚于 1（带 1 处非锚点错配），与锚于 4 的反引物结合区重叠
    digest_id = _create(client, "TAAACCCC")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 8)
    assert resp.status_code == 200
    assert resp.json()["products"] == []


def test_linear_bindings_never_cross_ends(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    resp = _predict(client, digest_id, "GGGG", "AAAA", 16).json()
    # 线性下结合区不得越界；所有产物坐标均在 [0, L] 内
    for product in resp["products"]:
        assert 0 <= product["start"] < product["end"] <= 16
        assert product["wraps_origin"] is False


# ---------- 错配与 3' 端约束 ----------


def test_one_mismatch_allowed_outside_anchor(client):
    # TAAA vs AAAA：仅引物 5' 第一位错配，3' 端两位精确 -> 允许，总错配 1
    digest_id = _create(client, "TAAATTGGGG")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 10)
    product = resp.json()["products"][0]
    assert product["length"] == 10
    assert product["forward_binding"]["mismatches"] == 1
    assert product["mismatches"] == 1


def test_two_mismatches_rejected(client):
    # 五碱基引物 AAAAT 比对 TTAAT：前两位均错配 -> 拒绝
    digest_id = _create(client, "TTAATTTGGGG")
    resp = _predict(client, digest_id, "AAAAT", "CCCC", 11)
    assert resp.json()["products"] == []


@pytest.mark.parametrize("template", ["AATATTGGGG", "AAATTTGGGG"])
def test_forward_3prime_last_two_must_match(client, template):
    # AAAA 的 3' 端两位（比对串下标 2、3）出现错配 -> 位点被拒
    digest_id = _create(client, template)
    resp = _predict(client, digest_id, "AAAA", "CCCC", 10)
    assert resp.json()["products"] == []


def test_reverse_3prime_last_two_must_match(client):
    # 反引物 CCCC 以 GGGG 比对，其 3' 端两位对应比对串开头；位置 7 破坏锚点
    digest_id = _create(client, "AAAATTCGGG")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 10)
    assert resp.json()["products"] == []


def test_reverse_one_non_anchor_mismatch_allowed(client):
    digest_id = _create(client, "AAAATTGGCG")  # GGGG 比对 GGCG：非锚点 1 错配
    resp = _predict(client, digest_id, "AAAA", "CCCC", 10)
    product = resp.json()["products"][0]
    assert product["reverse_binding"]["mismatches"] == 1
    assert product["mismatches"] == 1


def test_iupac_intersection_matching(client):
    # R={A,G} 比对 C：集合无交集，计 1 处非锚点错配 -> 允许
    digest_id = _create(client, "ACAATTGGGG")
    product = _predict(client, digest_id, "ARAA", "CCCC", 10).json()["products"][0]
    assert product["forward_binding"]["mismatches"] == 1
    # R 比对 G：有交集且相容 -> 0 错配
    digest_id2 = _create(client, "AGAATTGGGG")
    product2 = _predict(client, digest_id2, "ARAA", "CCCC", 10).json()["products"][0]
    assert product2["forward_binding"]["mismatches"] == 0


def test_dedup_by_start_and_length_keeps_fewer_mismatches(client):
    # 反引物两个锚点（6 精确 / 8 带 1 错配）结束于同一位置 8，
    # 产生相同 (start=0, length=8) 产物：只保留错配更少者
    digest_id = _create(client, "AATTTTGGT")
    resp = _predict(client, digest_id, "AA", "CC", 9)
    products = resp.json()["products"]
    assert len(products) == 1
    assert products[0]["length"] == 8
    assert products[0]["mismatches"] == 0
    assert products[0]["reverse_binding"]["start"] == 6


# ---------- 环形模板 ----------


def test_circular_product_wraps_origin(client):
    # L=16：正引物 AAAA 锚于 12，反引物 CCCC（GGGG）锚于 0；
    # 相向弧 12 -> 绕原点 -> 4，长度 8，序列 AAAA+GGGG
    digest_id = _create(client, "GGGGTTTTCCCCAAAA", topology="circular")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 16)
    products = resp.json()["products"]
    wrapped = [p for p in products if p["start"] == 12 and p["length"] == 8]
    assert len(wrapped) == 1
    product = wrapped[0]
    assert product["sequence"] == "AAAAGGGG"
    assert product["end"] == 4
    assert product["wraps_origin"] is True


def test_circular_full_circle_adjacent_regions(client):
    # 反引物 GGGG（CCCC 比对）锚于 8..12，恰在正引物锚点 12 前闭合：
    # 弧长取模为 0 -> 产物为整整一圈（长度 L），结合区不重叠
    digest_id = _create(client, "GGGGTTTTCCCCAAAA", topology="circular")
    resp = _predict(client, digest_id, "AAAA", "GGGG", 16)
    product = resp.json()["products"][0]
    assert product["length"] == 16
    assert product["sequence"] == "AAAAGGGGTTTTCCCC"
    assert product["start"] == 12
    assert product["end"] == 12
    assert product["wraps_origin"] is True


def test_circular_product_never_longer_than_one_turn(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG", topology="circular")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 16)
    for product in resp.json()["products"]:
        assert product["length"] <= 16


def test_circular_wrapped_binding_reports_modulo_coordinates(client):
    # 反引物结合区自身跨原点：锚点 10，end 取模为 1
    digest_id = _create(client, "GGAAAAAAAGG", topology="circular")  # L=11
    products = _predict(client, digest_id, "AAAA", "CC", 11).json()["products"]
    wrapped_bindings = [
        p for p in products if p["reverse_binding"]["start"] == 10
    ]
    assert wrapped_bindings
    binding = wrapped_bindings[0]["reverse_binding"]
    assert binding["end"] == 1
    assert binding["wraps_origin"] is True


def test_linear_and_circular_differ_for_same_primers(client):
    sequence = "GGGGTTTTCCCCAAAA"
    linear_id = _create(client, sequence, topology="linear")
    circular_id = _create(client, sequence, topology="circular")
    linear = _predict(client, linear_id, "AAAA", "CCCC", 16).json()["products"]
    circular = _predict(
        client, circular_id, "AAAA", "CCCC", 16
    ).json()["products"]
    # 线性下 AAAA 在右端、GGGG 在左端：相背，无产物
    assert linear == []
    # 环形下可绕原点相向
    assert any(p["wraps_origin"] for p in circular)


# ---------- 状态码与校验 ----------


def test_missing_record_returns_404(client):
    resp = _predict(client, 999, "AAAA", "CCCC", 10)
    assert resp.status_code == 404
    assert "not found" in resp.text


def test_max_product_length_exceeds_template_returns_422(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    resp = _predict(client, digest_id, "AAAA", "CCCC", 17)
    assert resp.status_code == 422
    assert "max_product_length 17 exceeds template length 16" in resp.text


def test_non_positive_max_product_length_returns_422(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    for value in (0, -1):
        resp = _predict(client, digest_id, "AAAA", "CCCC", value)
        assert resp.status_code == 422


@pytest.mark.parametrize("field", ["forward_primer", "reverse_primer"])
def test_illegal_or_short_primer_returns_422(client, field):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    payload = {"forward_primer": "AAAA", "reverse_primer": "CCCC", "max_product_length": 16}
    for bad in ("", "ACGU", "A"):  # 空 / 非 IUPAC / 不足 2 碱基
        payload[field] = bad
        resp = client.post(f"/digests/{digest_id}/pcr-products", json=payload)
        assert resp.status_code == 422, (field, bad)


def test_missing_fields_return_422(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    resp = client.post(
        f"/digests/{digest_id}/pcr-products", json={"forward_primer": "AAAA"}
    )
    assert resp.status_code == 422


def test_lowercase_primers_normalized(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    resp = _predict(client, digest_id, "aaaa", "cccc", 16)
    assert resp.status_code == 200
    data = resp.json()
    assert data["forward_primer"] == "AAAA"
    assert data["reverse_primer"] == "CCCC"
    assert data["product_count"] == 1


def test_primer_longer_than_template_yields_empty(client):
    digest_id = _create(client, "ACGT")
    resp = _predict(client, digest_id, "ACGTACGT", "ACGT", 4)
    assert resp.status_code == 200
    assert resp.json()["products"] == []


def test_no_product_returns_empty_array(client):
    digest_id = _create(client, "AAAAAAAAAAAA")
    resp = _predict(client, digest_id, "CCCC", "GGGG", 12)
    assert resp.status_code == 200
    assert resp.json()["products"] == []
    assert resp.json()["product_count"] == 0


# ---------- 只读 ----------


def test_prediction_does_not_write_database(client):
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    before = client.get("/digests").json()
    record_before = client.get(f"/digests/{digest_id}").json()
    resp = _predict(client, digest_id, "AAAA", "CCCC", 16)
    assert resp.status_code == 200
    assert client.get("/digests").json() == before
    assert client.get(f"/digests/{digest_id}").json() == record_before


def test_prediction_ignores_saved_fragments(client):
    # 即便保存的酶切结果没有任何切口（酶不识别），预测仍只基于原始序列工作
    digest_id = _create(client, "AAAATTTTCCCCGGGG")
    saved = client.get(f"/digests/{digest_id}").json()
    assert all(frag["left_end"]["type"] == "none" for frag in saved["fragments"][:1])
    resp = _predict(client, digest_id, "AAAA", "CCCC", 16)
    assert resp.json()["product_count"] == 1


# ---------- 核心算法直接测试 ----------


def test_find_bindings_anchor_rules():
    # 正引物 AAAA：0 处精确；位置 1 为 1 处非锚点错配；锚点破坏的位点不出现
    bindings = find_bindings("TAAA", "AAAA", circular=False, strand="+")
    assert [(b["start"], b["mismatches"]) for b in bindings] == [(0, 1)]


def test_find_bindings_reverse_uses_reverse_complement():
    # CCCC 以 GGGG 比对 top 链
    bindings = find_bindings("AAAATTTTCCCCGGGG", "CCCC", circular=False, strand="-")
    assert [b["start"] for b in bindings] == [12]
    assert all(b["strand"] == "-" for b in bindings)


def test_find_bindings_linear_bounds():
    bindings = find_bindings("AAAA", "AAAA", circular=False, strand="+")
    assert [b["start"] for b in bindings] == [0]
    assert bindings[0]["end"] == 4
    assert bindings[0]["wraps_origin"] is False


def test_find_bindings_circular_wraps_origin():
    # 序列首尾拼成跨原点位点 GG：起点 10（位置 10=G, 0=G）
    bindings = find_bindings("GGAAAAAAAGG", "GG", circular=True, strand="+")
    wrapped = [b for b in bindings if b["start"] == 10]
    assert len(wrapped) == 1
    assert wrapped[0]["end"] == 1
    assert wrapped[0]["wraps_origin"] is True


def test_find_bindings_too_many_sites_raises():
    from app.pcr import MAX_BINDING_SITES, TooManyBindingSitesError

    with pytest.raises(TooManyBindingSitesError):
        find_bindings(
            "A" * (MAX_BINDING_SITES + 10),
            "N" * 3,
            circular=False,
            strand="+",
        )


def test_predict_core_returns_empty_for_overlapping_identical_primers():
    # 相同引物在均一模板上正、反结合区处处重叠
    assert predict_pcr_products("AAAAAAAA", "linear", "AAA", "AAA", 8) == []
    assert predict_pcr_products("AAAAAAAA", "circular", "AAA", "AAA", 8) == []
