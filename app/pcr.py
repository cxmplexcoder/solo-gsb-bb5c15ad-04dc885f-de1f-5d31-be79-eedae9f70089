"""PCR 扩增产物预测（只读）。

口径
----
* 模板为已保存记录的原始 IUPAC 序列与其线性/环形拓扑，不依赖酶切结果。
* 正引物（forward，结合于 top 链、5'->3' 向右延伸）以**原序列**在 top 链上
  匹配；反引物（reverse，结合于 bottom 链、5'->3' 向左延伸）以其
  **反向互补序列**在 top 链上匹配。两处都按 IUPAC 碱基集合"有交集即匹配"
  判定（复用 :func:`app.iupac.bases_match`）。
* 每条引物允许**至多一处错配**，但其 **3' 端最后两位必须精确匹配**
  （IUPAC 集合相交但不完全相容也算错配，见下）。
* 只枚举**相向**（forward 在左、reverse 在右，延伸方向彼此指向）且
  **结合区不重叠**的位点对；产物从正引物 5' 锚点起、到反引物 5' 锚点止，
  即覆盖 ``[f_start, r_start + k_rev)`` 的整个弧。
* 线性模板：结合区不得越界，产物不得越过分子端点；
  环形模板：结合区可跨原点（锚点仍归一化到 ``[0, L)``），产物弧可跨原点，
  但弧长（产物长度）不得超过一圈（``<= L``）。
* 产物长度不得超过请求的 ``max_product_length``（其本身不超过模板长度）。

错配口径
--------
沿引物 5'->3' 逐位比较其实际比对到的模板碱基：IUPAC 可能集合**无交集**
计为一个错配；有交集即相容（包含简并符号"可能"配对的情形）。反引物在
top 链上的比对序列为其反向互补，反向互补保持集合相交关系，因此直接在
该比对串上逐位计数即可。引物 3' 端两位（正引物为比对串末尾、反引物为
比对串开头）无交集即拒绝该位点，不计入"至多一处错配"的额度。
"""

from __future__ import annotations

from .iupac import bases_match, reverse_complement

# 每条引物允许的候选结合位点数上限：位点对枚举为两侧位点的笛卡尔积，
# 高简并引物在长模板上可能命中海量位点，超过该上限时无法在保证耗时的
# 前提下完成全枚举，路由以 422 拒绝。
MAX_BINDING_SITES = 1000

FORWARD = "+"
REVERSE = "-"


class TooManyBindingSitesError(Exception):
    """某条引物的候选结合位点数超过 :data:`MAX_BINDING_SITES`。"""

    def __init__(self, side: str, primer: str, count: int):
        self.side = side
        self.primer = primer
        self.count = count
        super().__init__(
            f"{side} primer {primer!r} has {count} candidate binding sites; "
            f"PCR product enumeration supports at most {MAX_BINDING_SITES} per primer"
        )


def find_bindings(
    sequence: str, primer: str, circular: bool, strand: str
) -> list[dict]:
    """枚举一条引物在模板上的全部合规结合位点（允许重叠、环形可跨原点）。

    ``strand`` 为 ``"+"`` 时以引物原序列比对 top 链，为 ``"-"`` 时以引物
    的反向互补序列比对 top 链。锚点为结合区在 top 链上的 5' 端起点
    （反引物锚点即其右侧、延伸方向一侧的边界），范围归一化到 ``[0, L)``；
    线性模板只保留完整落在分子内的结合区。

    每个位点返回 ``strand / start / end / length / wraps_origin /
    mismatches``：环形结合区跨原点时 ``end`` 为取模坐标（可能小于
    ``start``）、``wraps_origin`` 为真。位点数超过
    :data:`MAX_BINDING_SITES` 时抛出 :class:`TooManyBindingSitesError`。
    """
    length = len(sequence)
    k = len(primer)
    aligned = primer if strand == FORWARD else reverse_complement(primer)
    # 3' 端最后两位在比对串中的下标：正引物在末尾、反引物在开头
    anchor_indices = {k - 2, k - 1} if strand == FORWARD else {0, 1}

    if k == 0 or k > length:
        return []
    haystack = sequence + sequence[: k - 1] if circular else sequence
    last_start = length if circular else length - k + 1

    bindings: list[dict] = []
    for start in range(last_start):
        mismatches = 0
        for j in range(k):
            if j in anchor_indices:
                # 3' 端最后两位不允许错配：直接拒绝该位点
                if not bases_match(aligned[j], haystack[start + j]):
                    break
            elif not bases_match(aligned[j], haystack[start + j]):
                mismatches += 1
                if mismatches > 1:
                    break
        else:
            end_raw = start + k
            bindings.append(
                {
                    "strand": strand,
                    "start": start,
                    "end": end_raw % length if circular else end_raw,
                    "length": k,
                    "wraps_origin": circular and end_raw > length,
                    "mismatches": mismatches,
                }
            )

    if len(bindings) > MAX_BINDING_SITES:
        side = "forward" if strand == FORWARD else "reverse"
        raise TooManyBindingSitesError(side, primer, len(bindings))
    return bindings


def _binding_key(binding: dict) -> tuple[int, int]:
    return (binding["start"], binding["length"])


def predict_pcr_products(
    sequence: str,
    topology: str,
    forward_primer: str,
    reverse_primer: str,
    max_product_length: int,
) -> list[dict]:
    """枚举全部合规 PCR 产物，按 (start, length) 去重并升序返回。

    调用方须已保证序列与引物仅含 IUPAC 碱基、引物至少 2 碱基，且
    ``1 <= max_product_length <= len(sequence)``。整个计算只在内存中
    进行，不读写数据库。
    """
    length = len(sequence)
    circular = topology == "circular"

    forward_sites = find_bindings(sequence, forward_primer, circular, FORWARD)
    reverse_sites = find_bindings(sequence, reverse_primer, circular, REVERSE)
    # 位点按 (锚点, 长度) 排序：长度固定时即按锚点，保证枚举与去重确定
    forward_sites.sort(key=_binding_key)
    reverse_sites.sort(key=_binding_key)

    kf = len(forward_primer)
    kr = len(reverse_primer)
    min_span = kf + kr

    products: dict[tuple[int, int], dict] = {}
    for fwd in forward_sites:
        f = fwd["start"]
        for rev in reverse_sites:
            r = rev["start"]
            if circular:
                # 相向弧：自 f 沿正向到反引物结合区末端，弧长 1..L
                span = (r + kr - f) % length
                if span == 0:
                    span = length  # 反引物末端恰在 f：产物为一整圈
                # 结合区不重叠 + 不超过一圈 + 长度上限
                if span < min_span or span > max_product_length:
                    continue
                start, product_length = f, span
            else:
                # 线性：必须 f 的结合区在左、r 的结合区在右且不重叠
                if r < f + kf:
                    continue
                span = r + kr - f
                if span > max_product_length:
                    continue
                start, product_length = f, span

            # 产物沿 top 链 5'->3' 自主引物锚点提取；环形弧跨原点时拼接
            if circular and start + product_length > length:
                product_sequence = (
                    sequence[start:] + sequence[: (start + product_length) % length]
                )
            else:
                product_sequence = sequence[start : start + product_length]

            product = {
                "start": start,
                "end": (start + product_length) % length if circular
                else start + product_length,
                "length": product_length,
                "sequence": product_sequence,
                "wraps_origin": circular and start + product_length > length,
                "forward_binding": fwd,
                "reverse_binding": rev,
                "mismatches": fwd["mismatches"] + rev["mismatches"],
            }
            # 同 (起点, 长度) 去重：保留总错配数更少的位点对；再相同则
            # 保留先枚举到者（位点已排序，结果确定）
            existing = products.get((start, product_length))
            if existing is None or product["mismatches"] < existing["mismatches"]:
                products[(start, product_length)] = product

    return [products[key] for key in sorted(products)]
