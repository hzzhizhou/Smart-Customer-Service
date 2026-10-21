"""metadata 过滤条件求值：向量库与 BM25 必须用同一套语义。

为什么单独抽出来（PROJECT_AUDIT §4.2(1)）：
Chroma 的 where 支持 `$in`/`$ne` 等算子，而 BM25 路的后置过滤此前只做
`metadata.get(k) == v` 等值比较。多类别预过滤构造的是 `{"doc_category": {"$in": [...]}}`，
BM25 一比较就把所有文档判为不匹配 —— 整条 BM25 路被清空，混合检索静默退化成纯向量
（日志上仍写着 hybrid_retriever）。把判据收到一处，两路才不会再各自漂移。
"""

_UNSUPPORTED = object()


def _eval_one(value, expected) -> bool:
    """求值单个字段的条件：字面量等值，或 `$in`/`$eq`/`$ne` 算子。"""
    if not isinstance(expected, dict):
        return value == expected
    if "$in" in expected:
        return value in expected["$in"]
    if "$ne" in expected:
        return value != expected["$ne"]
    if "$eq" in expected:
        return value == expected["$eq"]
    return _UNSUPPORTED


def matches_filter(metadata: dict, condition=None) -> bool:
    """判断一条文档的 metadata 是否满足过滤条件。

    空条件视为「不过滤」。条件里的多个字段是「与」关系，需全部满足。
    不认识的算子按不匹配处理（fail-closed）：宁可少召回，也不要静默放大范围。
    """
    if not condition:
        return True
    for key, expected in condition.items():
        result = _eval_one(metadata.get(key), expected)
        # 不认识的算子：判为不匹配（fail-closed）；正常求值结果直接决定该字段是否通过
        if result is _UNSUPPORTED or not result:
            return False
    return True
