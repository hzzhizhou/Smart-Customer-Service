"""售后流程按订单状态分流：这是业务规则，必须有断言兜住。

背景（实测发现的缺陷）：第一版把所有可退订单塞进同一套 6 步流程，
于是「已付款未发货」的用户会被要求「7 天内寄回商品」——没有货可寄，
用户照做不了，客服也接不住。

分流依据是「商品此刻在哪」，而不是用户说的是退货还是退款：
    paid / pending → undelivered（商家拦截，直接退款，无寄回）
    shipped        → in_transit（拒收，物流退回）
    delivered      → received（寄回 → 验收）
    仅退款         → refund_only（无实物退回）
"""
from shared.reply_templates import (FLOW_BY_SCENARIO, render_stage_timeline,
                                    return_flow_brief, scenario_of, stage_flow,
                                    stage_step)


def _keys(category: str, status: str) -> list:
    return [s["key"] for s in stage_flow(category, status)]


# ---------------- 场景判定 ----------------
def test_paid_order_is_undelivered_even_for_return():
    """已付款未发货：即便用户说的是「退货」，也没有货可寄。"""
    assert scenario_of("return", "paid") == "undelivered"
    assert scenario_of("return", "pending") == "undelivered"


def test_shipped_order_is_in_transit():
    assert scenario_of("return", "shipped") == "in_transit"


def test_delivered_order_is_received():
    assert scenario_of("return", "delivered") == "received"


def test_refund_only_applies_after_delivery_not_before():
    """分类是退款也要先看商品在哪：货没发出去时不存在「不退货只退款」这个选择。"""
    assert scenario_of("refund", "delivered") == "refund_only"
    assert scenario_of("refund", "paid") == "undelivered"
    assert scenario_of("refund", "shipped") == "in_transit"


def test_unknown_status_falls_back_to_conservative_flow():
    """状态缺失时按已签收处理：多画一步寄回，好过漏掉用户要做的动作。"""
    assert scenario_of("return", "") == "received"
    assert scenario_of("return", "some_new_status") == "received"


# ---------------- 流程内容 ----------------
def test_undelivered_has_no_mail_back_stage():
    keys = _keys("return", "paid")
    assert keys == ["submitted", "intercepting", "refunding", "completed"]
    # 未发货流程绝不能出现寄回相关环节
    assert "approved" not in keys and "shipped" not in keys and "received" not in keys


def test_in_transit_uses_reject_not_mail_back():
    keys = _keys("return", "shipped")
    assert "rejecting" in keys and "returning" in keys
    assert "approved" not in keys          # 不该让在途用户「7 天内寄回」
    assert "shipped" not in keys           # 「寄回商品」是在用户手上才有的环节


def test_delivered_keeps_full_mail_back_flow():
    keys = _keys("return", "delivered")
    assert keys == ["submitted", "approved", "shipped", "received", "refunding", "completed"]


def test_refund_only_skips_mail_back_and_acceptance():
    keys = _keys("refund", "delivered")
    assert keys == ["submitted", "approved", "refunding", "completed"]


def test_every_scenario_flow_is_well_formed():
    """每条流程都必须能查到定义，且首尾一致（以提交申请开始、以退款到账结束）。"""
    for scenario, keys in FLOW_BY_SCENARIO.items():
        flow = stage_flow("return" if scenario != "refund_only" else "refund",
                          {"undelivered": "paid", "in_transit": "shipped",
                           "received": "delivered", "refund_only": "delivered"}[scenario])
        assert [s["key"] for s in flow] == keys, scenario
        assert keys[0] == "submitted" and keys[-1] == "completed", scenario
        assert len(set(keys)) == len(keys), f"{scenario} 存在重复环节"


# ---------------- 步数与轨迹 ----------------
def test_step_count_follows_scenario():
    """同一个「提交申请」，在未发货是 1/4，在已签收是 1/6。"""
    assert stage_step("submitted", "return", "paid") == (1, 4)
    assert stage_step("submitted", "return", "delivered") == (1, 6)
    assert stage_step("intercepting", "return", "paid") == (2, 4)
    assert stage_step("rejecting", "return", "shipped") == (2, 6)


def test_non_sequential_stage_returns_zero_step():
    assert stage_step("rejected", "return", "delivered") == (0, 6)


def test_timeline_lists_only_this_scenario_stages():
    """未发货的轨迹里不能出现「寄回商品 / 商家验收」——那是不会发生的环节。"""
    text = render_stage_timeline([], "submitted", "return", "paid")
    assert "商家拦截" in text
    assert "寄回商品" not in text and "商家验收" not in text
    assert text.count("○") + text.count("▶") == 4


def test_flow_brief_matches_scenario():
    assert "寄回商品" not in return_flow_brief("return", "paid")
    assert "拒收包裹" in return_flow_brief("return", "shipped")
    assert "寄回商品" in return_flow_brief("return", "delivered")
