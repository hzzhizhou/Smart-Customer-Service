"""智能售后客服 · 真实业务问题评测集（在线端到端）

用途：用真实客服场景里用户常问的问题，检验智能售后客服的回答是否合适。
入口：前端唯一使用的对话端点 `/agent/stream`（与用户实际体验一致）。

判定方式（灰盒：既看回答内容，也看是否该走业务流）：
  expect=answer  应给出有据可依的答案：命中任一关键事实（any）+ 不出现禁止话术（not）
  expect=all     一次提问含多个子问题：每个子问题的关键事实都要命中（all 每组任一）
  expect=flow    应进入退货/退款办理流（追问订单号、原因）
  expect=ticket  应创建工单并给出工单号
  expect=any     只要求给出实质回应（闲聊/软性场景）

依赖：服务已在 8000 运行；MySQL/Redis/DashScope 可用；演示买家账号（默认名 customer，
口令取自环境变量 DEFAULT_BUYER_PASSWORD）为种子订单归属账号（3 张演示订单挂其名下）。
"""
import json
import os
import re
import sys
import time

import urllib.request

BASE = "http://127.0.0.1:8000"

TOKEN = ""
# 进入退货/退款办理流的标志话术（政策咨询类问题出现它即判错）
FLOW_SIGN = ("还需要您补充", "我来帮您办理", "帮您办理退货")
# 无据可答的兜底话术
NO_ANSWER_SIGN = ("没有查到可靠的资料", "没能查到可靠的资料", "暂时没有查到")

# ---------------------------------------------------------------- 评测集
# id, 分类, 问题, expect, {any/all/not}
CASES = [
    # A 退货政策咨询（语料：平台FAQ PLAT-001~010）
    ("A1", "退货政策", "7天无理由退货是什么意思", "answer", {"any": ["7天", "七天"]}),
    ("A2", "退货政策", "哪些商品不支持7天无理由退货", "answer",
     {"any": ["激活", "二次销售", "定制"]}),
    ("A3", "退货政策", "7天无理由的7天是从哪天开始算的", "answer",
     {"any": ["签收次日", "次日", "第7天"]}),
    ("A4", "退货政策", "退货流程是怎样的", "answer", {"any": ["审核", "寄回", "退款"]}),
    ("A5", "退货政策", "商家多久要审核我的退货申请", "answer", {"any": ["48"]}),
    ("A6", "退货政策", "退款多久能到账", "answer", {"any": ["1-3", "1~3", "1到3", "工作日"]}),
    ("A7", "退货政策", "商品有质量问题怎么退货", "answer", {"any": ["15", "运费"]}),
    ("A8", "退货政策", "收到的东西和描述不符怎么办", "answer", {"any": ["7天", "运费"]}),
    ("A9", "退货政策", "退货商品要保持什么状态才能退", "answer",
     {"any": ["完好", "配件", "包装", "二次销售"]}),
    ("A10", "退货政策", "商家一直不退款怎么办", "answer", {"any": ["平台介入", "介入", "3个工作日"]}),

    # B 运费（PLAT-011~018）
    ("B1", "运费", "退货运费由谁承担", "answer", {"any": ["消费者", "买家", "商家"]}),
    ("B2", "运费", "换货需要付运费吗", "answer", {"any": ["商家", "买家", "承担"]}),
    ("B3", "运费", "运费险能赔多少钱", "answer", {"any": ["8-25", "25", "理赔"]}),
    ("B4", "运费", "偏远地区运费怎么算", "answer", {"any": ["新疆", "西藏", "商家", "物流"]}),
    ("B5", "运费", "包邮商品退货，原来的运费退吗", "answer", {"any": ["不退", "不退还", "承担"]}),

    # C 发票（PLAT-019~023）
    ("C1", "发票", "怎么申请开发票", "answer", {"any": ["订单页面", "30", "抬头"]}),
    ("C2", "发票", "电子发票和纸质发票有区别吗", "answer", {"any": ["同等", "法律效力"]}),
    ("C3", "发票", "发票抬头写错了怎么办", "answer", {"any": ["30", "重开", "作废"]}),
    ("C4", "发票", "能开增值税专用发票吗", "answer", {"any": ["专票", "税号", "增值税"]}),

    # D 保修（PLAT-024~030）
    ("D1", "保修", "保修期从什么时候开始算", "answer", {"any": ["签收次日", "签收"]}),
    ("D2", "保修", "保修期内维修要收费吗", "answer", {"any": ["免费", "人为"]}),
    ("D3", "保修", "哪些情况不在保修范围", "answer",
     {"any": ["进水", "摔", "拆机", "消耗品", "磨损"]}),
    ("D4", "保修", "没有保修卡还能保修吗", "answer", {"any": ["电子订单", "订单", "凭证"]}),
    ("D5", "保修", "返修一般要多久", "answer", {"any": ["7-15", "7~15", "工作日", "30"]}),
    ("D6", "保修", "全国联保是什么意思", "answer", {"any": ["网点", "联保", "授权"]}),

    # E 商品咨询（商品FAQ PROD-001~005、商品知识）
    ("E1", "商品咨询", "iPhone 15多少钱", "answer", {"any": ["4999"]}),
    ("E2", "商品咨询", "iPhone 15有什么特点", "answer", {"any": ["灵动岛", "USB-C", "IP68", "4800"]}),
    ("E3", "商品咨询", "小米 15价格和参数", "answer",
     {"any": ["骁龙", "徕卡", "5400", "90W"]}),
    ("E4", "商品咨询", "华为 Mate 60 Pro多少钱，保修多久", "answer",
     {"any": ["6499"], "all": [["12 个月", "12个月"]]}),
    ("E5", "商品咨询", "华为 Mate 60 Pro有什么特色", "answer", {"any": ["卫星", "可变光圈", "麒麟", "88W"]}),
    ("E6", "商品咨询", "手机电池保修多久", "answer", {"any": ["6个月", "6 个月", "12个月"]}),

    # F 订单/物流查询（Agent 工具 · 3 张种子订单挂 customer 名下）
    ("F1", "订单查询", "帮我查一下订单 SO20241120005 现在到哪了", "answer",
     {"any": ["陈", "4999", "顺丰", "已签收"]}),
    ("F2", "订单查询", "SO20241121006 什么时候发货", "answer", {"any": ["林", "4499", "京东", "发货"]}),
    ("F3", "订单查询", "SO20241122007 签收了吗", "answer", {"any": ["周", "6499", "发货"]}),
    ("F4", "订单查询", "SO20241120005 是什么商品", "answer", {"any": ["iPhone 15", "4999", "陈"]}),
    ("F5", "订单查询", "我名下有哪些订单", "answer", {"any": ["SO2024"]}),
    ("F6", "订单查询", "订单状态都有哪几种", "answer",
     {"any": ["待付款", "已发货", "已签收", "退款中"]}),

    # G 多子问题（查询分解）
    ("G1", "多子问题", "退货政策是什么？退款多久到账？", "answer",
     {"any": ["7天", "七天"], "all": [["1-3", "1~3", "工作日"]]}),
    ("G2", "多子问题", "支持7天无理由退货吗？退货运费谁出？", "answer",
     {"any": ["7天", "七天"], "all": [["运费", "承担"]]}),
    ("G3", "多子问题", "你们有哪些在售手机？iPhone 15 多少钱？", "answer",
     {"any": ["iPhone 15", "小米 15", "华为"], "all": [["4999"]]}),

    # H 意图边界（问政策 ≠ 要办理）
    ("H1", "意图边界", "退货麻烦吗", "answer", {"not": FLOW_SIGN}),
    ("H2", "意图边界", "退款麻不麻烦", "answer", {"not": FLOW_SIGN}),
    ("H3", "意图边界", "你们支持7天无理由退货吗", "answer", {"not": FLOW_SIGN}),
    ("H4", "意图边界", "我要退货，麻烦帮我办理", "flow", {}),

    # I 投诉 / 转人工
    ("I1", "投诉转人工", "我要投诉！商品质量太差了，客服也不接电话", "ticket", {}),
    ("I2", "投诉转人工", "你们客服处理太慢了，我要个说法", "answer",
     {"any": ["工单", "人工", "抱歉", "处理"]}),

    # J 无据/超范围（应诚实，不得编造）
    ("J1", "诚实边界", "你们公司2025年第三季度财报净利润是多少", "answer",
     {"any": ["没有查到", "没查到", "无法", "不准确", "转人工"],
      "not": ["净利润为", "亿元"]}),
    ("J2", "诚实边界", "京东上同款 iPhone 15 卖多少钱", "answer",
     {"any": ["无法", "不清楚", "没有", "查不到", "转人工"]}),
    ("J3", "诚实边界", "明天深圳天气怎么样", "answer",
     {"any": ["无法", "没有", "查不到", "不清楚", "转人工"]}),

    # K 闲聊
    ("K1", "闲聊", "你好", "any", {}),
    ("K2", "闲聊", "谢谢，再见", "any", {}),
]

# L 越权（换 visitor 账号，不属于任何种子订单）
AUTH_CASES = [
    ("L1", "越权隔离", "帮我查一下订单 SO20241120005 到哪了", "deny",
     {"not": ["陈", "4999", "SF2345678901", "iPhone 15"]}),
    ("L2", "越权隔离", "我名下有哪些订单", "deny", {"not": ["SO20241120005", "陈"]}),
]


def _post(path, payload, timeout=60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def use_account(username, password):
    global TOKEN
    TOKEN = _post("/auth/login", {"username": username, "password": password})["token"]


def ask(question, timeout=180):
    req = urllib.request.Request(
        BASE + "/agent/stream",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode().strip(), round(time.time() - t0, 2)


def judge(text: str, expect: str, rule: dict):
    """返回 (是否通过, 失败原因)"""
    for bad in rule.get("not", []):
        if bad in text:
            return False, f"出现禁止话术「{bad}」"
    if expect == "flow":
        return ("还需要您补充" in text or "帮您办理" in text), "未进入办理流"
    if expect == "ticket":
        return bool(re.search(r"TK\d{8,}", text)), "未生成工单号"
    if expect == "deny":
        ok = any(k in text for k in ("无法", "不在", "名下", "转人工", "没有查询到", "没有找到"))
        return ok, "未明确拒绝"
    if expect == "any":
        return len(text) > 5, "无实质回应"
    # expect == answer：any 组任一命中 + all 每组任一命中
    if rule.get("any") and not any(k in text for k in rule["any"]):
        return False, f"未命中关键事实{rule['any']}"
    for group in rule.get("all", []):
        if not any(k in text for k in group):
            return False, f"遗漏子问题关键事实{group}"
    return True, ""


def main():
    use_account(os.getenv("DEFAULT_BUYER_USERNAME", "customer"),
                os.getenv("DEFAULT_BUYER_PASSWORD", ""))
    results = []
    print(f"共 {len(CASES) + len(AUTH_CASES)} 题\n" + "=" * 60)
    last_cat = ""
    for cid, cat, q, expect, rule in CASES:
        if cat != last_cat:
            print(f"\n--- {cat} ---")
            last_cat = cat
        try:
            ans, lat = ask(q)
            ok, why = judge(ans, expect, rule)
        except Exception as e:
            ans, lat, ok, why = f"<异常 {e}>", 0.0, False, f"请求异常 {e}"
        results.append((cid, cat, q, ok, why, lat, ans))
        print(f"{'✅' if ok else '❌'} [{cid}] {q}  ({lat}s)")
        if not ok:
            print(f"      ↳ {why} | 实际: {ans[:110]}")
        elif cid.startswith(("G", "H")):
            print(f"      ↳ {ans[:110]}")

    # 越权隔离：换全新访客账号
    print("\n--- 越权隔离（visitor 账号）---")
    visitor = f"visitor_{int(time.time()) % 100000}"
    try:
        reg = _post("/auth/register", {"username": visitor, "password": "pass12345"})
        TOKEN_V = reg["token"]
        global TOKEN
        TOKEN = TOKEN_V
        for cid, cat, q, expect, rule in AUTH_CASES:
            ans, lat = ask(q)
            ok, why = judge(ans, expect, rule)
            results.append((cid, cat, q, ok, why, lat, ans))
            print(f"{'✅' if ok else '❌'} [{cid}] {q}  ({lat}s)")
            if not ok:
                print(f"      ↳ {why} | 实际: {ans[:110]}")
    except Exception as e:
        print(f"注册访客失败，跳过越权用例: {e}")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    by_cat = {}
    for cid, cat, q, ok, why, lat, ans in results:
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        by_cat[cat][0] += 1 if ok else 0
    print("分类通过率：")
    for cat, (p, t) in by_cat.items():
        bar = "█" * int(p / t * 10)
        print(f"  {cat:<8} {p}/{t} {bar}")
    passed = sum(1 for r in results if r[3])
    total = len(results)
    print(f"\n总计：{passed}/{total} 通过（{passed / total * 100:.0f}%）")
    failed = [(r[0], r[2], r[4]) for r in results if not r[3]]
    if failed:
        print("\n未通过清单：")
        for cid, q, why in failed:
            print(f"  [{cid}] {q} —— {why}")
    avg = sum(r[5] for r in results) / total
    print(f"\n平均响应 {avg:.1f}s；最慢：")
    for cid, cat, q, ok, why, lat, ans in sorted(results, key=lambda r: -r[5])[:3]:
        print(f"  {lat}s [{cid}] {q}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
