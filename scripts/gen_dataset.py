#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指令数据集生成器 —— 企业考勤核算规则

背景：这个数据集不是凭空编的，规则来自本人实际交付的「企业考勤数据治理与核算自动化流水线」
项目（对接钉钉开放平台 API，250+ 员工月度考勤核算）。原项目里规则引擎是硬编码的 Python 判定；
这里把同一套规则转成「自然语言输入 → 结构化判定」的指令对，用于 LoRA 微调，使小模型能够
按公司口径复现判定，而不依赖写死的 if-else。

产物：train.jsonl / valid.jsonl（chat 格式，trl SFTTrainer 可直接读）
"""
import json
import os
import random

random.seed(20260916)

SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["建国", "秀英", "志强", "桂英", "海燕", "俊杰", "丽娜", "晓东", "文静", "鹏飞",
         "雅琴", "浩然", "思远", "秀兰", "国栋", "丹丹", "子涵", "宇航", "静怡", "嘉伟",
         "швея".replace("швея", "凤霞"), "建军", "晓雯", "立诚", "佩玲", "宏宇", "玉梅", "泽楷"]

WEEKDAY_NAME = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

WORK_START = "08:30"   # 上班达标线
WORK_END = "17:30"     # 下班达标线

SYS = ("你是企业考勤核算助手。请根据公司考勤口径判定单日打卡记录，"
       "仅输出 JSON，不要额外解释。字段：date, weekday, status, late_minutes, "
       "early_leave_minutes, overtime_minutes, issues, note。")


def rand_name():
    return random.choice(SURNAMES) + random.choice(GIVEN)


def t2m(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def m2t(v):
    return "%02d:%02d" % (v // 60, v % 60)


def rand_time(lo, hi):
    """返回 [lo, hi) 分钟区间内的随机时刻"""
    return random.randrange(lo, hi)


def is_weekend(wd):
    return wd >= 5


def build_case(kind):
    name = rand_name()
    day = random.randint(1, 30)
    wd = random.randint(0, 6)
    date = "2026-09-%02d" % day
    weekday = WEEKDAY_NAME[wd]
    weekend = is_weekend(wd)

    punch_in = rand_time(t2m("07:30"), t2m("08:25"))
    punch_out = rand_time(t2m("17:35"), t2m("19:30"))
    issues = []
    note_parts = []

    if kind == "normal":
        status, late, early, ot = "达标", 0, 0, 0
        if weekend:
            ot = punch_out - t2m(WORK_END)
            status = "休息日出勤"
            note_parts.append("休息日打卡计入加班")
    elif kind == "late":
        punch_in = rand_time(t2m("08:31"), t2m("10:20"))
        late = punch_in - t2m(WORK_START)
        status, early, ot = "迟到", 0, 0
        note_parts.append("上班打卡晚于 08:30，记迟到 %d 分钟" % late)
    elif kind == "early_leave":
        punch_out = rand_time(t2m("15:00"), t2m("17:29"))
        early = t2m(WORK_END) - punch_out
        status, late, ot = "早退", 0, 0
        note_parts.append("下班打卡早于 17:30，记早退 %d 分钟" % early)
    elif kind == "late_and_early":
        punch_in = rand_time(t2m("08:35"), t2m("09:40"))
        punch_out = rand_time(t2m("15:30"), t2m("17:20"))
        late = punch_in - t2m(WORK_START)
        early = t2m(WORK_END) - punch_out
        status, ot = "迟到+早退", 0
        note_parts.append("当日同时存在迟到与早退")
    elif kind == "overtime":
        punch_out = rand_time(t2m("19:00"), t2m("22:30"))
        ot = punch_out - t2m(WORK_END)
        status, late, early = "达标（含加班）", 0, 0
        note_parts.append("下班晚于 17:30，超出部分计入加班 %d 分钟" % ot)
    elif kind == "missing":
        status, late, early, ot = "缺卡", 0, 0, 0
        issues.append("缺卡")
        note_parts.append("当日仅有 1 次打卡，无法核算完整工时，需提报补卡审批")
        # 50% 概率只保留上班或下班
        if random.random() < 0.5:
            text = "%s %s(%s) 打卡 %s，无下班打卡记录。" % (name, date, weekday, m2t(punch_in))
        else:
            text = "%s %s(%s) 无上班打卡记录，打卡 %s。" % (name, date, weekday, m2t(punch_out))
        return text, {"date": date, "weekday": weekday, "status": status, "late_minutes": late,
                      "early_leave_minutes": early, "overtime_minutes": ot,
                      "issues": issues or ["无"], "note": "；".join(note_parts)}
    elif kind == "duplicate":
        dup = m2t(punch_in + random.randint(3, 20))
        dup2 = m2t(punch_out - random.randint(3, 20))
        status, late, early, ot = "达标", 0, 0, 0
        issues.append("重复打卡")
        note_parts.append("存在重复打卡，上班取最早一次 %s、下班取最晚一次 %s"
                          % (m2t(punch_in), m2t(punch_out)))
        text = ("%s %s(%s) 打卡记录：%s、%s、%s、%s，共 4 次。"
                % (name, date, weekday, m2t(punch_in), dup, dup2, m2t(punch_out)))
        return text, {"date": date, "weekday": weekday, "status": status, "late_minutes": late,
                      "early_leave_minutes": early, "overtime_minutes": ot,
                      "issues": issues, "note": "；".join(note_parts)}
    elif kind == "overnight":
        start = rand_time(t2m("20:00"), t2m("22:00"))
        end_raw = rand_time(t2m("01:00"), t2m("04:00"))
        ot = (end_raw + 24 * 60) - t2m(WORK_END)
        status, late, early = "跨天班次", 0, 0
        issues.append("跨天班次")
        note_parts.append("下班打卡时刻早于上班，判定为次日凌晨，加班时长按加 24 小时计算，共 %d 分钟" % ot)
        text = ("%s %s(%s) 上班打卡 %s，下班打卡 %s（次日凌晨）。"
                % (name, date, weekday, m2t(start), m2t(end_raw)))
        return text, {"date": date, "weekday": weekday, "status": status, "late_minutes": 0,
                      "early_leave_minutes": 0, "overtime_minutes": ot,
                      "issues": issues, "note": "；".join(note_parts)}
    elif kind == "makeup":
        status, late, early, ot = "达标（补卡）", 0, 0, 0
        issues.append("审批补卡")
        note_parts.append("原缺卡记录已关联已审批补卡单，按补卡时间 %s 计入" % m2t(punch_in))
        text = ("%s %s(%s) 上班打卡 %s，下班打卡 %s；另有一张已审批的上班补卡单，"
                "补卡时间 %s。" % (name, date, weekday, m2t(punch_in), m2t(punch_out), m2t(punch_in)))
        return text, {"date": date, "weekday": weekday, "status": status, "late_minutes": late,
                      "early_leave_minutes": early, "overtime_minutes": ot,
                      "issues": issues, "note": "；".join(note_parts)}

    # 非早退/迟到分支的加班计算
    if kind == "normal" and not weekend and punch_out > t2m(WORK_END) + 30:
        ot = punch_out - t2m(WORK_END)
        note_parts.append("下班晚于 17:30 超过 30 分钟，计入加班 %d 分钟" % ot)

    if not note_parts:
        note_parts.append("打卡时间符合考勤口径，无需特殊处理")
    if not issues:
        issues.append("无")

    if weekend and kind == "normal":
        text = ("%s %s(%s，休息日) 打卡 %s、%s。"
                % (name, date, weekday, m2t(punch_in), m2t(punch_out)))
    else:
        text = ("%s %s(%s) 上班打卡 %s，下班打卡 %s。"
                % (name, date, weekday, m2t(punch_in), m2t(punch_out)))

    return text, {"date": date, "weekday": weekday, "status": status, "late_minutes": late,
                  "early_leave_minutes": early, "overtime_minutes": ot,
                  "issues": issues, "note": "；".join(note_parts)}


KINDS = (["normal"] * 3 + ["late"] * 5 + ["early_leave"] * 4 + ["late_and_early"] * 3 +
         ["overtime"] * 5 + ["missing"] * 5 + ["duplicate"] * 3 + ["overnight"] * 4 +
         ["makeup"] * 3)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    os.makedirs(data_dir, exist_ok=True)

    rows = []
    for kind in KINDS * 5:                      # 约 175 条
        text, ans = build_case(kind)
        rows.append({
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "请判定以下考勤记录：\n" + text},
                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)},
            ]
        })
    random.shuffle(rows)
    n_valid = max(12, len(rows) // 10)
    valid, train = rows[:n_valid], rows[n_valid:]

    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(os.path.join(data_dir, "train.jsonl"), train)
    dump(os.path.join(data_dir, "valid.jsonl"), valid)
    print("train: %d 条 -> %s" % (len(train), os.path.join(data_dir, "train.jsonl")))
    print("valid: %d 条 -> %s" % (len(valid), os.path.join(data_dir, "valid.jsonl")))
    print("\n--- 样例（第 1 条）---")
    m = train[0]["messages"]
    print("[user]", m[1]["content"])
    print("[assistant]", m[2]["content"])


if __name__ == "__main__":
    main()
