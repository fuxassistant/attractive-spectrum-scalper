"""Render a frozen conditional experiment without fitting or selecting a model.

Run with .venv/Scripts/python.exe -m experiments.zone_conditional_v1.report OUTDIR.
The report verifies saved metadata, data, implementation and source identities.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "base": "Контекст",
    "memory": "+ память зоны",
    "memory_spectrum": "+ память + спектр",
    "memory_mbo": "+ память + MBO",
    "all": "+ память + спектр + MBO",
    "past_placebo": "Контекст + чужая память",
    "train_prior": "Частоты train",
}
SHORT = {
    "base": "Контекст", "memory": "Память", "memory_spectrum": "Память\n+ спектр",
    "memory_mbo": "Память\n+ MBO", "all": "Все", "past_placebo": "Чужая\nпамять",
    "train_prior": "Частоты\ntrain",
}
BLOCKS = {"train": "Обучение · 03–14.08", "calibration": "Калибровка · 17–21.08",
          "evaluation": "Проверка · 24–28.08", "extension": "Продолжение · 31.08–01.09"}
COLORS = {"base": "#516a7a", "memory": "#247e83", "memory_spectrum": "#7869a9",
          "memory_mbo": "#ba812f", "all": "#a74e57", "past_placebo": "#7c7c7c", "train_prior": "#b6b5ac"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value, places=0):
    if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "да" if value else "нет"
    if isinstance(value, str):
        return value
    return f"{value:,.{places}f}".replace(",", " ")


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def stamp(value):
    return pd.Timestamp(int(value), unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC"


def verify_inputs(out):
    manifest = read_json(out / "manifest.json")
    if manifest.get("status") != "COMPLETE_DIAGNOSTIC_ONLY":
        raise ValueError("Only COMPLETE_DIAGNOSTIC_ONLY conditional runs can be rendered")
    inputs = {"manifest.json": digest(out / "manifest.json")}
    verified = {}

    def verify(path, expected):
        path = Path(path).resolve()
        key = str(path)
        if key not in verified:
            verified[key] = digest(path)
        if verified[key] != expected:
            raise ValueError(f"Identity mismatch: {path}")
        inputs[key] = verified[key]

    for group in ("artifacts", "metadata_hashes"):
        for name, expected in manifest[group].items():
            verify(out / name, expected)
    for name, expected in manifest["implementation_sha256"].items():
        verify(ROOT / name, expected)
    cfg = read_json(out / "config.json")
    if cfg != read_json(Path(__file__).with_name("config.json")):
        raise ValueError("Saved and current frozen configurations differ")
    parent = ROOT / cfg["source_run"]
    verify(parent / "manifest.json", manifest["source_manifest_sha256"])
    for receipt in manifest["receipts"]:
        day = out / "days" / receipt["date"]
        if read_json(day / "receipt.json") != receipt:
            raise ValueError(f"Saved day receipt differs: {day}")
        inputs[str(day / "receipt.json")] = digest(day / "receipt.json")
        for name, expected in receipt["artifact_sha256"].items():
            verify(day / name, expected)
        for name, expected in receipt["source_hashes"].items():
            verify(name, expected)
        verify(parent / "days" / receipt["date"] / "receipt.json", receipt["source_receipt_sha256"])
    required = ["dataset", "predictive_metrics", "daily_metrics", "calibration", "contrasts",
                "policy_summary", "random_controls", "policy_selection", "policy_clock"]
    if any(name + ".parquet" not in manifest["artifacts"] for name in required):
        raise ValueError("Required report input is absent from the hash manifest")
    data = {name: pd.read_parquet(out / (name + ".parquet")) for name in required}
    for name in ("independent_review.json", "diagnostic_gates.json", "execution_review.json"):
        if (out / name).exists():
            inputs[str(out / name)] = digest(out / name)
    return manifest, cfg, data, inputs


def policy_row(frame, arm, block, condition="base"):
    selected = frame.loc[frame.arm.eq(arm) & frame.block.eq(block) & frame.condition.eq(condition)]
    if len(selected) != 1:
        raise ValueError(f"Expected one policy summary: {arm}/{block}/{condition}")
    return selected.iloc[0]


def paired_support(contrasts, arm, reference):
    records = contrasts.loc[contrasts.arm.eq(arm) & contrasts.reference.eq(reference)
                            & contrasts.block.isin(["evaluation", "extension"])]
    supported = len(records) == 4 and bool(records.delta.lt(0).all())
    return supported, records


def plot_audit(out, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(17.5, 15.5))
    metrics = data["predictive_metrics"]
    arms = list(ARMS)
    x = np.arange(len(arms))
    ax = axes[0, 0]
    for block, offset, color in [("evaluation", -.16, "#247e83"), ("extension", .16, "#ba812f")]:
        rows = metrics.loc[metrics.block.eq(block) & metrics.cohort.eq("all_known")].set_index("arm").reindex(arms)
        ax.bar(x + offset, rows.logloss, .30, color=color, label=BLOCKS[block])
    ax.set(title="1. Вероятностная ошибка на тех же известных исходах\nМеньше log loss — лучше; среднее по дням", ylabel="Мультиклассовый log loss", xticks=x,
           xticklabels=[SHORT[a] for a in arms])
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -.20))
    ax = axes[0, 1]
    c = data["contrasts"]
    pairs = [("memory", "base"), ("memory_spectrum", "memory"), ("memory_mbo", "memory"),
             ("all", "memory"), ("memory", "past_placebo")]
    labels = ["Память − контекст", "+ спектр − память", "+ MBO − память", "Все − память", "Память − чужая память*"]
    for block, shift, color in [("evaluation", -.13, "#247e83"), ("extension", .13, "#ba812f")]:
        rows = [c.loc[c.arm.eq(a) & c.reference.eq(b) & c.block.eq(block) & c.metric.eq("logloss")].iloc[0] for a, b in pairs]
        values = np.array([r.delta for r in rows])
        errors = np.array([[r.delta-r.ci_low for r in rows], [r.ci_high-r.delta for r in rows]])
        ax.errorbar(values, np.arange(len(rows))+shift, xerr=np.maximum(0, errors), fmt="o", color=color,
                    markersize=4, capsize=3, label=BLOCKS[block])
    ax.axvline(0, color="#555", lw=.8)
    ax.set(title="2. Добавочная информация: парная разница log loss\nСлева лучше; *только строки с прошлым донором", xlabel="Δ log loss · 95% описательный bootstrap по дням",
           yticks=np.arange(len(pairs)), yticklabels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], ls="--", color="#b0b5b7", lw=1, label="Идеальная калибровка")
    for arm in list(ARMS)[:-1]:
        rows = data["calibration"].loc[data["calibration"].arm.eq(arm) & data["calibration"].block.eq("evaluation")].sort_values("predicted")
        ax.plot(rows.predicted, rows.observed, color=COLORS[arm], marker="o", markersize=3.5, lw=1.25, label=ARMS[arm])
    ax.set(title="3. Возврат раньше дальнейшего удаления · 24–28.08\nДесять фиксированных bins, веса дня и момента", xlabel="Прогноз P(RETURN_FIRST)",
           ylabel="Наблюдаемая доля RETURN_FIRST", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")

    def plot_capital(ax, block, title):
        policies = list(ARMS)[:-1]
        x = np.arange(len(policies))
        for condition, offset, color, label in [("base", -.17, "#247e83", "К зоне · base"), ("stress", .17, "#a74e57", "К зоне · stress")]:
            rows = [policy_row(data["policy_summary"], a, block, condition) for a in policies]
            bars = ax.bar(x+offset, [r.net_usd for r in rows], .31, color=color, label=label)
            for bar, row in zip(bars, rows):
                if not row.capital_curve_complete:
                    bar.set_hatch("///")
        base = [policy_row(data["policy_summary"], a, block) for a in policies]
        away = [policy_row(data["policy_summary"], a, block, "away") for a in policies]
        ax.scatter(x, [r.random_p95_usd if pd.notna(r.random_p95_usd) else np.nan for r in base],
                   color="#282d30", marker="D", s=25, label="Случайный допуск · base p95", zorder=4)
        ax.scatter(x, [r.net_usd if r.capital_curve_complete else np.nan for r in away], color="#ba812f", marker="x", s=40,
                   label="Обратное направление · base", zorder=4)
        ax.axhline(0, color="#555", lw=.8)
        ax.set(title=title, ylabel="Чистый результат · USD", xticks=x,
               xticklabels=[SHORT[a]+f"\nn={int(r.trades)}" for a, r in zip(policies, base)])
        ax.legend(fontsize=7.7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0, -.21))

    plot_capital(axes[1, 1], "evaluation", "4. Отдельная механическая игра · 24–28.08\nОдин контракт, один счёт на модель и блок")
    plot_capital(axes[2, 1], "extension", "6. Те же правила участия · 31.08–01.09\nПорог заморожен на 17–21.08, без поиска по PnL")
    ax = axes[2, 0]
    target = data["dataset"]
    order = list(BLOCKS)
    bottom = np.zeros(len(order))
    labels = {"RETURN_FIRST": "Возврат первым", "ADVERSE_FIRST": "Удаление первым", "TIMEOUT": "Нет события за 300 с", "CENSORED": "Неизвестно"}
    for state, color in [("RETURN_FIRST", "#247e83"), ("ADVERSE_FIRST", "#a74e57"), ("TIMEOUT", "#ba812f"), ("CENSORED", "#c5cace")]:
        values = np.array([target.loc[target.block.eq(b), "state"].eq(state).mean() for b in order])*100
        ax.bar(np.arange(len(order)), values, bottom=bottom, color=color, label=labels[state], hatch="///" if state=="CENSORED" else None)
        bottom += values
    ax.set(title="5. Что произошло первым: состав всех кандидатов\nЭто доли строк; метрики выше балансируют дни и моменты", ylabel="Доля кандидатов · %", ylim=(0, 100),
           xticks=np.arange(len(order)), xticklabels=[b.split(" · ")[0]+f"\nn={target.block.eq(k).sum()}" for k,b in BLOCKS.items()])
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0, -.17))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Память ценовой зоны: добавочный прогноз и отдельная проверка капиталом", x=.055, ha="left", fontsize=18)
    fig.text(.055, .025, "Все даты ранее просматривались. CENSORED — маска неизвестного, а не четвёртый рыночный класс.\n"
             "Нижний log loss не гарантирует прибыль. Случайный допуск сохраняет только приближённый контекст/активность; p95 относится к base.\n"
             "Штриховка капиталов = незавершённый счёт; extension содержит лишь два дня. Интервалы bootstrap описательные.", fontsize=10, color="#48555d")
    fig.subplots_adjust(left=.065, right=.975, top=.93, bottom=.14, hspace=.67, wspace=.32)
    path = out / "zone_conditional_audit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def memory_example(cfg, data, inputs):
    d = data["dataset"]
    eligible = d.loc[d.block.eq("evaluation") & (d.t_ns//1_000_000_000 % 86400).ge(8*3600)].sort_values(["t_ns", "candidate_id"])
    if eligible.empty:
        raise ValueError("No evaluation candidate for the deterministic history example")
    chosen = eligible.iloc[0]
    path = ROOT / cfg["source_run"] / "days" / chosen.date / "history_tokens.parquet"
    if str(path.resolve()) not in inputs:
        raise ValueError("Example history was not source-hash verified")
    tokens = pd.read_parquet(path, filters=[("candidate_id", "==", int(chosen.candidate_id))]).sort_values("position")
    if tokens.empty or (tokens.available_ns>chosen.t_ns).any():
        raise ValueError("Invalid history example")
    rows = []
    for r in tokens.itertuples():
        rows.append([int(r.position), int(r.episode_id), stamp(r.available_ns).split(" ")[1], r.outcome,
                     int(r.entry_side), int(r.exit_side), fmt(r.core_dwell_seconds, 2), fmt(r.duration_seconds, 2),
                     fmt(r.previous_absence_seconds, 2), fmt(r.approach_efficiency, 3)])
    fields = ["x_mem_log_recent_edge_returns", "x_mem_log_recent_gate_aborts", "x_mem_log_max_probe_seconds", "x_mem_log_max_probe_distance"]
    decoded = [fmt(np.expm1(chosen[f]), 2) for f in fields]
    return chosen, rows, decoded


def local_section(out, cfg, inputs):
    """An optional separately frozen audit; never merged into model features."""
    local = out / "local"
    if not (local / "manifest.json").exists():
        return ""
    manifest = read_json(local / "manifest.json")
    if manifest.get("status") != "COMPLETE_DESCRIPTIVE_ONLY" or manifest.get("local_training_admission") != "NOT_INCLUDED_IN_MODEL":
        raise ValueError("Optional local evidence is not a completed separate descriptive audit")
    if manifest["parent_manifest_sha256"] != inputs["manifest.json"]:
        raise ValueError("Local evidence belongs to another conditional run")
    inputs[str(local / "manifest.json")] = digest(local / "manifest.json")

    def verify(path, expected):
        path = Path(path).resolve()
        key = str(path)
        actual = inputs.get(key)
        if actual is None:
            actual = digest(path)
        if actual != expected:
            raise ValueError(f"Local evidence identity mismatch: {path}")
        inputs[key] = actual

    for name, expected in manifest["implementation_sha256"].items():
        verify(ROOT / name, expected)
    for name, expected in manifest["artifacts"].items():
        verify(local / name, expected)
    source = ROOT / cfg["source_run"]
    verify(source / "manifest.json", manifest["source_episode_manifest_sha256"])
    for receipt in manifest["receipts"]:
        raw = Path(receipt["raw_dir"])
        verify(raw / "manifest.json", receipt["manifest_sha256"])
        for name, expected in receipt["payload_sha256"].items():
            verify(raw / name, expected)
        canonical = Path(receipt["canonical_market"])
        verify(canonical / "manifest.json", receipt["canonical_manifest_sha256"])
        for name, expected in receipt["canonical_payload_sha256"].items():
            verify(canonical / name, expected)
        day = source / "days" / receipt["date"]
        verify(day / "episodes.parquet", receipt["episode_sha256"])
        verify(day / "candidates.parquet", receipt["candidate_sha256"])
    local_data = pd.read_parquet(local / "local_mbo_diagnostics.parquet")
    rows = []
    for (window, region), group in local_data.groupby(["window", "region"]):
        rows.append(["При уходе" if window=="dep" else "При решении", "Ядро зоны" if region=="zone" else "Коридор до ухода",
                     len(group), int(group.trade_window_known.sum()), int(group.level_window_known.sum()), int(group.known.sum())])
    inventory = pd.read_parquet(local / "source_inventory.parquet")
    if int(inventory.source_present.sum()) != manifest["days_with_source"] or len(local_data) != manifest["rows"]:
        raise ValueError("Local inventory/count mismatch")
    return '<section><h2>Отдельная проверка: MBO непосредственно в зоне и коридоре</h2>' + \
        f'<p>Сохранены {fmt(manifest["rows"])} наблюдений для {manifest["days_with_source"]} дней с исходными sidecar. Для остальных {manifest["days_without_source"]} дней локального источника нет: {fmt(manifest["missing_source_candidates"])} кандидатов не получают выдуманных локальных потоков. Эта таблица не использовалась ни в одном обученном X.</p>' + \
        table(["Окно 30 с", "Ценовая область", "Строк", "T поддержан", "Уровни поддержаны", "Оба поддержаны"], rows) + \
        '<p>Точная лента T считается один раз; удаления разделяются по связи с F и известности роли. Коридор заморожен от зоны до цены подтверждённого ухода. Известный ноль допускается только после проверки непрерывности, полноты публикаций и сохранения сумм по всем ценам. Неполные наблюдаемые суммы сохранены с масками; по ним нельзя делать вывод об отсутствии интереса.</p>' + \
        '<p class="muted">Семь доступных дней не дают обучающего префикса для этой локальной ветви. Глобальный MBO в модели и локальный профиль зоны проверяются раздельно. Полная локальная поддержка окна не устанавливает намерения продавцов или покупателей.</p><p><a href="local/manifest.json">Локальный manifest</a> · <a href="local/local_mbo_diagnostics.parquet">Наблюдения и маски</a></p></section>'


def render(directory):
    out = Path(directory).resolve()
    manifest, cfg, data, inputs = verify_inputs(out)
    png = plot_audit(out, data)
    models = read_json(out / "models.json")
    thresholds = read_json(out / "thresholds.json")
    memory_ok, _ = paired_support(data["contrasts"], "memory", "base")
    headline = ("Память уменьшила обе ошибки в обоих последующих блоках" if memory_ok
                else "Устойчивое добавочное преимущество памяти пока не подтверждено")
    d = data["dataset"]
    chosen, memory_rows, decoded = memory_example(cfg, data, inputs)
    local_html = local_section(out, cfg, inputs)
    cards = [(fmt(manifest["days"]), "ранее просмотренных дней"), (fmt(len(d)), "кандидатов · все маски сохранены"),
             (fmt(int(d.label_known.sum())), "известных геометрических исходов"),
             (fmt(len(data["policy_clock"])), "общих моментов до score-фильтра")]
    target_rows = []
    for block in BLOCKS:
        g = d.loc[d.block.eq(block)]
        target_rows.append([BLOCKS[block], len(g), *[int(g.state.eq(x).sum()) for x in ["RETURN_FIRST", "ADVERSE_FIRST", "TIMEOUT", "CENSORED"]],
                            fmt(100*g.placebo_known.mean(), 1)+"%"])
    metric_rows = []
    m = data["predictive_metrics"]
    for block in ["calibration", "evaluation", "extension"]:
        for arm in ARMS:
            r = m.loc[m.block.eq(block) & m.arm.eq(arm) & m.cohort.eq("all_known")].iloc[0]
            metric_rows.append([BLOCKS[block], ARMS[arm], int(r.rows), int(r.days), fmt(r.logloss, 5), fmt(r.brier, 5), fmt(r.return_brier, 5)])
    contrast_rows = []
    for r in data["contrasts"].itertuples():
        if r.block == "calibration":
            continue
        contrast_rows.append([BLOCKS[r.block], ARMS[r.arm]+" − "+ARMS[r.reference], r.metric, int(r.days),
                              fmt(r.delta, 5), f"[{fmt(r.ci_low, 5)}; {fmt(r.ci_high, 5)}]",
                              "есть донор" if r.cohort=="donor_supported" else "все известные"])
    account_rows = []
    admission_rows = []
    for block in ["evaluation", "extension"]:
        for arm in list(ARMS)[:-1]:
            r = policy_row(data["policy_summary"], arm, block)
            st = policy_row(data["policy_summary"], arm, block, "stress")
            away = policy_row(data["policy_summary"], arm, block, "away")
            values = []
            for rr in [r, st, away]:
                values.append(fmt(rr.net_usd)+(" (неполный)" if not rr.capital_curve_complete else ""))
            account_rows.append([BLOCKS[block], ARMS[arm], int(r.proposed), int(r.trades), int(r.active_days), *values,
                                 fmt(r.random_median_usd), fmt(r.random_p95_usd), f"{int(r.random_complete_runs)}/{int(r.random_seeds)}",
                                 fmt(r.max_drawdown_usd), int(r.unresolved)])
            sufficient = bool(r.capital_curve_complete and st.capital_curve_complete and r.net_usd>0 and st.net_usd>0
                              and pd.notna(r.random_p95_usd) and r.net_usd>r.random_p95_usd and r.trades>=30 and r.active_days>=4)
            admission_rows.append((arm, block, sufficient))
    train_rows = [[ARMS[a], len(r["columns"]), len(r["mean"]), r["training_rows"], str(r["iterations"]),
                   stamp(r["max_training_label_ns"])] for a,r in models["models"].items()]
    threshold_rows = [[ARMS[a], fmt(r["threshold"], 5), r["calibration_rows"], fmt(100*r["acceptance_rate"], 1)+"%", stamp(r["available_ns"])]
                      for a,r in thresholds.items()]
    support_rows = []
    for block in BLOCKS:
        g = d.loc[d.block.eq(block)]
        cols = ["x_mbo_mbo_dep_known", "x_base_mbo_now_known", "x_spec_spectrum_known"]
        support_rows.append([BLOCKS[block], *[fmt(100*g[col].eq(1).mean(), 1)+"%" if col in g else "—" for col in cols],
                             fmt(100*g.placebo_known.mean(), 1)+"%"])
    donor_rows = []
    for block in ["calibration", "evaluation", "extension"]:
        for arm in ["memory", "past_placebo"]:
            rr=m.loc[m.block.eq(block)&m.arm.eq(arm)&m.cohort.eq("donor_supported")]
            if len(rr):
                r=rr.iloc[0]
                donor_rows.append([BLOCKS[block],ARMS[arm],int(r.rows),fmt(r.logloss,5),fmt(r.brier,5)])
    calibration_rows = [[BLOCKS[r.block], ARMS[r.arm], f"[{r.bin/10:.1f}; {(r.bin+1)/10:.1f}{']' if r.bin==9 else ')'}",
                         int(r.rows), fmt(r.weight, 2), fmt(r.predicted, 4), fmt(r.observed, 4)]
                        for r in data["calibration"].itertuples() if r.block in ["evaluation", "extension"]]
    censor_rows = [[BLOCKS[b], reason, int(len(g))] for (b,reason),g in d.loc[~d.label_known].groupby(["block","reason"])] if "reason" in d else []
    economic_count = sum(x[2] for x in admission_rows)
    summary_text = (f"По зафиксированным диагностическим условиям {economic_count} из {len(admission_rows)} сочетаний модели и последующего блока "
                    "имеют положительные завершённые base/stress счета, превышают base p95 случайного допуска и достаточны по числу сделок/дней. "
                    "Это не допуск к торговле: независимой записи в опыте нет.")
    picture = base64.b64encode(png.read_bytes()).decode("ascii")
    fragments = ["<!doctype html><html lang=\"ru\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>Условный опыт: память ценовой зоны</title><style>"
        ":root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#263640;background:#f3f5f5;line-height:1.6}body{margin:0}main{max-width:1480px;margin:auto;padding:36px 28px 70px}"
        "h1{font-size:clamp(29px,3.2vw,46px);line-height:1.14;max-width:1120px;letter-spacing:-.035em;margin:12px 0 20px}h2{font-size:25px;margin:0 0 12px;line-height:1.25}"
        "p{max-width:1120px;margin:10px 0 16px}.eyebrow{color:#427b80;text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:700}.muted{color:#657780}.badge{display:inline-block;background:#e4eced;padding:4px 10px;border-radius:4px;font-size:12px;font-weight:700}"
        ".cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card,section{background:white;border:1px solid #dce4e6;border-radius:10px}.card{padding:18px}.card b{display:block;font-size:29px;line-height:1.3}.card span{font-size:13px;color:#60757e}"
        "section{padding:25px;margin:20px 0}.callout{padding:15px 20px;background:#e7f0ef;border-left:4px solid #2f8384;max-width:1120px}.caution{background:#f7f0e6;border-left-color:#b88438}"
        "img{width:100%;display:block}.table-wrap{overflow-x:auto;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;background:#edf2f3;font-weight:600}td,th{padding:10px 12px;border-bottom:1px solid #e6ebed;white-space:nowrap}td:first-child,th:first-child{white-space:normal}tr:hover td{background:#f8fafb}"
        "code{font-family:Consolas,monospace;font-size:.9em;background:#edf2f3;padding:1px 4px;border-radius:3px}a{color:#246f77}details{margin-top:20px}summary{cursor:pointer;color:#316f76;font-weight:600}footer{font-size:12px;color:#667981;overflow-wrap:anywhere}@media(max-width:780px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}section{padding:18px}h2{font-size:22px}}"
        "</style><main><div class=\"eyebrow\">GCZ6 · условный опыт · 2026</div>",
        f"<h1>{esc(headline)}</h1><span class=\"badge\">COMPLETE · DIAGNOSTIC ONLY</span>",
        "<p>Проверяем, помогает ли история завершённых контактов с конкретной ценовой зоной предсказать следующее событие после учёта геометрии, текущего движения, волатильности, спреда, времени суток и активности. Спектр и глобальный MBO добавляются отдельными ветвями. Торговый счёт проверяется отдельно.</p>",
        '<div class="cards">'+''.join(f'<div class="card"><b>{esc(a)}</b><span>{esc(b)}</span></div>' for a,b in cards)+'</div>',
        '<div class="callout caution"><b>Все 22 даты ранее просматривались.</b> Хронологические блоки предотвращают обучение на более поздних строках в этом опыте, но не создают нетронутого holdout. Продолжение содержит только два дня; интервалы по дням показывают разброс, а не доказанную переносимость.</div>',
        '<section><h2>Одна популяция, шесть замороженных моделей</h2><p>Один multinomial logistic classifier на ветвь, C=0.1; медианы, missing indicators, масштабы и коэффициенты получены только на 03–14 августа. Модели не переобучаются на следующих блоках. Каждому дню даётся одинаковый вес; несколько зон в одном моменте делят вес момента.</p>',
        table(["Ветка", "Что добавлено к общему контексту"], [[ARMS["base"], "Расстояние/риск, ATR, спред, время, M1 движение и общая активность MBO"], [ARMS["memory"], "Последние ≤8 событий своей зоны, посещения, dwell, проходы, отбои и короткие пробы"], [ARMS["memory_spectrum"], "Спектр часовой occupancy, автокорреляция и интервалы посещений"], [ARMS["memory_mbo"], "Направленные потоки и состояние книги при уходе и сейчас; окна 30 с"], [ARMS["all"], "Память + спектр + глобальные потоки/книга"], [ARMS["past_placebo"], "Строго прошлая память другой абсолютной зоны в сходном грубом контексте"]]),
        '<p>Чужая память — причинно доступный контроль устаревшей информации, не рандомизированное вмешательство. Основное сравнение с ним ограничено одинаковыми строками, где донор найден. Контекст совпадает по грубым корзинам, поэтому остаточные различия возможны.</p></section>',
        f'<section><h2>Прогноз, калибровка и капитал</h2><img alt="Шесть панелей: ошибки моделей, парные различия, калибровка, состав таргета и два блока торгового результата" src="data:image/png;base64,{picture}"><p class="muted">Base и stress — отдельные последовательные счета. Нулевое участие остаётся WAIT. Случайный допуск использует те же общие моменты; его вероятность зависит от контекстной корзины и оценивается на калибровке.</p></section>',
        '<section><h2>Таргет: какое событие произошло первым</h2><p><code>RETURN_FIRST</code> — midpoint достиг или пересёк ближнюю замороженную границу зоны, включая перескок. <code>ADVERSE_FIRST</code> — раньше достигнута граница удаления на 1R от midpoint решения. <code>TIMEOUT</code> — ни одно событие не наступило на непрерывно поддержанном пути за 300 секунд. Это геометрия; ни fill, ни прибыль в определение не входят.</p><p><code>CENSORED</code> остаётся неизвестным наблюдением и исключается из вероятностных метрик; будущая маска не фильтрует допуск к торговле. Временные bins 30/60/120/300 с сохранены для аудита; отдельная hazard-модель в этом опыте не обучалась.</p>',
        table(["Блок", "Все строки", "Возврат первым", "Удаление первым", "Timeout", "Неизвестно", "Есть прошлый донор"], target_rows),
        table(["Блок", "Причина неизвестного", "Строк"], censor_rows) if censor_rows else "", "</section>",
        '<section><h2>Ошибка вероятностей и добавочная информация</h2><p>Log loss и multiclass Brier учитывают все три известных исхода. В таблице Δ отрицательное значение означает улучшение относительно указанной ветви. Проверка памяти требует уменьшения обеих ошибок в обоих последующих блоках; это диагностический критерий, а не проверка прибыли.</p>',
        table(["Блок", "Сравнение", "Метрика", "Дней", "Δ", "95% интервал", "Когорта"], contrast_rows),
        '<details><summary>Все абсолютные ошибки, включая постоянные частоты train</summary>', table(["Блок", "Ветка", "Строк", "Дней", "Log loss", "Brier 3 класса", "Brier возврата"], metric_rows), "</details>",
        '<details><summary>Честное сравнение со старой памятью другой зоны: только общая поддержка</summary>', table(["Блок", "Ветка", "Строк с донором", "Log loss", "Brier"], donor_rows), "</details>",
        '<details><summary>Поддержка каждой точки калибровки: строки и суммарный вес</summary><p>Редкие крайние scores могут соответствовать единичным наблюдениям. Соединение точек на рисунке помогает читать профиль, но не является оценкой непрерывной функции калибровки.</p>',
        table(["Блок", "Ветка", "Bin score", "Строк", "Сумма весов", "Прогноз", "Наблюдаемый возврат"], calibration_rows), "</details></section>",
        '<section><h2>Механическая игра после затрат</h2><p>В каждом общем моменте ближайшая допустимая зона выбрана до модельного score. Допуск: <code>P(RETURN_FIRST) ≥ порог</code>; порог — 80-й процентиль scores 17–21 августа, без использования их исходов. Нет выбора порога по PnL, будущей квоты сделок или отбора по известности будущего.</p><p>Исполнение и цены выходов наследуются из проверенного zone_episode_lab: один GC, BBO после 100 мс, комиссия $30 за круг; stress — 250 мс и один тик проскальзывания на каждое исполнение. Один открытый контракт, лимиты риска и остановки счёта те же. Score прогнозирует геометрический возврат, а не ожидаемые доллары.</p>',
        f'<div class="callout caution">{esc(summary_text)}</div>',
        table(["Блок", "Ветка", "Допущено", "Сделок", "Дней", "Net base $", "Net stress $", "Обратно $", "Random медиана $", "Random p95 $", "Завершено random", "Max DD $", "Не закрыто"], account_rows),
        '<p class="muted">Random — 99 детерминированных Bernoulli-допусков с вероятностью по контекстной корзине, оценённой на калибровке и сглаженной к общей частоте. Это приближённое совпадение активности, а не точное совпадение числа будущих сделок. Медиана и p95 скрываются при хотя бы одном незавершённом случайном счёте. p95 всегда относится к базовому исполнению.</p>',
        '<details><summary>Замороженные score-пороги и время их доступности</summary>', table(["Ветка", "Порог", "Моментов калибровки", "Пропущено на калибровке", "Доступен"], threshold_rows), "</details></section>",
        '<section><h2>Конкретная память, доступная в момент решения</h2>',
        f'<p>Пример выбран до анализа прибыли: первый кандидат проверочного блока после 08:00 UTC. ID <code>{int(chosen.candidate_id)}</code>; решение {esc(stamp(chosen.t_ns))}; зона {chosen.zone_low_nano/1e9:.1f}–{chosen.zone_high_nano/1e9:.1f} USD/oz; расстояние {chosen.distance_r:.2f}R. Ни исход этой будущей сделки, ни будущая траектория в таблицу не включены.</p>',
        table(["Порядок", "Эпизод", "Известен, UTC", "Исход контакта", "Вход ±", "Уход ±", "Внутри, с", "Эпизод, с", "До него отсутствовала, с", "Прямота подхода"], memory_rows),
        f'<p>По этим завершённым эпизодам: возвратов через край <b>{decoded[0]}</b>, прерванных подтверждений ухода <b>{decoded[1]}</b>, максимальная короткая проба <b>{decoded[2]} с</b> и <b>{decoded[3]} USD/oz</b>. Для входа +1 означает прибытие снизу вверх, −1 — сверху вниз. Для ухода +1 означает подтверждённый выход выше зоны, −1 — ниже зоны. В модели положительные масштабы поданы через log1p; здесь они обратно переведены в исходные единицы.</p>',
        '<p>Модель пока получает агрегаты последних событий, а не последовательный event encoder. Содержательная память и её статистическое резюме — разные уровни представления; этот опыт проверяет добавочную пользу выбранного резюме.</p></section>',
        '<section><h2>Поддержка MBO и спектра</h2>',
        table(["Блок", "MBO при уходе", "MBO сейчас", "Спектр известен", "Чужая память известна"], support_rows),
        '<p>В обучении используется глобальный MBO-контекст канонических секунд: направления сделок, состав добавлений/удалений, replenishment и глубина книги. Общая активность уже присутствует в baseline. Глобальные потоки не локализуют действия на исторической зоне. Доступные семь дней локальных sidecar остаются отдельной описательной проверкой: обучающего префикса для локальной ветви нет.</p><p>Окно привязано к последнему полностью опубликованному секундному endpoint; лаг сохранён. Missing не заменяется выдуманным нулевым потоком. Неоднозначные удаления и matched-F counters не являются доказательством намерений участников или чистой отменой.</p></section>',
        local_html,
        '<section><h2>Воспроизводимость и границы вывода</h2><p>Перед чтением данных renderer проверяет SHA-256 корневых parquet/JSON, замороженных исходников, дневных payloads и входных источников. Известность признаков и таргетов, исполнение и независимые проверки отражены в отдельных receipts. Renderer не обучает модели и не выбирает победителя.</p>',
        table(["Ветка", "Исходных признаков", "После missing indicators", "Строк train", "Итерации", "Последний зрелый label train"], train_rows),
        '<p>Незавершённые наблюдения не превращаются в проигрыши. Геометрический hit не превращается в исполнимую прибыль. Уменьшение ошибки не означает принятие торговой политики. Результат на ранее просмотренных датах требует новой записи, если появится устойчивый кандидат.</p>',
        '<p><a href="manifest.json">Manifest</a> · <a href="models.json">Коэффициенты и preprocessing</a> · <a href="thresholds.json">Пороги</a> · <a href="renderer_receipt.json">Проверка renderer</a> · <a href="zone_conditional_audit.png">Рисунок PNG</a></p></section>',
        f'<footer>Источник: {esc(out)}<br>Manifest SHA-256: {esc(inputs["manifest.json"])}<br>Источник предыдущего опыта: {esc(cfg["source_run"])}. Время полного опыта: {manifest["elapsed_seconds"]:.1f} с.</footer></main></html>']
    report = out / "report.html"
    report.write_text("".join(fragments), encoding="utf-8", newline="\n")
    receipt = dict(status="RENDERED_PENDING_VISUAL_REVIEW", scope="No fitting, selection or source mutation",
                   implementation_sha256=digest(__file__), source_manifest_sha256=inputs["manifest.json"],
                   verified_inputs=inputs, outputs={p.name:digest(p) for p in [report,png]},
                   selected_example=dict(candidate_id=int(chosen.candidate_id),decision_ns=int(chosen.t_ns),rule="first evaluation candidate after08UTC; no target or PnL selection"),
                   memory_incremental_diagnostic=memory_ok,economic_block_gates_passed=economic_count,
                   untouched_holdout=False,visual_review="PENDING")
    (out / "renderer_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"report":str(report),"figure":str(png),"verified_inputs":len(inputs),"visual_review":"PENDING"},ensure_ascii=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    render(parser.parse_args().directory)
