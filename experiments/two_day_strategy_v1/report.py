"""Render the two-day expert experiment without recomputing its evidence.

All monetary figures are account replay results in USD. Pair/day diagnostics
restart an account; continuous policy curves never concatenate those diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POLICIES = {
    "adaptive": "Выбор по двум дням",
    "frozen_first_pair": "Первый выбор без изменений",
    "frozen_first_raw_pair": "Первый сырой лидер без изменений",
    "raw_leader": "Лидер пары без фильтра устойчивости",
    "always_toward": "Все входы к зоне",
    "always_away": "Все входы от зоны",
    "reverse_adaptive": "Обратная сторона выбранных входов",
    "reverse_raw_leader": "Обратная сторона входов сырого лидера",
    "WAIT": "WAIT",
}
SHORT = {
    "toward_all": "Все к зоне", "away_all": "Все от зоны",
    "memory_return": "Память удержания", "permeable_departure": "Проходимость",
    "failed_escape": "Отмена ухода", "place_migration": "Миграция",
    "flow_away": "Поток от зоны", "flow_return": "Поток к зоне",
    "stalled_flow_return": "Поток без продвижения",
    "aligned_return": "Согласие масштабов", "WAIT": "WAIT",
}
COLORS = {
    "adaptive": "#087d79", "frozen_first_pair": "#8c6b3f", "raw_leader": "#d06441",
    "frozen_first_raw_pair": "#ad8844", "always_toward": "#4772aa", "always_away": "#925ba0",
    "reverse_adaptive": "#7d8891", "reverse_raw_leader": "#707f56", "WAIT": "#333d44",
}


def esc(value):
    return html.escape(str(value))


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (ValueError, TypeError):
        return False


def money(value, complete=True):
    if not bool(complete):
        return "не завершено"
    if not finite(value):
        return "нет данных"
    v = float(value)
    return "0 $" if abs(v) < .5 else f"{v:+,.0f} $".replace(",", " ")


def number(value, digits=2):
    return f"{float(value):.{digits}f}" if finite(value) else "—"


def day(value):
    return pd.Timestamp(value).strftime("%d.%m") if value and str(value) != "nan" else "—"


def title(catalog, strategy):
    return catalog.get(str(strategy), {}).get("title", str(strategy))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "axes.labelcolor": "#32424d", "text.color": "#24333d",
        "axes.edgecolor": "#aeb9c0", "grid.color": "#d9e0e4",
        "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    })


def save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap(daily, dates, catalog, out, condition):
    strategies = list(catalog)
    if not strategies:
        strategies = list(daily.strategy.drop_duplicates())
    matrix = np.full((len(strategies), len(dates)), np.nan)
    counts = np.zeros(matrix.shape, dtype=int)
    incomplete = np.zeros(matrix.shape, dtype=bool)
    z = daily[daily.condition.eq(condition)]
    for i, strategy in enumerate(strategies):
        for j, date in enumerate(dates):
            row = z[z.strategy.eq(strategy) & z.date.astype(str).eq(str(date))]
            if row.empty:
                continue
            row = row.iloc[0]
            counts[i, j] = int(row.trades)
            incomplete[i, j] = not bool(row.capital_curve_complete)
            if not incomplete[i, j]:
                matrix[i, j] = float(row.net_usd)
    extent = max(1., float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1.
    fig, ax = plt.subplots(figsize=(13.6, max(6.2, len(strategies) * .56)), layout="constrained")
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e8ecef")
    mesh = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent), aspect="auto")
    for i in range(len(strategies)):
        for j in range(len(dates)):
            if incomplete[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#abb5bd", lw=0))
                label = "не заверш."
            elif np.isfinite(matrix[i, j]):
                label = f"{matrix[i, j]:+,.0f}".replace(",", " ")
            else:
                label = "нет данных"
            color = "white" if finite(matrix[i, j]) and abs(matrix[i, j]) > extent * .60 else "#24333d"
            ax.text(j, i, label + f"\nn={counts[i, j]}", ha="center", va="center", fontsize=8.6, color=color)
    ax.set_xticks(range(len(dates)), [day(d) for d in dates])
    ax.set_yticks(range(len(strategies)), [SHORT.get(s, s) for s in strategies])
    ax.set_xticks(np.arange(-.5, len(dates), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(strategies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    label = "Базовое исполнение" if condition == "base" else "Стресс исполнения"
    ax.set_title(f"{label}: результат каждой стратегии по дням\nUSD после стоимости · n — исполненные сделки · отдельный счёт на каждый день", fontsize=12)
    fig.colorbar(mesh, ax=ax, label="USD", fraction=.025, pad=.02)
    save(fig, out, f"heatmap_{condition}.png")


def forward_plot(selections, out, mobile=False, raw=False):
    z = selections[selections.next_date.fillna("").ne("")].copy()
    fig, ax = plt.subplots(figsize=(5.5, max(6.8, len(z) * .95)) if mobile else (12, max(5.2, len(z) * .62)), layout="constrained")
    y = np.arange(len(z))
    prefix = "raw_next_" if raw else "next_"
    base = np.array([float(getattr(r, prefix + "base_net_usd")) if bool(getattr(r, prefix + "base_complete")) else np.nan for r in z.itertuples()])
    stress = np.array([float(getattr(r, prefix + "stress_net_usd")) if bool(getattr(r, prefix + "stress_complete")) else np.nan for r in z.itertuples()])
    ax.barh(y - .15, base, height=.27, color="#087d79", label="База")
    ax.barh(y + .15, stress, height=.27, color="#c47b45", label="Стресс")
    labels = [f"{day(r.next_date)} · {SHORT.get(r.raw_leader if raw else r.selected_strategy, r.raw_leader if raw else r.selected_strategy)}" for r in z.itertuples()]
    if mobile:
        labels = ["\n".join(textwrap.wrap(t, 20)) for t in labels]
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="#55656e", lw=.8)
    ax.grid(axis="x", alpha=.5)
    ax.set_xlabel("Следующий день · USD после стоимости")
    heading = "Перенос сырого лидера пары" if raw else "Перенос допустимого выбора"
    ax.set_title(heading + "\nПравило заморожено по предыдущей паре", fontsize=11 if mobile else 13)
    ax.legend(frameon=False, loc="best", fontsize=10)
    for i, r in enumerate(z.itertuples()):
        if not getattr(r, prefix + "base_complete") or not getattr(r, prefix + "stress_complete"):
            ax.text(.02, i, "не завершено", transform=ax.get_yaxis_transform(), fontsize=9)
    stem = "forward_raw" if raw else "forward"
    save(fig, out, stem + ("_mobile" if mobile else "") + ".png")


def equity(trades, summary, out, mobile=False, compact=False):
    dates = summary["dates"][2:]
    start = pd.Timestamp(dates[0], tz="UTC")
    stop = pd.Timestamp(dates[-1], tz="UTC") + pd.Timedelta(days=1)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 8.5) if mobile else (12, 8), layout="constrained")
    initial = float(summary.get("account", {}).get("initial_capital_usd", 100000.))
    policies = ["adaptive", "frozen_first_raw_pair", "raw_leader"] if compact else list(POLICIES)
    results = pd.DataFrame(summary.get("results", []))
    for ax, condition in zip(axes, ("base", "stress")):
        for policy in policies:
            z = trades[trades.policy.eq(policy) & trades.condition.eq(condition)].sort_values("exit_ns")
            stats = results[results.policy.eq(policy) & results.condition.eq(condition)] if len(results) else pd.DataFrame()
            if stats.empty:
                continue
            complete = bool(stats.iloc[0].capital_curve_complete) if len(stats) else True
            x = [start]
            pnl = [0.]
            if len(z):
                good = z.exit_ns.gt(0)
                x.extend(pd.to_datetime(z.loc[good, "exit_ns"].astype("int64"), utc=True).tolist())
                pnl.extend((z.loc[good, "equity_usd"].astype(float) - initial).tolist())
            # Extending an incomplete line would falsely imply a known terminal balance.
            if complete:
                x.append(stop)
                pnl.append(pnl[-1])
            label = POLICIES.get(policy, policy) + (" · неполно" if not complete else "")
            ax.step(x, pnl, where="post", color=COLORS.get(policy, "#697b85"),
                    lw=2.6 if policy == "adaptive" else 1.4, alpha=1 if policy == "adaptive" else .85,
                    linestyle="--" if policy in ("frozen_first_pair", "frozen_first_raw_pair", "reverse_adaptive") else "-", label=label)
            if not complete:
                ax.scatter([x[-1]], [pnl[-1]], marker="x", color=COLORS.get(policy), s=45)
        ticks = [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12) for d in dates]
        ax.set_xticks(ticks[::2] if mobile else ticks, [day(d) for d in (dates[::2] if mobile else dates)], fontsize=9)
        ax.set_xlim(start, stop)
        ax.axhline(0, color="#55656e", lw=.8)
        ax.grid(alpha=.45)
        ax.set_ylabel("Накопленный результат · USD")
        ax.set_title("База" if condition == "base" else "Стресс: задержка и проскальзывание", fontsize=11 if mobile else 13)
    axes[0].legend(loc="best", frameon=False, fontsize=8 if mobile else 9, ncol=1 if mobile else 2)
    fig.suptitle("Один непрерывный счёт на восемь дней\nКапитал без пополнений; одна позиция одновременно", fontsize=12 if mobile else 15)
    stem = "overview" if compact else "equity"
    save(fig, out, stem + ("_mobile" if mobile else "") + ".png")


def picture(stem, alt, mobile=True, scroll=False):
    source = f'<source media="(max-width:640px)" srcset="{stem}_mobile.png">' if mobile else ""
    wrapper = "figure scroll-figure" if scroll else "figure"
    return f'<figure class="{wrapper}"><a href="{stem}.png" target="_blank"><picture>{source}<img src="{stem}.png" alt="{esc(alt)}" loading="lazy"></picture></a></figure>'


def account_table(summary):
    rows = []
    for row in summary.get("results", []):
        complete = bool(row.get("capital_curve_complete", False))
        rows.append(f'<tr><th>{esc(POLICIES.get(row["policy"], row["policy"]))}</th>'
                    f'<td>{"база" if row["condition"] == "base" else "стресс"}</td>'
                    f'<td>{money(row.get("net_usd"), complete)}</td><td>{int(row.get("trades", 0))}</td>'
                    f'<td>{number(row.get("max_drawdown_usd"), 0)} $</td>'
                    f'<td>{"завершён" if complete else "НЕ ЗАВЕРШЁН"}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Политика</th><th>Исполнение</th><th>Результат</th><th>Сделки</th><th>Макс. просадка</th><th>Счёт</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'


def profile_table(profiles, row, reference="adaptive"):
    z = profiles[profiles.pair_start.eq(row["pair_start"]) & profiles.pair_end.eq(row["pair_end"]) & profiles.period.eq("pair")]
    if "reference_policy" in z.columns:
        z = z[z.reference_policy.eq(reference)]
    lines = []
    for r in z.itertuples():
        count = getattr(r, "nonmissing_n", getattr(r, "known_n", 0))
        lines.append(f'<tr><th>{esc(r.title)}</th><td>{number(r.all_median)}</td><td>{number(r.selected_median)}</td><td>{int(count)} / {int(r.selected_n)}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Наблюдение до входа</th><th>Все возможности</th><th>Выбранные</th><th>Непустых значений</th></tr></thead><tbody>' + "".join(lines) + '</tbody></table></div>'


def expert_table(scores, row, catalog):
    z = scores[scores.pair_start.eq(row["pair_start"]) & scores.pair_end.eq(row["pair_end"])].sort_values(["eligible", "score_usd"], ascending=False)
    lines = []
    for r in z.itertuples():
        selected = r.strategy == row["selected_strategy"]
        lines.append(f'<tr class="{"chosen" if selected else ""}"><th>{esc(title(catalog, r.strategy))}{" ✓" if selected else ""}</th>'
                     f'<td>{money(r.base_net_usd, r.base_complete)}</td><td>{money(r.stress_net_usd, r.stress_complete)}</td>'
                     f'<td>{money(r.stress_day1_net_usd, r.stress_complete)} / {money(r.stress_day2_net_usd, r.stress_complete)}</td>'
                     f'<td>{int(r.stress_day1_trades)} / {int(r.stress_day2_trades)}</td><td>{"да" if r.eligible else "нет"}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Стратегия</th><th>База, пара</th><th>Стресс, пара</th><th>Стресс: день 1 / 2</th><th>Сделки: день 1 / 2</th><th>Прошла отбор</th></tr></thead><tbody>' + "".join(lines) + '</tbody></table></div>'


def pair_card(row, scores, profiles, catalog, index):
    selected = row["selected_strategy"]
    key = scores.pair_start.eq(row["pair_start"]) & scores.pair_end.eq(row["pair_end"]) & scores.strategy.eq(selected)
    chosen = scores.loc[key].iloc[0] if key.any() else None
    next_date = row.get("next_date")
    future = bool(next_date) and str(next_date) != "nan"
    name = title(catalog, selected)
    stats = [
        ("Пара · база", money(row.get("selected_pair_base_net_usd"))),
        ("Пара · стресс", money(row.get("selected_pair_stress_net_usd"))),
        ("Предложено входов", str(int(row.get("selected_pair_offered", 0)))),
        ("Прошли условия отбора", str(int(row.get("eligible_count", 0)))),
    ]
    narrative = row.get("interpretation", catalog.get(selected, {}).get("meaning", ""))
    if selected == "WAIT":
        limit = "WAIT означает недостаток двухдневной опоры по нашему правилу. Это не утверждение об отсутствии возможностей вообще."
    else:
        limit = "Выбор описывает вознаграждение внутри этой выборки и правил исполнения. Название стратегии само по себе не устанавливает причину движения или общий режим рынка."
    support = ""
    if chosen is not None and selected != "WAIT":
        support = (f'<p>В стрессе на первом дне: <b>{money(chosen.stress_day1_net_usd)}</b> '
                   f'({int(chosen.stress_day1_trades)} сделок), на втором: <b>{money(chosen.stress_day2_net_usd)}</b> '
                   f'({int(chosen.stress_day2_trades)} сделок). Результат пары без лучшей сделки: '
                   f'<b>{money(chosen.stress_without_best_trade_usd)}</b>. Это проверка концентрации результата, не новый счёт с переигранными ограничениями.</p>')
    next_block = ('<p class="muted">Для последней пары следующий день находится за пределами исследования. Перенос не проверен.</p>')
    if future:
        next_block = (f'<div class="next-day"><h4>Проверка {day(next_date)} · выбор уже заморожен</h4>'
                      f'<p>База: <b>{money(row.get("next_base_net_usd"), row.get("next_base_complete", False))}</b>, '
                      f'стресс: <b>{money(row.get("next_stress_net_usd"), row.get("next_stress_complete", False))}</b>. '
                      f'Предложено входов: {int(row.get("next_offered", 0))}; исполнено база / стресс: '
                      f'{int(row.get("next_base_trades", 0))} / {int(row.get("next_stress_trades", 0))}.</p>'
                      '<p class="small">Это отдельный диагностический счёт следующего дня. Итог непрерывной политики находится на графике капитала.</p></div>')
    raw_block = ""
    if "raw_pair_base_net_usd" in row:
        raw = row.get("raw_leader", "WAIT")
        raw_row = scores.loc[scores.pair_start.eq(row["pair_start"]) & scores.pair_end.eq(row["pair_end"]) & scores.strategy.eq(raw)]
        raw_complete = bool(raw_row.iloc[0].base_complete) if len(raw_row) else True
        raw_stress_complete = bool(raw_row.iloc[0].stress_complete) if len(raw_row) else True
        raw_net = float(row.get("raw_pair_base_net_usd", 0))
        raw_stress_net = float(row.get("raw_pair_stress_net_usd", 0))
        if raw_net <= 0:
            raw_reading = "Лидер сравнения оказался убыточным даже в базовом исполнении. Его первое место означает меньший убыток среди сравниваемых вариантов, а не вознаграждение за этот способ входа."
        elif raw_stress_net <= 0:
            raw_reading = "В базовом исполнении пара дала положительный итог, но стресс его не сохранил. Такая победа пока не даёт устойчивой опоры для следующего дня."
        else:
            raw_reading = "Общий итог пары положителен в обоих условиях. Для переноса дополнительно проверяются поддержка и положительный результат каждого из двух дней."
        raw_days = ""
        if len(raw_row):
            r = raw_row.iloc[0]
            raw_days = '<div class="table-wrap"><table><thead><tr><th>День опоры</th><th>База</th><th>Стресс</th><th>Сделки база / стресс</th></tr></thead><tbody>'
            for i, d in enumerate((row["pair_start"], row["pair_end"]), 1):
                raw_days += (f'<tr><th>{day(d)}</th><td>{money(r[f"base_day{i}_net_usd"], r.base_complete)}</td>'
                             f'<td>{money(r[f"stress_day{i}_net_usd"], r.stress_complete)}</td>'
                             f'<td>{int(r[f"base_day{i}_trades"])} / {int(r[f"stress_day{i}_trades"])}</td></tr>')
            raw_days += '</tbody></table></div>'
        raw_next = '<p class="muted">Следующего дня в этой выборке нет; перенос лидера ещё не проверен.</p>'
        if future:
            raw_next = (f'<p><b>Следующий день {day(next_date)}:</b> база {money(row.get("raw_next_base_net_usd"), row.get("raw_next_base_complete", False))}; '
                        f'стресс {money(row.get("raw_next_stress_net_usd"), row.get("raw_next_stress_complete", False))}. '
                        f'Исполнено {int(row.get("raw_next_base_trades", 0))} / {int(row.get("raw_next_stress_trades", 0))} сделок.</p>')
        raw_block = f'''<div class="raw-diagnostic"><div class="eyebrow">Лидер сравнения · без фильтра устойчивости</div>
          <h3>{esc(title(catalog, raw))}</h3><p>{esc(raw_reading)}</p>
          <p><b>Какие входы это означает:</b> {esc(catalog.get(raw, {}).get('rule', ''))}</p>
          <p>Пара: база <b>{money(row.get('raw_pair_base_net_usd'), raw_complete)}</b>, стресс <b>{money(row.get('raw_pair_stress_net_usd'), raw_stress_complete)}</b>.
          Предложено входов: {int(row.get('raw_pair_offered', 0))}.</p>{raw_days}{raw_next}
          <details><summary>Какие состояния выбирал этот лидер</summary><p class="small">Профиль описывает отбор по прошлой паре. Признаки, входящие в правило, не служат независимым объяснением результата.</p>{profile_table(profiles, row, 'raw_leader')}</details></div>'''
    metrics_html = '<div class="metrics">' + ''.join(f'<div><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in stats) + '</div>' if selected != 'WAIT' else ''
    competition = (f'<p>Отрыв по критерию отбора: <b>{money(row.get("score_gap_usd"))}</b>; ближайший допустимый конкурент: '
                   f'<b>{esc(title(catalog, row.get("runner_up", "WAIT")))}</b>. При единственном допустимом варианте отрыв считается от нулевого WAIT.</p>'
                   f'<p>Обратная сторона тех же предложенных входов за пару: база <b>{money(row.get("reverse_pair_base_net_usd"), row.get("reverse_pair_base_complete", True))}</b>, '
                   f'стресс <b>{money(row.get("reverse_pair_stress_net_usd"), row.get("reverse_pair_stress_complete", True))}</b>.</p>') if selected != 'WAIT' else ''
    adaptive_profile = (f'<details><summary>Что было видно в выбранных состояниях рынка</summary><p class="small">Медианы до входа. Признаки правила не являются независимым подтверждением объяснения.</p>{profile_table(profiles, row)}</details>') if selected != 'WAIT' else ''
    if selected == 'WAIT' and future:
        next_block = f'<p>На {day(next_date)} допустимая политика сохранила нулевую позицию и результат 0 USD.</p>'
    return f'''<article class="pair" id="pair-{index}">
      <div class="eyebrow">Пара {index + 1} · {day(row['pair_start'])}–{day(row['pair_end'])}</div>
      {raw_block}<h3>Допустимый выбор: {esc(name)}</h3><p class="meaning">{esc(narrative)}</p>
      {metrics_html}{support}{competition}
      {next_block}<p class="caution">{esc(limit)}</p>
      <details><summary>Сравнить все стратегии этой пары</summary>{expert_table(scores, row, catalog)}</details>
      {adaptive_profile}</article>'''


def random_table(summary):
    names = {"coin": "Случайная сторона на выбранных входах", "coin_direction": "Случайная сторона на выбранных входах", "matched_activity": "Случайный отбор с прошлыми частотами", "random_acceptance": "Случайный отбор с прошлыми частотами", "matched_random": "Случайный отбор с прошлыми частотами"}
    rows = []
    for r in summary.get("random_controls", []):
        reference = "Сырой лидер" if r.get("reference_policy") == "raw_leader" else "Допустимый выбор"
        rows.append(f'<tr><th>{esc(names.get(r["kind"], r["kind"]))}<br><span class="small">{reference}</span></th><td>{esc(r["condition"])}</td>'
                    f'<td>{money(r.get("median_usd"))}</td><td>{money(r.get("p05_usd"))} … {money(r.get("p95_usd"))}</td>'
                    f'<td>{int(r.get("complete_runs", 0))} / {int(r.get("n", 0))}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Контроль</th><th>Исполнение</th><th>Медиана</th><th>5–95% по seed</th><th>Завершено</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'


def scenes_section(out):
    path = out / "scenes.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = []
    for scene in payload.get("scenes", []):
        def scene_picture(key, alt):
            image = str(scene[key])
            mobile = str(scene.get(key.replace("_image", "_mobile_image"), ""))
            if not mobile:
                candidate = str(Path(image).with_name(Path(image).stem + "_mobile.png")).replace("\\", "/")
                if (out / candidate).exists():
                    mobile = candidate
            source = f'<source media="(max-width:640px)" srcset="{esc(mobile)}">' if mobile else ""
            cls = "figure" if mobile else "figure scroll-figure"
            return f'<figure class="{cls}"><a href="{esc(image)}" target="_blank"><picture>{source}<img src="{esc(image)}" alt="{esc(alt)}" loading="lazy"></picture></a></figure>'
        features = scene.get("features", {})
        if isinstance(features, dict):
            facts = "".join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in features.items())
        else:
            facts = f'<tr><td>{esc(features)}</td></tr>'
        cards.append(f'''<article class="pair" id="scene-{esc(scene['id'])}">
          <div class="eyebrow">Пример исполнения сырого лидера · выбран по времени</div>
          <h3>{esc(scene.get('strategy_label', scene['strategy']))}</h3>
          <p>Решение: <b>{esc(scene.get('decision_time_utc', scene.get('date', '')))}</b>.
          Стратегия выбрана по паре {day(scene['pair_start'])}–{day(scene['pair_end'])}.</p>
          {scene_picture('before_image', 'Доступные до решения цена, область и условия выбранной стратегии; будущее скрыто')}
          <details><summary>Числа, доступные до входа</summary><div class="table-wrap"><table><tbody>{facts}</tbody></table></div></details>
          <details class="scene-reveal"><summary>Раскрыть исполнение и следующий путь</summary>
          <p>Результат одной исполненной сделки: <b>{money(scene.get('net_usd'))}</b>. Причина выхода: {esc(scene.get('reason', ''))}.
          Вход {esc(scene.get('entry_time_utc', ''))}; выход {esc(scene.get('exit_time_utc', ''))}.</p>
          {scene_picture('after_image', 'Реальный последующий путь, рыночный вход и выход с задержкой')}
          <p class="small">Это наглядный эпизод, выбранный без просмотра исхода. Частоты успеха оцениваются по полным таблицам, а не по этим примерам.</p></details></article>''')
    return '<section id="scenes"><h2>Как выглядело выбранное открытие</h2><p>Для каждого встретившегося типа сырого лидера показан первый исполненный вход по времени. Выбор иллюстраций не зависит от прибыли сделки. Допустимая политика в этом запуске могла оставаться WAIT; эти сделки относятся к диагностике принудительного следования лидеру.</p>' + "".join(cards) + '</section>'


def cost_section(out):
    path = out / "diagnostics.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("cost_attribution", []):
        rows.append(f'<tr><th>{"База" if r["condition"] == "base" else "Стресс"}</th>'
                    f'<td>{money(r["mid_move_usd"])}</td><td>{money(-r["spread_usd"])}</td>'
                    f'<td>{money(-r["commission_usd"])}</td><td>{money(-r["slippage_usd"])}</td><td>{money(r["net_usd"])}</td></tr>')
    return '''<div class="notice"><h3>Что объясняет убыток сырого лидера</h3>
      <p>На исполненных входах направленная составляющая по средней цене bid/ask почти нулевая. Спред, комиссия и проскальзывание превращают этот путь в существенный убыток.</p>
      <div class="table-wrap"><table><thead><tr><th>Исполнение</th><th>Движение midpoint</th><th>Спред</th><th>Комиссия</th><th>Проскальзывание</th><th>Итого</th></tr></thead><tbody>''' + "".join(rows) + '''</tbody></table></div>
      <p class="small">Разложены те же исполнения, время выходов и остановки счёта. Midpoint не является исполнимой ценой. Это добавленная после первого результата диагностика, а не повторный опыт без издержек: при других издержках границы и выходы могли бы измениться. Результат относится к выбранным правилам и не устанавливает отсутствие направленной информации вообще.</p></div>'''


CSS = """
.raw-diagnostic{background:#fff6e9;border-left:4px solid #b98746;padding:20px;margin:22px 0 28px}.raw-diagnostic h3{font-size:23px}.raw-diagnostic .eyebrow{color:#8d6533}
:root{color-scheme:light;--ink:#22343e;--muted:#5a6b76;--line:#d7e0e5;--teal:#087d79}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f6f7;color:var(--ink);font:17px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:48px 28px 88px}h1,h2,h3,h4{line-height:1.2;letter-spacing:-.025em}h1{font-size:clamp(31px,5vw,55px);max-width:950px;margin:12px 0 24px}h2{font-size:29px;margin:52px 0 20px}h3{font-size:25px;margin:12px 0 18px}h4{font-size:20px;margin:0 0 12px}p{max-width:90ch}.lead{font-size:21px;color:#455c68}.eyebrow{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:750}.muted,.small{color:var(--muted)}.small{font-size:14px}a{color:#086f84;text-underline-offset:.2em}nav{display:flex;gap:10px;flex-wrap:wrap;margin:28px 0}nav a{padding:8px 13px;border:1px solid var(--line);background:white;border-radius:5px;text-decoration:none}.notice{padding:20px 24px;background:#e2eeed;border-left:4px solid var(--teal);margin:28px 0}.notice p{margin:5px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.metrics>div{background:#f1f5f6;padding:15px;border-radius:6px;min-width:0}.metrics span{display:block;font-size:13px;color:var(--muted)}.metrics b{display:block;font-size:22px;line-height:1.4;margin-top:6px}.pair{background:#fff;border:1px solid var(--line);padding:28px;margin:22px 0;border-radius:10px;scroll-margin-top:20px}.meaning{font-size:19px}.next-day{background:#eef5f4;border-left:4px solid var(--teal);padding:20px;margin:22px 0}.next-day p:last-child{margin-bottom:0}.caution{font-size:15px;color:#6c5848}.figure{margin:26px 0;padding:8px;background:white;border:1px solid var(--line);border-radius:6px}.figure img{display:block;width:100%;height:auto}.table-wrap{overflow-x:auto;margin:18px 0;border:1px solid var(--line);border-radius:5px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}th:first-child{white-space:normal;min-width:210px}thead{background:#edf2f4}tbody th{font-weight:550}tbody tr:last-child>*{border-bottom:0}.chosen{background:#e6f2ee}details{border-top:1px solid var(--line);padding:16px 0 0;margin-top:16px}summary{font-weight:650;cursor:pointer;min-height:32px}details[open]>summary{margin-bottom:14px}.catalog{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.catalog article{padding:19px;background:white;border:1px solid var(--line);border-radius:6px}.catalog h3{font-size:19px;margin:0 0 10px}.catalog p{margin:0;font-size:15px}.files{display:flex;gap:12px 24px;flex-wrap:wrap}.footer{border-top:1px solid var(--line);padding-top:24px;margin-top:40px}
@media(max-width:640px){body{font-size:16px}main{padding:28px 16px 56px}.lead{font-size:19px}h2{font-size:25px;margin-top:36px}.pair{padding:19px 16px}.metrics{grid-template-columns:repeat(2,1fr);gap:8px}.metrics>div{padding:12px}.metrics b{font-size:20px}.catalog{grid-template-columns:1fr}.notice{padding:15px}.next-day{padding:16px}.scroll-figure{overflow-x:auto}.scroll-figure img{width:1100px;max-width:none}.figure:not(.scroll-figure){margin-left:-6px;margin-right:-6px;padding:2px}nav a{font-size:14px}.table-wrap{margin-left:-4px;margin-right:-4px}th,td{padding:10px}}
"""


def render(summary, scores, selections, daily, profiles, trades, out, config):
    catalog = summary["strategy_catalog"]
    dates = summary["dates"]
    for condition in ("base", "stress"):
        heatmap(daily, dates, catalog, out, condition)
    for mobile in (False, True):
        forward_plot(selections, out, mobile)
        if "raw_next_base_net_usd" in selections.columns:
            forward_plot(selections, out, mobile, raw=True)
        equity(trades, summary, out, mobile)
        equity(trades, summary, out, mobile, compact=True)
    cards = "".join(pair_card(r, scores, profiles, catalog, i) for i, r in enumerate(selections.to_dict("records")))
    results = pd.DataFrame(summary.get("results", []))
    def adaptive(condition):
        r = results[results.policy.eq("adaptive") & results.condition.eq(condition)]
        return money(r.iloc[0].net_usd, r.iloc[0].capital_curve_complete) if len(r) else "нет данных"
    n_forward = int(selections.next_date.fillna("").ne("").sum())
    n_wait = int((selections.selected_strategy.eq("WAIT") & selections.next_date.fillna("").ne("")).sum())
    findings = summary.get("conclusions", [])
    if isinstance(findings, str):
        findings = [findings]
    finding_html = "".join(f"<p>{esc(f)}</p>" for f in findings) if isinstance(findings, list) else ""
    if not finding_html and n_wait == n_forward and n_forward:
        finding_html += f'<p><b>Результат:</b> на всех {n_forward} днях применения допустимая политика выбрала WAIT. Ни одно правило не получило требуемой опоры на обоих предыдущих днях.</p>'
    raw_result = results[results.policy.eq("raw_leader") & results.condition.eq("base")]
    matched = [r for r in summary.get("random_controls", []) if r.get("reference_policy") == "raw_leader" and r.get("kind") == "matched_activity" and r.get("condition") == "base"]
    if len(raw_result) and matched:
        finding_html += (f'<p>Непрерывный результат принудительного следования сырому лидеру: <b>{money(raw_result.iloc[0].net_usd, raw_result.iloc[0].capital_curve_complete)}</b>; '
                         f'медиана случайного отбора с частотами прошлых пар: <b>{money(matched[0]["median_usd"])}</b>. '
                         'Сравнение частот и направления показано ниже; смена названия победителя сама по себе не доказывает смену режима.</p>')
    raw_forward = ""
    if "raw_next_base_net_usd" in selections.columns:
        raw_forward = ('<h3>Как переносился лидер без фильтра устойчивости</h3>' +
                       picture('forward_raw', 'Результат сырого лидера прошлой пары на следующем дне') +
                       '<p>Этот диагностический выбор может быть лучшим среди убыточных стратегий. Его результат показывает, что происходит при принудительном следовании двухдневному лидеру.</p>')
    entries = "".join(f'<article><h3>{esc(v["title"])}</h3><p>{esc(v["rule"])}</p></article>' for v in catalog.values())
    links = "".join(f'<a href="{f}" download>{esc(f)}</a>' for f in ["summary.json", "config.json", "pair_scores.parquet", "selections.parquet", "daily_policy.parquet", "market_profiles.parquet", "trades.parquet", "diagnostics.json", "scenes.json"] if (out / f).exists())
    pair_nav = "".join(f'<a href="#pair-{i}">{day(r.pair_start)}–{day(r.pair_end)}</a>' for i, r in enumerate(selections.itertuples()))
    fee = summary.get("execution", config.get("execution", {})).get("fee_roundtrip_per_oz", .3)
    base_latency = summary.get("execution", config.get("execution", {})).get("latency_ms", 100)
    stress = summary.get("stress", config.get("stress", {}))
    min_trades = config.get("selection", {}).get("min_trades_each_day", 3)
    report = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Какую стратегию рынок поощряет · две недели GCZ6</title><style>{CSS}</style></head><body><main>
    <div class="eyebrow">Исследование двухдневного выбора · GCZ6 · 2026</div>
    <h1>Какую стратегию открытий рынок поощряет?</h1>
    <p class="lead">{day(dates[0])}–{day(dates[-1])}: десять торговых дней, последовательный выбор по предыдущим двум дням и проверка на следующем. Каждое объяснение связано с конкретными входами, стоимостью и опровержением.</p>
    <nav><a href="#result">Непрерывный результат</a><a href="#daily">Карта вознаграждения</a><a href="#pairs">Выбор каждой пары</a><a href="#method">Правила опыта</a><a href="#files">Данные</a></nav>
    <div class="notice"><p><b>Что означает выбор модели.</b> Среди заранее заданных стратегий она выбирает наиболее устойчиво оплачивавшуюся на обоих прошлых днях. Это измерение условного вознаграждения в нашей популяции возможностей; причинное объяснение рынка требует дополнительной проверки.</p>{finding_html}</div>
    <div class="metrics"><div><span>Непрерывная политика · база</span><b>{adaptive('base')}</b></div><div><span>Непрерывная политика · стресс</span><b>{adaptive('stress')}</b></div><div><span>Дней переноса выбора</span><b>{n_forward}</b></div><div><span>Из них выбран WAIT</span><b>{n_wait}</b></div></div>
    <p class="small">Все деньги — USD при одном контракте 100 oz, после моделируемой стоимости. Капитал изначально 100 000 USD. Даты уже исследовались ранее; это разработка метода, не нетронутая проверочная выборка.</p>
    <section id="result"><h2>Переживает ли стратегия смену дня?</h2>
    {picture('overview', 'Непрерывный результат адаптивного выбора и двух контрольных политик')}
    <p>Каждая линия — один последовательный счёт на восьми днях применения. «Первый сырой лидер» сохраняет лидера первой пары. «Лидер без фильтра» обновляет его по общей базовой прибыли пары. WAIT сохраняет нулевую позицию. Незавершённый счёт обрывается и не считается полным результатом.</p>
    {cost_section(out)}
    {raw_forward}
    {picture('forward', 'Результат замороженной стратегии на следующем дне, база и стресс')}
    <p class="small">Столбцы показывают отдельные диагностические счета каждого следующего дня. Они не суммируются для построения непрерывной линии: ограничения капитала и последовательность позиций применяются заново ко всему периоду.</p>
    <details><summary>Все непрерывные политики, просадки и завершённость</summary>{picture('equity', 'Все непрерывные контрольные политики в базовом и стрессовом исполнении')}{account_table(summary)}</details>
    <details><summary>Контроль случайной стороны и случайного отбора</summary>{random_table(summary)}
    <p class="small">Диапазон 5–95% описывает варианты случайного контроля на том же рынке. Это не доверительный интервал будущей доходности. Частоты отбора определены по прошлой паре; будущие количества сделок не подгоняются. Нулевые контроли допустимого WAIT неинформативны: у них также нет открытий. Дополнительные контроли сырого лидера введены после первого результата со сплошным WAIT и служат диагностике; исходное правило допуска не изменено.</p></details></section>
    <section id="daily"><h2>Карта вознаграждения за десять дней</h2>
    <p>В каждой ячейке — результат после стоимости и число исполненных сделок. Дневные счета независимы. После достижения дневного лимита потерь новые входы прекращаются, поэтому многие значения сосредоточены около −2 000 USD. Штриховка означает незавершённое исполнение; такой результат нельзя ранжировать как обычную прибыль.</p>
    <p class="small">На телефоне карту можно прокрутить горизонтально; нажатие откроет отдельное изображение.</p>
    {picture('heatmap_base', 'Матрица базового результата стратегий по десяти дням', mobile=False, scroll=True)}
    <details><summary>Та же карта при более тяжёлом исполнении</summary>{picture('heatmap_stress', 'Матрица стрессового результата стратегий по десяти дням', mobile=False, scroll=True)}</details></section>
    <section id="pairs"><h2>«Модель выбрала эту стратегию. Что это говорит про рынок?»</h2>
    <p>Сначала читаем свидетельство двух дней, затем проверяем перенос. Выбор фиксируется до начала следующего UTC-дня. Последняя пара имеет только ретроспективное описание. Соседние пары перекрываются и не являются независимыми испытаниями.</p><nav>{pair_nav}</nav>{cards}</section>
    {scenes_section(out)}
    <section id="method"><h2>Что именно исследуется</h2>
    <p>Перебирается конечный словарь правил входа. Каждый эксперт выбирает подмножество причинно доступных состояний и сторону: к старой зоне либо от неё. Это первый инструмент выбора объяснимой стратегии; новые правила из данных здесь не генерируются.</p>
    <div class="catalog">{entries}</div>
    <h3>Правило двухдневного выбора</h3><p>Для допуска нужны завершённые базовый и стрессовый счета, минимум {min_trades} исполненные сделки на каждом дне в обоих условиях и строго положительный результат каждого дня. Критерий — меньшая из двух стрессовых дневных прибылей. Побеждает больший критерий; при равенстве порядок задаёт ID стратегии. Если никто не допущен, выбирается WAIT. «Сырой лидер» показывается отдельно и этого фильтра не проходит.</p>
    <h3>Одинаковый договор исполнения</h3><p>Цель к зоне — её замороженная ближняя граница. Цель от зоны зеркальна и имеет ту же исходную валовую дистанцию. Стоп задан замороженным R; срок 300 секунд отсчитывается от входа. Сигнал пересечения вызывает рыночный выход с задержкой, поэтому заполнение ровно на границе не гарантируется. Это отличается от геометрического исхода прежнего атласа.</p>
    <p>База: bid/ask, комиссия за круг {number(fee)} USD/oz и задержка {number(base_latency, 0)} мс. Стресс: задержка {number(stress.get('latency_ms', 250), 0)} мс и {number(stress.get('slippage_ticks_per_fill', 1), 0)} тик проскальзывания на каждое исполнение. Размер — один контракт; одновременные позиции запрещены, действуют ограничения риска счёта. Глубина исполнения, очередь, влияние наших заявок и брокерская маржа не моделируются.</p>
    <h3>Границы интерпретации</h3><p>Популяция требует прежних завершённых посещений и нахождения снаружи зоны. Память места внутридневная. Признаки потока относятся к глобальной книге на текущем маршруте цены, а не к сделкам непосредственно на запомненной области. Неизвестная сторона объёма сохраняется неизвестной. Совпадение потока и продвижения не доказывает намерения участников.</p>
    <p>Порогам атласа и медиане активности предшествует префикс 3–14 августа. Эти десять дней и перекрывающиеся пары позволяют проверить устройство инструмента, устойчивость выбора и ошибки объяснений. Они не устанавливают устойчивое торговое преимущество или готовность к реальному исполнению.</p></section>
    <section id="files" class="footer"><h2>Проверяемые артефакты</h2><div class="files">{links}</div>
    <p class="small">Источник чисел — сохранённые таблицы данного запуска. Графики не пересчитывают сигналы, выбор или исполнение. Все календарные границы и отметки — UTC.</p></section></main></body></html>'''
    (out / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Directory containing summary and parquet results")
    parser.add_argument("--out", dest="out_dir")
    args = parser.parse_args()
    selected = args.out_dir or args.run
    if not selected:
        parser.error("provide a result directory")
    out = Path(selected)
    if not out.is_absolute():
        out = ROOT / out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    config_path = out / "config.json"
    if not config_path.exists():
        config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tables = {name: pd.read_parquet(out / f"{name}.parquet") for name in ("pair_scores", "selections", "daily_policy", "market_profiles", "trades")}
    style()
    render(summary, tables["pair_scores"], tables["selections"], tables["daily_policy"], tables["market_profiles"], tables["trades"], out, config)
    paths = [out / "report.html", *sorted(out.glob("*.png"))]
    receipt = {"status": "RENDERED_AWAITING_VISUAL_REVIEW", "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               "source_summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
               "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "render_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "report_complete", "report": str(out / "report.html"), "images": len(paths) - 1}))


if __name__ == "__main__":
    main()
