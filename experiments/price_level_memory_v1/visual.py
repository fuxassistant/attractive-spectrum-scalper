"""Build the compact inline fragment and a static scientific audit figure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

LEVEL_KEYS = ("price", "mid_dwell_s", "arrivals", "present_at_restart", "pass_up", "pass_down", "reject_up", "reject_down",
              "jump_over", "censored", "active", "observed_age_s", "age_interval_known", "trade_buy_qty", "trade_sell_qty",
              "trade_unknown_qty", "bid_passive_fill", "bid_unlinked_remove", "bid_refill", "ask_passive_fill",
              "ask_unlinked_remove", "ask_refill")
AVERAGE_KEYS = ("window_seconds", "quote_known", "trade_known", "known", "twa_mid", "twa_mid_sigma_population",
                "vwap_trade", "vwap_trade_sigma_population", "equal_trade_price_mean", "trade_size_weighting_gap",
                "trade_price_basis_gap", "trade_timing_gap")


def clean(value):
    if isinstance(value, float):
        return round(value, 5) if math.isfinite(value) else None
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def compact(data):
    result = dict(date=data["date"], level_keys=list(LEVEL_KEYS), scenes=[])
    for key in ("heat_bin_seconds", "price_bin_width"):
        if key in data:
            result[key] = data[key]
    for source in data["scenes"]:
        scene = {key: source[key] for key in ("id", "title", "asof_utc", "mid", "path", "heat", "coverage_fraction")}
        for key in ("heat_bin_seconds", "price_bin_width"):
            if key in source:
                scene[key] = source[key]
            if not scene.get(key, result.get(key)):
                raise ValueError(f"Explicit {key} is required; heat coordinates are lower/left edges")
        scene["levels"] = [[level.get(k, level.get("absence_verified") if k == "age_interval_known" else None)
                            for k in LEVEL_KEYS] for level in source["levels"]]
        for key in ("history_heat", "history_path", "history_bin_seconds", "history_start_utc", "current_quote_known"):
            if key in source:
                scene[key] = source[key]
        scene["averages"] = [{key: row.get(key) for key in AVERAGE_KEYS} for row in source["averages"]]
        result["scenes"].append(scene)
    return clean(result)


def plot_audit(data, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from datetime import datetime, timezone
    import numpy as np
    preferred = ("chop", "sharp_down", "horizon_contrast", "old_zone_return")
    by_id = {scene["id"]: scene for scene in data["scenes"]}
    scenes = [by_id[key] for key in preferred if key in by_id]
    scenes += [scene for scene in data["scenes"] if scene not in scenes][:4-len(scenes)]
    ncols = min(2, len(scenes)); nrows = math.ceil(len(scenes)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8*nrows), squeeze=False)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for ax, scene in zip(axes.flat, scenes):
        if scene["id"] == "old_zone_return" and scene.get("history_heat"):
            scene = dict(scene, heat=scene["history_heat"], path=scene["history_path"],
                         heat_bin_seconds=scene["history_bin_seconds"], title=scene["title"]+" · история с 00:00")
        step = scene.get("heat_bin_seconds", data.get("heat_bin_seconds"))
        bin_width = scene.get("price_bin_width", data.get("price_bin_width"))
        asof = datetime.fromisoformat(scene["asof_utc"]).timestamp()
        last_clock = min(asof, max(row[0] for row in scene["path"]))
        patches, weights = [], []
        for sec, price, dwell, quantity in scene["heat"]:
            if sec >= last_clock or dwell <= 0:
                continue
            x = mdates.date2num(datetime.fromtimestamp(sec, timezone.utc))
            patches.append(Rectangle((x, price), (min(sec+step, last_clock)-sec)/86400, bin_width))
            weights.append(min(1, max(0, dwell/step)))
        collection = PatchCollection(patches, cmap="Blues", edgecolors="none", alpha=.62)
        collection.set_array(np.asarray(weights));collection.set_clim(0,1);ax.add_collection(collection)
        path = scene["path"]
        clocks = [datetime.fromtimestamp(row[0], timezone.utc) for row in path]
        def values(i):return [np.nan if row[i] is None else row[i] for row in path]
        ax.plot(clocks, values(1), color="#243343", lw=1.4)
        ax.plot(clocks, values(2), color="#8b54a2", lw=1, linestyle="--")
        ax.plot(clocks, values(3), color="#8b54a2", lw=.7, linestyle="--",alpha=.7)
        ax.plot(clocks, values(4), color="#8b54a2", lw=.7, linestyle="--",alpha=.7)
        ax.plot(clocks, values(5), color="#158e7b", lw=1.1)
        ax.plot(clocks, values(6), color="#158e7b", lw=.7,alpha=.7)
        ax.plot(clocks, values(7), color="#158e7b", lw=.7,alpha=.7)
        if scene["id"] == "old_zone_return":
            level = math.floor(scene["mid"]/bin_width)*bin_width + bin_width/2
            ax.axhline(level,color="#b57628",linestyle=":",lw=1.1)
            ax.scatter(clocks[-1],scene["mid"],color="#b57628",s=24,zorder=5)
        ax.autoscale_view();ax.set_xlim(right=datetime.fromtimestamp(last_clock, timezone.utc));ax.set_title(f"{scene['title']}\n{scene['asof_utc']} · mid {scene['mid']:.2f}",loc="left",fontsize=11)
        ax.set_xlabel("UTC");ax.set_ylabel("Цена · USD/oz");ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",tz=timezone.utc))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5));ax.grid(alpha=.13)
        ax.tick_params(labelsize=9);ax.spines[["top","right"]].set_visible(False)
    for ax in list(axes.flat)[len(scenes):]:ax.set_visible(False)
    fig.suptitle("Память ценовых уровней: только опубликованное прошлое каждого среза",x=.06,ha="left",fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color="#243343",label="Mid · M1"),Line2D([0],[0],color="#8b54a2",linestyle="--",label="BB20 close ±2σ, ddof=1"),Line2D([0],[0],color="#158e7b",label="VWAP20 exact T ±2σ, population")],loc="lower center",ncol=3,frameon=False)
    fig.text(.06,.055,"Синяя заливка: доля времени native midpoint в ячейке. Цена уровня абсолютная; будущие посещения не включены.",fontsize=9)
    fig.subplots_adjust(left=.06,right=.98,top=.90,bottom=.12,hspace=.38,wspace=.18)
    path = destination / "price_level_memory_audit.png"
    fig.savefig(path,dpi=145);plt.close(fig)
    return path


def render(data_path, output):
    data_path, output = Path(data_path).resolve(), Path(output).resolve()
    source = json.loads(data_path.read_text(encoding="utf-8"))
    data = compact(source)
    template = Path(__file__).with_name("visual_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    fragment = template.replace("__DATA_JSON__", payload)
    if len(fragment.encode("utf-8")) >= 1_000_000:
        raise ValueError("Inline visual must remain below 1 MB; reduce source heat resolution")
    if any(token in fragment.lower() for token in ("<!doctype", "<html", "<head>", "<body")):
        raise ValueError("Expected only an inline fragment")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8", newline="\n")
    if output.read_text(encoding="utf-8") != fragment:
        raise ValueError("Written fragment differs")
    figure = plot_audit(source, data_path.parent)
    review = dict(status="RENDERED", source_data=str(data_path), source_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        fragment=str(output), fragment_bytes=output.stat().st_size, figure=str(figure), scenes=len(source["scenes"]),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(Path(__file__).with_name("visual_template.html").read_bytes()).hexdigest(),
        fragment_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
        prices_are_absolute=True, zero_and_unknown_distinct=True, economic_admission="NOT_ESTABLISHED",
        renderer_python=sys.executable, visual_inspection="PENDING", browser_interaction="PENDING")
    (data_path.parent/"render_review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data");parser.add_argument("output")
    args=parser.parse_args();print(json.dumps(render(args.data,args.output),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
