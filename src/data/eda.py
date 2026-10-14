from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sentence_lengths(lengths: list[int], max_content_tokens: int, out_path: Path) -> None:
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    upper = max(max(lengths), max_content_tokens)
    ax.hist(
        lengths,
        bins=range(0, upper + 2),
        color="#3B6D8C",
        edgecolor="white",
        linewidth=0.4,
    )
    n_over = sum(1 for length in lengths if length > max_content_tokens)
    ax.axvline(max_content_tokens, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_title("Eğitim kümesi cümle uzunluğu")
    ax.set_xlabel("Temizlenmiş kelime sayısı (<start> ve <end> hariç)")
    ax.set_ylabel("Altyazı sayısı")
    ax.text(
        max_content_tokens - 0.6,
        ax.get_ylim()[1] * 0.97,
        f"{max_content_tokens} kelime sınırı\n({n_over} altyazı üstünde)",
        ha="right",
        va="top",
        color="#C44E52",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_word_frequencies(counts: Counter, min_word_freq: int, out_path: Path) -> None:
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    top = counts.most_common(20)
    labels = [word for word, _ in reversed(top)]
    values = [count for _, count in reversed(top)]
    axes[0].barh(labels, values, color="#3B6D8C")
    axes[0].set_title("En sık 20 kelime")
    axes[0].set_xlabel("Sıklık")

    freqs = sorted(counts.values(), reverse=True)
    kept = sum(1 for count in freqs if count >= min_word_freq)
    axes[1].plot(range(1, len(freqs) + 1), freqs, color="#3B6D8C", linewidth=1.5)
    axes[1].axhline(min_word_freq, color="#C44E52", linestyle="--", linewidth=1.2, label=f"eşik {min_word_freq}")
    axes[1].axvline(kept, color="#6B8F71", linestyle=":", linewidth=1.2, label=f"sözlükteki kelime {kept}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Sıklık sırası")
    axes[1].set_xlabel("Sıra")
    axes[1].set_ylabel("Sıklık")
    axes[1].legend()

    fig.suptitle("Eğitim kümesi kelime sıklığı")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
