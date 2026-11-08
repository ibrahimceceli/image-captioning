"""Flickr8k görüntü altyazılama demosu (Gradio).

    python -m app.app                  # app/config.yaml'daki cihaz (auto)
    python -m app.app --device cpu     # GPU eğitimle paylaşılıyorsa
    python -m app.app --port 7861

Fotoğraf yüklenir; dört modelin altyazısı, çıkarım süresi ve parametre sayısı yan yana
gösterilir; attention destekleyen modelde üretilen her kelimenin haritası çizilir.
Modeller app/captioners.py'de bir kez yüklenir; yollar app/config.yaml'dadır.
"""

from __future__ import annotations

import argparse
import sys

import gradio as gr
import numpy as np

from app.captioners import CONFIG_PATH, MAX_BEAM, CaptionerRegistry, CaptionResult
from src.visualize_attention import visualize_attention

NO_RESULT = "—"
EXAMPLE_COUNT = 6
MAX_UPLOAD = "20mb"  # tek fotoğraf için yeterli; çok büyük dosyalar reddedilir


def example_images() -> list[str]:
    """Doğrulama kümesinden ilk birkaç görüntü (test kümesi demoda kullanılmaz). Veri yoksa boş."""
    try:
        from src.data.config import load_config, repo_root
        from src.data.split import load_split

        config = load_config()
        root = repo_root()
        images = root / config["images_dir"]
        val = load_split(root / config["split_path"])["val"]
        paths = [images / name for name in val[:EXAMPLE_COUNT]]
        return [str(path) for path in paths if path.is_file()]
    except Exception:
        return []


def idle_meta(captioner) -> str:
    if captioner.ready:
        return f"✅ hazır · **{captioner.params:,}** parametre"
    return f"⚠️ {captioner.status}"


def result_meta(result: CaptionResult | str) -> str:
    if isinstance(result, str):
        return f"⚠️ {result}"
    return (
        f"⏱ **{result.ms:.0f} ms** ({result.detail}) · **{result.params:,}** parametre"
    )


def figure_to_array(figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def render_attention(state: dict | None, key: str | None, registry: CaptionerRegistry):
    """-> (açıklama Markdown'ı, harita görüntüsü | None)"""
    if not key:
        return "Attention destekleyen model tanımlı değil.", None
    captioner = registry[key]
    if not state:
        if not captioner.ready:
            return f"**{captioner.label}**: ⚠️ {captioner.status}", None
        return "Önce bir fotoğraf yükleyip **Altyazı üret**'e basın.", None
    result = state["results"].get(key)
    if isinstance(result, str) or result is None:
        return f"**{captioner.label}**: ⚠️ {result or 'sonuç yok'}", None
    if result.attention is None:
        return f"**{captioner.label}**: bu model attention ağırlığı döndürmedi.", None
    columns = 4 if len(result.attention_words) <= 12 else 6
    figure = visualize_attention(state["crop"], result.attention_words, result.attention, columns=columns)
    caption = result.caption or "(boş altyazı)"
    header = (
        f"**{captioner.label}** — *{caption}* · {len(result.attention_words)} kelime · "
        f"{state['method']} · 224×224 kırpım üzerinde, ortak renk ölçeği"
    )
    return header, figure_to_array(figure)


def build_demo(registry: CaptionerRegistry) -> gr.Blocks:
    captioners = registry.captioners
    attention_choices = [(c.label, c.key) for c in captioners if c.attention_capable]
    default_attention = next(
        (c.key for c in captioners if c.attention_capable and c.ready),
        attention_choices[0][1] if attention_choices else None,
    )

    def on_generate(image, method, beam_size, attention_key):
        if image is None:
            gr.Warning("Önce bir fotoğraf yükleyin.")
            return tuple(gr.skip() for _ in range(2 * len(captioners) + 3))
        prepared, results = registry.run_all(image, method, int(beam_size))
        label = "greedy" if method == "greedy" else f"beam {int(beam_size)}"
        state = {"crop": prepared.crop, "results": results, "method": label}
        captions, metas = [], []
        for captioner in captioners:
            result = results[captioner.key]
            captions.append(NO_RESULT if isinstance(result, str) else (result.caption or "(boş altyazı)"))
            metas.append(result_meta(result))
        header, heatmaps = render_attention(state, attention_key, registry)
        return (*captions, *metas, state, header, heatmaps)

    def on_method(method):
        return gr.Slider(interactive=method == "beam")

    # analytics_enabled=False: Gradio'nun anonim kullanım istatistiği göndermesini kapatır.
    with gr.Blocks(title="Flickr8k Altyazılama Demosu", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Flickr8k görüntü altyazılama\n"
            "Bir fotoğraf yükleyin; dört modelin altyazısı yan yana üretilir. "
            "Süreler tek görüntü içindir (önbellek tabanlı modellerde ResNet50 encoder + decoder)."
        )
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                image = gr.Image(type="pil", label="Fotoğraf", sources=["upload", "clipboard"], height=320)
                method = gr.Radio(["greedy", "beam"], value="greedy", label="Çözümleme")
                beam_size = gr.Slider(1, MAX_BEAM, value=3, step=1, label="Beam genişliği", interactive=False)
                run = gr.Button("Altyazı üret", variant="primary")
                examples = example_images()
                if examples:
                    gr.Examples(examples, inputs=image, label="Örnekler (doğrulama kümesi)")
                gr.Markdown(f"<small>{registry.encoder_status}</small>")
            with gr.Column(scale=2):
                caption_boxes, meta_boxes = [], []
                for captioner in captioners:
                    with gr.Group():
                        caption_boxes.append(gr.Textbox(
                            label=captioner.label, value=NO_RESULT, lines=2, interactive=False,
                        ))
                        meta_boxes.append(gr.Markdown(idle_meta(captioner), padding=True))

        gr.Markdown("## Attention haritaları")
        attention_model = gr.Radio(
            attention_choices, value=default_attention, label="Model (attention destekleyenler)",
        )
        initial_header, _ = render_attention(None, default_attention, registry)
        attention_header = gr.Markdown(initial_header)
        attention_image = gr.Image(label="Kelime başına attention", type="numpy", interactive=False)

        method.change(on_method, method, beam_size)
        run.click(
            on_generate,
            inputs=[image, method, beam_size, attention_model],
            outputs=[*caption_boxes, *meta_boxes, state, attention_header, attention_image],
        )
        attention_model.change(
            lambda current, key: render_attention(current, key, registry),
            inputs=[state, attention_model],
            outputs=[attention_header, attention_image],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flickr8k altyazılama Gradio demosu")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--device", help="auto | cpu | cuda (varsayılan: app/config.yaml)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio genel bağlantısı (dışarı açar)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows cp1254 konsolunda karakter hatası olmasın
    registry = CaptionerRegistry(args.config, device=args.device)
    registry.load_all()
    print(f"Cihaz: {registry.device} · {registry.encoder_status}")
    for captioner in registry.captioners:
        print(f"  {captioner.label}: {captioner.status}")
    build_demo(registry).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        max_file_size=MAX_UPLOAD,
    )


if __name__ == "__main__":
    main()
