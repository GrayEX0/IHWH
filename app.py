import io
import zipfile
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# PDF export
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# -----------------------------
# Helpers
# -----------------------------

def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(x)))

def load_image(file) -> Optional[Image.Image]:
    if not file:
        return None
    return Image.open(file).convert("RGBA")

def parse_hex_color(hex_str: str) -> Tuple[int, int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    return (255, 255, 255, 255)

def fit_image_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def fit_image_contain(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_img = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    canvas_img.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2), resized)
    return canvas_img

def get_font(font_name: str, font_size: int, uploaded_ttf_bytes: Optional[bytes]) -> ImageFont.FreeTypeFont:
    font_size = clamp_int(font_size, 8, 200)

    if uploaded_ttf_bytes:
        try:
            return ImageFont.truetype(io.BytesIO(uploaded_ttf_bytes), font_size)
        except Exception:
            pass

    candidates = {
        "DejaVu Sans": ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "DejaVu Sans Bold": ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "DejaVu Serif": ["DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
        "DejaVu Serif Bold": ["DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    }.get(font_name, ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])

    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    return ImageFont.load_default()

def draw_centered(draw: ImageDraw.ImageDraw, text: str, bbox, font, fill):
    if not text:
        return
    x0, y0, x1, y1 = bbox
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    x = x0 + (x1 - x0 - tw) // 2
    y = y0 + (y1 - y0 - th) // 2
    draw.text((x, y), text, font=font, fill=fill)

def make_zip(fronts: List[Tuple[str, Image.Image]], backs: List[Tuple[str, Image.Image]]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, img in fronts:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            z.writestr(f"fronts/{fname}", buf.getvalue())
        for fname, img in backs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            z.writestr(f"backs/{fname}", buf.getvalue())
    return bio.getvalue()

def export_pdf(images: List[Image.Image], page_size_name: str, cards_per_row: int, cards_per_col: int) -> bytes:
    page_w, page_h = A4 if page_size_name == "A4" else letter
    margin = 10 * mm
    gap = 4 * mm

    grid_w = page_w - 2 * margin
    grid_h = page_h - 2 * margin
    cell_w = (grid_w - gap * (cards_per_row - 1)) / cards_per_row
    cell_h = (grid_h - gap * (cards_per_col - 1)) / cards_per_col

    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=(page_w, page_h))

    idx = 0
    while idx < len(images):
        for r in range(cards_per_col):
            for col in range(cards_per_row):
                if idx >= len(images):
                    break
                img = images[idx].convert("RGBA")
                rgb = Image.new("RGB", img.size, (255, 255, 255))
                rgb.paste(img, mask=img.split()[-1])

                iw, ih = rgb.size
                scale = min(cell_w / iw, cell_h / ih)
                rw, rh = iw * scale, ih * scale

                x = margin + col * (cell_w + gap) + (cell_w - rw) / 2
                y = page_h - margin - (r + 1) * cell_h - r * gap + (cell_h - rh) / 2

                tmp = io.BytesIO()
                rgb.save(tmp, format="PNG")
                tmp.seek(0)
                c.drawImage(ImageReader(tmp), x, y, width=rw, height=rh, mask='auto')

                idx += 1
            if idx >= len(images):
                break
        c.showPage()

    c.save()
    return bio.getvalue()


# -----------------------------
# Deck building + loop checking
# -----------------------------

def build_cycle_indices(n: int, rng: random.Random, base_order: List[int]) -> List[Tuple[int, int]]:
    """
    Creates one single loop from indices.
    base_order defines the starting order; optional shuffle happens outside.
    """
    if n < 2:
        raise ValueError("Need at least 2 images.")
    order = base_order[:]
    pairs = []
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        pairs.append((a, b))
    return pairs

def check_single_loop_indices(pairs: List[Tuple[int, int]], n: int) -> Tuple[bool, str]:
    if not pairs:
        return False, "No cards were generated."
    have = [a for a, _ in pairs]
    who = [b for _, b in pairs]

    if len(set(have)) != n:
        return False, "Not every image appears exactly once as TOP."
    if len(set(who)) != n:
        return False, "Not every image appears exactly once as BOTTOM."
    if set(have) != set(who) or set(have) != set(range(n)):
        return False, "Mismatch between top/bottom usage."
    mapping = {a: b for a, b in pairs}

    start = have[0]
    visited = set()
    cur = start
    for _ in range(n):
        if cur in visited:
            return False, "Loop repeats early (not a single clean cycle)."
        visited.add(cur)
        cur = mapping[cur]
    if cur != start:
        return False, "Does not return to start (not a closed loop)."
    if len(visited) != n:
        return False, "Not all images are connected."
    return True, "Perfect single-loop deck verified."


# -----------------------------
# Card Rendering
# -----------------------------

@dataclass
class CardStyle:
    card_w: int
    card_h: int
    split_ratio: float
    padding: int

    top_color: Tuple[int, int, int, int]
    bottom_color: Tuple[int, int, int, int]
    top_bg_img: Optional[Image.Image]
    bottom_bg_img: Optional[Image.Image]

    box_margin: int
    box_border: int
    box_border_color: Tuple[int, int, int, int]
    box_fill_color: Tuple[int, int, int, int]

    show_labels: bool
    top_label: str
    bottom_label: str
    font_name: str
    label_font_size: int
    label_color: Tuple[int, int, int, int]
    uploaded_ttf_bytes: Optional[bytes]

    divider_thickness: int
    divider_color: Tuple[int, int, int, int]


def build_front(have_img: Image.Image, who_img: Image.Image, style: CardStyle) -> Image.Image:
    W, H = style.card_w, style.card_h
    top_h = int(H * style.split_ratio)
    bottom_h = H - top_h

    card = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    top_area = Image.new("RGBA", (W, top_h), style.top_color)
    if style.top_bg_img:
        top_area = Image.alpha_composite(top_area, fit_image_cover(style.top_bg_img, W, top_h))
    card.paste(top_area, (0, 0), top_area)

    bottom_area = Image.new("RGBA", (W, bottom_h), style.bottom_color)
    if style.bottom_bg_img:
        bottom_area = Image.alpha_composite(bottom_area, fit_image_cover(style.bottom_bg_img, W, bottom_h))
    card.paste(bottom_area, (0, top_h), bottom_area)

    draw = ImageDraw.Draw(card)

    if style.divider_thickness > 0:
        y = top_h
        draw.rectangle([0, y - style.divider_thickness // 2, W, y + style.divider_thickness // 2],
                       fill=style.divider_color)

    pad = style.padding
    font = get_font(style.font_name, style.label_font_size, style.uploaded_ttf_bytes)
    label_h = int(min(top_h, bottom_h) * 0.18) if style.show_labels else 0

    if style.show_labels:
        draw_centered(draw, style.top_label, (pad, pad, W - pad, pad + label_h), font, style.label_color)
        draw_centered(draw, style.bottom_label, (pad, top_h + pad, W - pad, top_h + pad + label_h),
                      font, style.label_color)

    # Top image box
    top_x0 = pad + style.box_margin
    top_x1 = W - pad - style.box_margin
    top_y0 = pad + label_h
    top_y1 = top_h - pad

    # Bottom image box
    bot_x0 = pad + style.box_margin
    bot_x1 = W - pad - style.box_margin
    bot_y0 = top_h + pad + label_h
    bot_y1 = H - pad

    draw.rectangle([top_x0, top_y0, top_x1, top_y1],
                   fill=style.box_fill_color, outline=style.box_border_color, width=style.box_border)
    draw.rectangle([bot_x0, bot_y0, bot_x1, bot_y1],
                   fill=style.box_fill_color, outline=style.box_border_color, width=style.box_border)

    def paste_into_box(img: Image.Image, x0, y0, x1, y1):
        inner_x0 = x0 + style.box_border + 6
        inner_y0 = y0 + style.box_border + 6
        inner_x1 = x1 - style.box_border - 6
        inner_y1 = y1 - style.box_border - 6
        inner_w = max(1, inner_x1 - inner_x0)
        inner_h = max(1, inner_y1 - inner_y0)
        placed = fit_image_contain(img, inner_w, inner_h)
        card.paste(placed, (inner_x0, inner_y0), placed)

    paste_into_box(have_img, top_x0, top_y0, top_x1, top_y1)
    paste_into_box(who_img, bot_x0, bot_y0, bot_x1, bot_y1)
    return card

def build_back(back_img: Optional[Image.Image], style: CardStyle) -> Image.Image:
    W, H = style.card_w, style.card_h
    back = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    if back_img:
        fitted = fit_image_cover(back_img, W, H)
        back.paste(fitted, (0, 0), fitted)
    else:
        draw = ImageDraw.Draw(back)
        draw.rectangle([0, 0, W, H], fill=(240, 240, 240, 255))
    return back


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Image-Only I Have… Who Has…", layout="wide")
st.title("🃏 Image-Only I Have… Who Has… Deck Generator")
st.caption("Upload images, Check the loop, Download, Enjoy")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Card Settings")
    card_w = st.number_input("Card width (px)", 400, 2000, 825, 25)
    card_h = st.number_input("Card height (px)", 400, 2600, 1125, 25)
    split_ratio = st.slider("Top section height (%)", 35, 70, 55) / 100.0
    padding = st.slider("Padding (px)", 0, 80, 24)

    st.subheader("Top / Bottom Background")
    top_mode = st.radio("Top background mode", ["Color", "Image"], horizontal=True)
    if top_mode == "Color":
        top_color_hex = st.color_picker("Top color", "#F7E8FF")
        top_bg_file = None
    else:
        top_color_hex = "#FFFFFF"
        top_bg_file = st.file_uploader("Upload top background image", type=["png", "jpg", "jpeg", "webp"], key="topbg")

    bottom_mode = st.radio("Bottom background mode", ["Color", "Image"], horizontal=True)
    if bottom_mode == "Color":
        bottom_color_hex = st.color_picker("Bottom color", "#E8F7FF")
        bottom_bg_file = None
    else:
        bottom_color_hex = "#FFFFFF"
        bottom_bg_file = st.file_uploader("Upload bottom background image", type=["png", "jpg", "jpeg", "webp"], key="bottombg")

    st.subheader("Image Boxes")
    box_margin = st.slider("Box margin (px)", 0, 80, 18)
    box_border = st.slider("Box border thickness (px)", 0, 20, 4)
    box_border_color_hex = st.color_picker("Box border color", "#222222")
    box_fill_color_hex = st.color_picker("Box fill color", "#FFFFFF")

    st.subheader("Labels (optional)")
    show_labels = st.checkbox("Show labels", value=True)
    custom_labels = st.checkbox("Customize labels", value=False, disabled=not show_labels)
    if show_labels:
        if custom_labels:
            top_label = st.text_input("Top label", value="I have")
            bottom_label = st.text_input("Bottom label", value="Who has")
        else:
            top_label = "I have"
            bottom_label = "Who has"
    else:
        top_label = ""
        bottom_label = ""

    font_name = st.selectbox("Label font", ["DejaVu Sans", "DejaVu Sans Bold", "DejaVu Serif", "DejaVu Serif Bold"])
    uploaded_ttf = st.file_uploader("Optional: upload .ttf font", type=["ttf"], key="ttf")
    uploaded_ttf_bytes = uploaded_ttf.read() if uploaded_ttf else None
    label_font_size = st.slider("Label font size", 14, 120, 54, disabled=not show_labels)
    label_color_hex = st.color_picker("Label color", "#111111", disabled=not show_labels)

    st.subheader("Divider Line")
    divider_thickness = st.slider("Divider thickness (px)", 0, 20, 6)
    divider_color_hex = st.color_picker("Divider color", "#111111")

    st.subheader("Back of Card")
    back_img_file = st.file_uploader("Upload backside image", type=["png", "jpg", "jpeg", "webp"], key="backimg")

    st.subheader("PDF Export")
    page_size_name = st.selectbox("Page size", ["A4", "Letter"])
    cards_per_row = st.slider("Cards per row", 1, 4, 2)
    cards_per_col = st.slider("Cards per column", 1, 5, 2)

# Load backgrounds/back
top_bg_img = load_image(top_bg_file) if top_bg_file else None
bottom_bg_img = load_image(bottom_bg_file) if bottom_bg_file else None
back_img = load_image(back_img_file) if back_img_file else None

style = CardStyle(
    card_w=int(card_w),
    card_h=int(card_h),
    split_ratio=float(split_ratio),
    padding=int(padding),
    top_color=parse_hex_color(top_color_hex),
    bottom_color=parse_hex_color(bottom_color_hex),
    top_bg_img=top_bg_img,
    bottom_bg_img=bottom_bg_img,
    box_margin=int(box_margin),
    box_border=int(box_border),
    box_border_color=parse_hex_color(box_border_color_hex),
    box_fill_color=parse_hex_color(box_fill_color_hex),
    show_labels=bool(show_labels),
    top_label=top_label.strip(),
    bottom_label=bottom_label.strip(),
    font_name=font_name,
    label_font_size=int(label_font_size),
    label_color=parse_hex_color(label_color_hex),
    uploaded_ttf_bytes=uploaded_ttf_bytes,
    divider_thickness=int(divider_thickness),
    divider_color=parse_hex_color(divider_color_hex),
)

# --- Upload images ---
st.subheader("1) Upload images (2+)")
uploaded_images = st.file_uploader(
    "Upload images (each image becomes an item). No text is used.",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

# Session state: store images + order
def init_images(files):
    imgs = []
    for idx, f in enumerate(files):
        try:
            pil = Image.open(f).convert("RGBA")
            imgs.append({"id": idx, "pil": pil})
        except Exception:
            pass
    st.session_state.imgs = imgs
    st.session_state.order = list(range(len(imgs)))

if "imgs" not in st.session_state:
    st.session_state.imgs = []
if "order" not in st.session_state:
    st.session_state.order = []

if uploaded_images:
    # If uploads changed (count differs), re-init
    if len(st.session_state.imgs) != len(uploaded_images):
        init_images(uploaded_images)

# --- Reorder grid ---
st.subheader("2) Reorder images (thumbnail grid)")
if not st.session_state.imgs or len(st.session_state.imgs) < 2:
    st.info("Upload at least 2 images to enable reordering.")
else:
    cols_count = st.slider("Grid columns", 2, 6, 4)
    thumb_size = st.slider("Thumbnail size (px)", 80, 260, 160)

    def swap_positions(order, i, j):
        order[i], order[j] = order[j], order[i]

    def move_to(order, from_idx, to_idx):
        item = order.pop(from_idx)
        order.insert(to_idx, item)

    # Quick tools
    tool_a, tool_b, tool_c = st.columns([1, 1, 2])
    with tool_a:
        if st.button("↻ Reset to upload order"):
            st.session_state.order = list(range(len(st.session_state.imgs)))
    with tool_b:
        if st.button("🔀 Shuffle order"):
            tmp = st.session_state.order[:]
            random.shuffle(tmp)
            st.session_state.order = tmp
    with tool_c:
        st.caption("Tip: use ⬆️⬇️ to change the loop order (the deck is built from this order).")

    order = st.session_state.order
    n = len(order)

    # Build grid rows
    rows = (n + cols_count - 1) // cols_count
    idx_in_order = 0

    for r in range(rows):
        row_cols = st.columns(cols_count)
        for c in range(cols_count):
            if idx_in_order >= n:
                break

            pos = idx_in_order
            img_idx = order[pos]
            pil = st.session_state.imgs[img_idx]["pil"]

            with row_cols[c]:
                st.image(pil, width=thumb_size, caption=f"#{pos+1}")

                # Move buttons: linear order
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("⬅️", key=f"left_{pos}"):
                        if pos > 0:
                            swap_positions(order, pos, pos - 1)
                            st.session_state.order = order
                            st.rerun()
                with b2:
                    if st.button("➡️", key=f"right_{pos}"):
                        if pos < n - 1:
                            swap_positions(order, pos, pos + 1)
                            st.session_state.order = order
                            st.rerun()
                with b3:
                    if st.button("⬆️", key=f"up_{pos}"):
                        if pos - cols_count >= 0:
                            swap_positions(order, pos, pos - cols_count)
                            st.session_state.order = order
                            st.rerun()
                with b4:
                    if st.button("⬇️", key=f"down_{pos}"):
                        if pos + cols_count < n:
                            swap_positions(order, pos, pos + cols_count)
                            st.session_state.order = order
                            st.rerun()

                # Move-to control
                move_to_pos = st.selectbox(
                    "Move to position",
                    options=list(range(1, n + 1)),
                    index=pos,
                    key=f"moveto_{pos}"
                )
                if move_to_pos != pos + 1:
                    move_to(order, pos, move_to_pos - 1)
                    st.session_state.order = order
                    st.rerun()

            idx_in_order += 1

# --- Build deck ---
st.subheader("3) Build deck (perfect loop)")

colA, colB = st.columns([1, 2])
with colA:
    shuffle_before_build = st.checkbox("Shuffle before building the loop", value=False)
    seed_on = st.checkbox("Use fixed shuffle seed", value=False)
    seed_val = st.number_input("Seed", 0, 999999, 1234, disabled=not seed_on)
    build_btn = st.button("✅ Build Deck + Check Loop", type="primary")

preview = st.empty()
status = st.empty()

if build_btn:
    if not st.session_state.imgs or len(st.session_state.imgs) < 2:
        status.error("Please upload at least 2 images.")
        st.stop()

    imgs = [d["pil"] for d in st.session_state.imgs]
    base_order = st.session_state.order[:]

    rng = random.Random(int(seed_val) if seed_on else None)
    if shuffle_before_build:
        rng.shuffle(base_order)

    pairs = build_cycle_indices(len(imgs), rng, base_order)
    ok, msg = check_single_loop_indices(pairs, len(imgs))
    status.success(msg) if ok else status.error(msg)

    # Preview first 6
    preview_cards = []
    for (have_i, who_i) in pairs[:6]:
        preview_cards.append(build_front(imgs[have_i], imgs[who_i], style))

    with preview.container():
        cols = st.columns(3)
        for i, card_img in enumerate(preview_cards):
            with cols[i % 3]:
                st.image(card_img, caption=f"Card {i+1}", use_container_width=True)

    # Exports
    st.subheader("Exports")
    back_card = build_back(back_img, style)

    fronts_out: List[Tuple[str, Image.Image]] = []
    backs_out: List[Tuple[str, Image.Image]] = []
    front_for_pdf: List[Image.Image] = []
    back_for_pdf: List[Image.Image] = []

    for idx, (have_i, who_i) in enumerate(pairs, start=1):
        front = build_front(imgs[have_i], imgs[who_i], style)
        fronts_out.append((f"card_{idx:03d}_front.png", front))
        backs_out.append((f"card_{idx:03d}_back.png", back_card))
        front_for_pdf.append(front)
        back_for_pdf.append(back_card)

    zip_bytes = make_zip(fronts_out, backs_out)
    st.download_button(
        "Download ZIP (Fronts + Backs PNG)",
        data=zip_bytes,
        file_name="i_have_who_has_image_only_cards.zip",
        mime="application/zip",
    )

    fronts_pdf = export_pdf(front_for_pdf, page_size_name, cards_per_row, cards_per_col)
    backs_pdf = export_pdf(back_for_pdf, page_size_name, cards_per_row, cards_per_col)

    st.download_button("Download PDF (Fronts)", data=fronts_pdf, file_name="fronts.pdf", mime="application/pdf")
    st.download_button("Download PDF (Backs)", data=backs_pdf, file_name="backs.pdf", mime="application/pdf")

    with st.expander("Show loop order (based on your current image order)"):
        st.write("This is the cycle (image positions in your ordered list):")
        for i, (a, b) in enumerate(pairs, start=1):
            st.write(f"{i:02d}. position #{base_order.index(a)+1} → position #{base_order.index(b)+1}")
