import io
import re
import zipfile
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# PDF export
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# -----------------------------
# Utilities
# -----------------------------

def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(x)))

def safe_stem(filename: str) -> str:
    # Remove extension, clean up separators
    stem = re.sub(r"\.[^.]+$", "", filename)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem

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

    # Streamlit Cloud typically has DejaVu
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

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    words = (text or "").split()
    if not words:
        return ""
    lines = []
    line = words[0]
    for w in words[1:]:
        test = f"{line} {w}"
        tw = draw.textbbox((0, 0), test, font=font)[2]
        if tw <= max_w:
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return "\n".join(lines)

def draw_wrapped_centered(draw: ImageDraw.ImageDraw, text: str, bbox, font, fill, line_spacing=1.15):
    x0, y0, x1, y1 = bbox
    max_w = max(1, x1 - x0)
    wrapped = wrap_text(draw, (text or "").strip(), font, max_w)
    if not wrapped:
        return
    lines = wrapped.split("\n")

    widths, heights = [], []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])

    total_h = 0
    for i, h in enumerate(heights):
        total_h += h
        if i < len(heights) - 1:
            total_h += int(h * (line_spacing - 1.0))

    start_y = y0 + (y1 - y0 - total_h) // 2
    y = start_y
    for i, ln in enumerate(lines):
        x = x0 + (max_w - widths[i]) // 2
        draw.text((x, y), ln, font=font, fill=fill)
        y += heights[i]
        if i < len(lines) - 1:
            y += int(heights[i] * (line_spacing - 1.0))

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
                # flatten alpha for PDF safety
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
# Card rendering
# -----------------------------

@dataclass
class CardStyle:
    card_w: int
    card_h: int
    split_ratio: float
    padding: int

    # backgrounds
    top_color: Tuple[int, int, int, int]
    bottom_color: Tuple[int, int, int, int]
    top_bg_img: Optional[Image.Image]
    bottom_bg_img: Optional[Image.Image]

    # image box
    box_margin: int
    box_border: int
    box_border_color: Tuple[int, int, int, int]
    box_fill_color: Tuple[int, int, int, int]

    # text
    font_name: str
    font_size_top: int
    font_size_bottom: int
    text_color_top: Tuple[int, int, int, int]
    text_color_bottom: Tuple[int, int, int, int]
    uploaded_ttf_bytes: Optional[bytes]

    # labels
    top_prefix: str
    bottom_prefix: str

    # divider
    divider_thickness: int
    divider_color: Tuple[int, int, int, int]

def build_front(have_item: str, who_item: str, front_img: Optional[Image.Image], style: CardStyle) -> Image.Image:
    W, H = style.card_w, style.card_h
    top_h = int(H * style.split_ratio)
    bottom_h = H - top_h

    card = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(card)

    # Top background
    top_area = Image.new("RGBA", (W, top_h), style.top_color)
    if style.top_bg_img:
        bg = fit_image_cover(style.top_bg_img, W, top_h)
        top_area = Image.alpha_composite(top_area, bg)
    card.paste(top_area, (0, 0), top_area)

    # Bottom background
    bottom_area = Image.new("RGBA", (W, bottom_h), style.bottom_color)
    if style.bottom_bg_img:
        bg = fit_image_cover(style.bottom_bg_img, W, bottom_h)
        bottom_area = Image.alpha_composite(bottom_area, bg)
    card.paste(bottom_area, (0, top_h), bottom_area)

    # Divider
    if style.divider_thickness > 0:
        y = top_h
        draw.rectangle(
            [0, y - style.divider_thickness // 2, W, y + style.divider_thickness // 2],
            fill=style.divider_color
        )

    pad = style.padding

    # Top text area
    top_text_h = int(top_h * 0.25)
    font_top = get_font(style.font_name, style.font_size_top, style.uploaded_ttf_bytes)
    font_bottom = get_font(style.font_name, style.font_size_bottom, style.uploaded_ttf_bytes)

    top_text = f"{style.top_prefix}{have_item}".strip()
    bottom_text = f"{style.bottom_prefix}{who_item}".strip()

    draw_wrapped_centered(draw, top_text, (pad, pad, W - pad, pad + top_text_h), font_top, style.text_color_top)

    # Image box (top section, below the top text)
    box_x0 = pad + style.box_margin
    box_x1 = W - pad - style.box_margin
    box_y0 = pad + top_text_h
    box_y1 = top_h - pad

    draw.rectangle(
        [box_x0, box_y0, box_x1, box_y1],
        fill=style.box_fill_color,
        outline=style.box_border_color,
        width=style.box_border
    )

    if front_img:
        inner_x0 = box_x0 + style.box_border + 6
        inner_y0 = box_y0 + style.box_border + 6
        inner_x1 = box_x1 - style.box_border - 6
        inner_y1 = box_y1 - style.box_border - 6
        inner_w = max(1, inner_x1 - inner_x0)
        inner_h = max(1, inner_y1 - inner_y0)
        placed = fit_image_contain(front_img, inner_w, inner_h)
        card.paste(placed, (inner_x0, inner_y0), placed)

    # Bottom text area
    draw_wrapped_centered(draw, bottom_text, (pad, top_h + pad, W - pad, H - pad), font_bottom, style.text_color_bottom)

    return card

def build_back(back_img: Optional[Image.Image], style: CardStyle) -> Image.Image:
    W, H = style.card_w, style.card_h
    back = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    if back_img:
        fitted = fit_image_cover(back_img, W, H)
        back.paste(fitted, (0, 0), fitted)
    else:
        # clean default
        draw = ImageDraw.Draw(back)
        draw.rectangle([0, 0, W, H], fill=(240, 240, 240, 255))
    return back


# -----------------------------
# Deck building + loop checking
# -----------------------------

def build_cycle(items: List[str], rng: random.Random) -> List[Tuple[str, str]]:
    """
    Create a single-cycle deck: (have_i -> who_{i+1}) with last -> first.
    Returns list of (have_item, who_item)
    """
    items = [i.strip() for i in items if i.strip()]
    if len(items) < 2:
        raise ValueError("Need at least 2 items to build a loop.")
    # Shuffle but keep deterministic if seeded
    order = items[:]
    rng.shuffle(order)
    pairs = []
    for i in range(len(order)):
        have_item = order[i]
        who_item = order[(i + 1) % len(order)]
        pairs.append((have_item, who_item))
    return pairs

def check_single_loop(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Verifies:
    - each item appears exactly once as 'have'
    - each item appears exactly once as 'who'
    - forms one cycle using mapping have->who
    """
    if not pairs:
        return {"ok": "false", "reason": "No pairs."}

    have_list = [a for a, _ in pairs]
    who_list = [b for _, b in pairs]
    have_set = set(have_list)
    who_set = set(who_list)

    if len(have_set) != len(have_list):
        return {"ok": "false", "reason": "Duplicate 'have' items detected."}
    if len(who_set) != len(who_list):
        return {"ok": "false", "reason": "Duplicate 'who has' items detected."}
    if have_set != who_set:
        return {"ok": "false", "reason": "Items in 'have' and 'who has' do not match exactly."}

    # mapping
    mapping = {a: b for a, b in pairs}
    if len(mapping) != len(pairs):
        return {"ok": "false", "reason": "Mapping size mismatch (duplicates in 'have')."}

    # walk cycle
    start = have_list[0]
    visited = set()
    cur = start
    for _ in range(len(pairs)):
        if cur in visited:
            return {"ok": "false", "reason": "Loop repeats early (not a single clean cycle)."}
        visited.add(cur)
        cur = mapping[cur]

    if cur != start:
        return {"ok": "false", "reason": "Does not return to start (not a closed loop)."}
    if len(visited) != len(pairs):
        return {"ok": "false", "reason": "Not all items are connected in one loop."}

    return {"ok": "true", "reason": "Perfect single-loop deck verified."}


def normalize_items_from_text(text: str) -> List[str]:
    lines = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
    return lines


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="I Have… Who Has… Card Generator", layout="wide")
st.title("🃏 I Have… Who Has… Card Generator (No CSV)")
st.caption("Type words and/or upload images → auto-build a perfect loop deck → export PNGs + PDFs.")

# Sidebar settings
with st.sidebar:
    st.header("Card Layout")

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

    st.subheader("Image Box")
    box_margin = st.slider("Image box margin (px)", 0, 80, 18)
    box_border = st.slider("Box border thickness (px)", 0, 20, 4)
    box_border_color_hex = st.color_picker("Box border color", "#222222")
    box_fill_color_hex = st.color_picker("Box fill color", "#FFFFFF")

    st.subheader("Text Style")
    font_name = st.selectbox("Font", ["DejaVu Sans", "DejaVu Sans Bold", "DejaVu Serif", "DejaVu Serif Bold"])
    uploaded_ttf = st.file_uploader("Optional: upload .ttf font", type=["ttf"], key="ttf")
    uploaded_ttf_bytes = uploaded_ttf.read() if uploaded_ttf else None

    font_size_top = st.slider("Top text size", 16, 120, 56)
    font_size_bottom = st.slider("Bottom text size", 16, 120, 56)
    text_color_top_hex = st.color_picker("Top text color", "#111111")
    text_color_bottom_hex = st.color_picker("Bottom text color", "#111111")

    st.subheader("Divider Line")
    divider_thickness = st.slider("Divider thickness (px)", 0, 20, 6)
    divider_color_hex = st.color_picker("Divider color", "#111111")

    st.subheader("Back of Card")
    back_img_file = st.file_uploader("Upload backside image (used for ALL cards)", type=["png", "jpg", "jpeg", "webp"], key="backimg")

    st.subheader("PDF Export")
    page_size_name = st.selectbox("Page size", ["A4", "Letter"])
    cards_per_row = st.slider("Cards per row", 1, 4, 2)
    cards_per_col = st.slider("Cards per column", 1, 5, 2)

# Load background/back images
top_bg_img = load_image(top_bg_file) if top_bg_file else None
bottom_bg_img = load_image(bottom_bg_file) if bottom_bg_file else None
back_img = load_image(back_img_file) if back_img_file else None

# Main: inputs
left, right = st.columns([1.05, 1])

with left:
    st.subheader("1) Add your items (no CSV)")
    st.write("You can type words, upload images, or both.")

    words_text = st.text_area(
        "Type one item per line (optional if you upload images)",
        value="cat\ndog\nfish\nbird",
        height=180
    )

    uploaded_images = st.file_uploader(
        "Upload images for the cards (optional). You can upload many at once.",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    st.subheader("2) Deck builder")
    seed_on = st.checkbox("Use a fixed shuffle seed (repeatable)", value=False)
    seed_val = st.number_input("Seed", 0, 999999, 1234, disabled=not seed_on)
    rng = random.Random(int(seed_val) if seed_on else None)

    st.write("**Text prefixes**")
    custom_prefix = st.checkbox("Customize the 'I have' / 'Who has' text", value=False)
    if custom_prefix:
        top_prefix = st.text_input("Top prefix", value="I have ")
        bottom_prefix = st.text_input("Bottom prefix", value="Who has ")
    else:
        top_prefix = "I have "
        bottom_prefix = "Who has "

    st.write("**How should images match items?**")
    match_mode = st.radio(
        "Image matching",
        ["Match by upload order (1st word gets 1st image)", "Match by filename (filename must match item text)"],
        index=0
    )

    build_btn = st.button("✅ Build Deck + Check Loop", type="primary")

with right:
    st.subheader("Preview")
    preview_slot = st.empty()
    status_slot = st.empty()

# Build style
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
    font_name=font_name,
    font_size_top=int(font_size_top),
    font_size_bottom=int(font_size_bottom),
    text_color_top=parse_hex_color(text_color_top_hex),
    text_color_bottom=parse_hex_color(text_color_bottom_hex),
    uploaded_ttf_bytes=uploaded_ttf_bytes,
    top_prefix=top_prefix,
    bottom_prefix=bottom_prefix,
    divider_thickness=int(divider_thickness),
    divider_color=parse_hex_color(divider_color_hex),
)

# When user clicks build
if build_btn:
    # 1) items from words or images
    items_from_words = normalize_items_from_text(words_text)

    imgs: List[Tuple[str, Image.Image]] = []
    if uploaded_images:
        for f in uploaded_images:
            try:
                imgs.append((f.name, Image.open(f).convert("RGBA")))
            except Exception:
                pass

    if not items_from_words and imgs:
        # derive item names from filenames
        items = [safe_stem(name) for name, _ in imgs]
    else:
        items = items_from_words[:]

    # de-dup while preserving order
    seen = set()
    deduped = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(it.strip())
    items = deduped

    if len(items) < 2:
        status_slot.error("Please provide at least 2 items (type words and/or upload at least 2 images).")
    else:
        # 2) create perfect loop
        pairs = build_cycle(items, rng)

        # 3) loop check
        loop_result = check_single_loop(pairs)
        if loop_result["ok"] == "true":
            status_slot.success(loop_result["reason"])
        else:
            status_slot.error(loop_result["reason"])

        # 4) create image map
        image_map: Dict[str, Image.Image] = {}
        if imgs:
            if match_mode.startswith("Match by upload order"):
                # pair by index order (min length)
                for i, item in enumerate(items):
                    if i < len(imgs):
                        image_map[item] = imgs[i][1]
            else:
                # match by filename stem (case-insensitive)
                stem_map = {safe_stem(name).strip().lower(): im for name, im in imgs}
                for item in items:
                    image_map[item] = stem_map.get(item.strip().lower())

        # 5) render a preview of first 6
        preview_fronts = []
        for i, (have_item, who_item) in enumerate(pairs[:6]):
            im = image_map.get(have_item)
            preview_fronts.append(build_front(have_item, who_item, im, style))

        with preview_slot.container():
            cols = st.columns(3)
            for i, img in enumerate(preview_fronts):
                with cols[i % 3]:
                    st.image(img, caption=f"Card {i+1}", use_container_width=True)

        # 6) full export
        st.subheader("Exports")
        fronts_out: List[Tuple[str, Image.Image]] = []
        backs_out: List[Tuple[str, Image.Image]] = []
        front_images_for_pdf: List[Image.Image] = []
        back_images_for_pdf: List[Image.Image] = []

        back_card = build_back(back_img, style)

        for idx, (have_item, who_item) in enumerate(pairs, start=1):
            im = image_map.get(have_item)
            front = build_front(have_item, who_item, im, style)

            fronts_out.append((f"card_{idx:03d}_front.png", front))
            backs_out.append((f"card_{idx:03d}_back.png", back_card))

            front_images_for_pdf.append(front)
            back_images_for_pdf.append(back_card)

        zip_bytes = make_zip(fronts_out, backs_out)
        st.download_button(
            "Download ZIP (Fronts + Backs PNG)",
            data=zip_bytes,
            file_name="i_have_who_has_cards_png.zip",
            mime="application/zip",
        )

        fronts_pdf = export_pdf(front_images_for_pdf, page_size_name, cards_per_row, cards_per_col)
        backs_pdf = export_pdf(back_images_for_pdf, page_size_name, cards_per_row, cards_per_col)

        st.download_button("Download PDF (Fronts)", data=fronts_pdf, file_name="i_have_who_has_fronts.pdf", mime="application/pdf")
        st.download_button("Download PDF (Backs)", data=backs_pdf, file_name="i_have_who_has_backs.pdf", mime="application/pdf")

        # Optional: show the exact loop order
        with st.expander("Show the generated loop (for checking)"):
            st.write("This is the exact cycle the generator created:")
            for i, (a, b) in enumerate(pairs, start=1):
                st.write(f"{i:02d}. {a}  →  {b}")
