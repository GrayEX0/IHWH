import io
import zipfile
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Optional: PDF export
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm

# -----------------------------
# Helpers
# -----------------------------

def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(x)))

def load_image(file) -> Optional[Image.Image]:
    if not file:
        return None
    img = Image.open(file).convert("RGBA")
    return img

def fit_image_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize and crop to fill (cover) target area."""
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def fit_image_contain(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize to fit inside target area (contain), preserving transparency."""
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_img = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    canvas_img.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2), resized)
    return canvas_img

def parse_hex_color(hex_str: str) -> Tuple[int, int, int, int]:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b, 255)
    return (255, 255, 255, 255)

def get_font(font_name: str, font_size: int, uploaded_ttf_bytes: Optional[bytes]) -> ImageFont.FreeTypeFont:
    # Streamlit Cloud often has DejaVu fonts available via PIL
    # We'll try a few common paths; fallback to default if missing.
    font_size = clamp_int(font_size, 8, 200)

    if uploaded_ttf_bytes:
        try:
            return ImageFont.truetype(io.BytesIO(uploaded_ttf_bytes), font_size)
        except Exception:
            pass

    candidates = []
    if font_name == "DejaVu Sans":
        candidates = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    elif font_name == "DejaVu Serif":
        candidates = ["DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
    elif font_name == "DejaVu Sans Bold":
        candidates = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    elif font_name == "DejaVu Serif Bold":
        candidates = ["DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]
    else:
        candidates = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    for c in candidates:
        try:
            return ImageFont.truetype(c, font_size)
        except Exception:
            continue

    return ImageFont.load_default()

def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, bbox: Tuple[int, int, int, int],
                       font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = bbox
    # measure
    text = text.strip()
    if not text:
        return
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    x = x0 + (x1 - x0 - w) // 2
    y = y0 + (y1 - y0 - h) // 2
    draw.text((x, y), text, font=font, fill=fill)

def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    """Simple word wrap."""
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
    max_w = x1 - x0
    wrapped = wrap_text_to_width(draw, text, font, max_w)
    if not wrapped:
        return
    lines = wrapped.split("\n")
    line_heights = []
    line_widths = []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        line_widths.append(bb[2] - bb[0])
        line_heights.append(bb[3] - bb[1])

    # total block height
    total_h = 0
    for i, lh in enumerate(line_heights):
        total_h += lh
        if i < len(line_heights) - 1:
            total_h += int(lh * (line_spacing - 1.0))

    start_y = y0 + (y1 - y0 - total_h) // 2
    y = start_y
    for i, ln in enumerate(lines):
        x = x0 + (max_w - line_widths[i]) // 2
        draw.text((x, y), ln, font=font, fill=fill)
        y += line_heights[i]
        if i < len(lines) - 1:
            y += int(line_heights[i] * (line_spacing - 1.0))

@dataclass
class CardStyle:
    card_w: int
    card_h: int
    split_ratio: float  # top height = ratio * card_h
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
    box_round: int  # not used in simple rect; kept for future

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

def build_front(card_have: str, card_who: str, front_img: Optional[Image.Image], style: CardStyle) -> Image.Image:
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

    # Divider line
    if style.divider_thickness > 0:
        y = top_h
        draw.rectangle([0, y - style.divider_thickness // 2, W, y + style.divider_thickness // 2],
                       fill=style.divider_color)

    # Image box in TOP section (centered)
    pad = style.padding
    box_margin = style.box_margin
    # Reserve space: top text at top, box in middle, leave small bottom space
    top_text_h = int(top_h * 0.25)
    box_y0 = pad + top_text_h
    box_y1 = top_h - pad
    box_x0 = pad + box_margin
    box_x1 = W - pad - box_margin

    # Keep box aspect by limiting height a bit so it doesn't feel cramped
    # (Optional: user can control margin if they want bigger/smaller)
    if box_y1 - box_y0 < 50:
        box_y0 = pad
    # Draw box
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=style.box_fill_color,
                   outline=style.box_border_color, width=style.box_border)

    # Place image inside box if provided
    if front_img:
        inner_x0 = box_x0 + style.box_border + 6
        inner_y0 = box_y0 + style.box_border + 6
        inner_x1 = box_x1 - style.box_border - 6
        inner_y1 = box_y1 - style.box_border - 6
        inner_w = max(1, inner_x1 - inner_x0)
        inner_h = max(1, inner_y1 - inner_y0)
        placed = fit_image_contain(front_img, inner_w, inner_h)
        card.paste(placed, (inner_x0, inner_y0), placed)

    # Text
    font_top = get_font(style.font_name, style.font_size_top, style.uploaded_ttf_bytes)
    font_bottom = get_font(style.font_name, style.font_size_bottom, style.uploaded_ttf_bytes)

    top_text = f"{style.top_prefix}{card_have}".strip()
    bottom_text = f"{style.bottom_prefix}{card_who}".strip()

    # Top text area: above the image box
    top_text_bbox = (pad, pad, W - pad, pad + top_text_h)
    draw_wrapped_centered(draw, top_text, top_text_bbox, font_top, style.text_color_top)

    # Bottom text area: centered in bottom section
    bottom_bbox = (pad, top_h + pad, W - pad, H - pad)
    draw_wrapped_centered(draw, bottom_text, bottom_bbox, font_bottom, style.text_color_bottom)

    return card

def build_back(back_img: Optional[Image.Image], style: CardStyle) -> Image.Image:
    W, H = style.card_w, style.card_h
    back = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    if back_img:
        fitted = fit_image_cover(back_img, W, H)
        back.paste(fitted, (0, 0), fitted)
    else:
        # default plain back
        draw = ImageDraw.Draw(back)
        draw.rectangle([0, 0, W, H], fill=(240, 240, 240, 255))
        font = get_font("DejaVu Sans Bold", 48, None)
        draw_centered_text(draw, "BACK", (0, 0, W, H), font, (80, 80, 80, 255))
    return back

def read_zip_images(zip_file) -> Dict[str, Image.Image]:
    """Read images from uploaded ZIP; return dict of filename -> PIL image."""
    if not zip_file:
        return {}
    zbytes = zip_file.read()
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    out = {}
    for name in z.namelist():
        low = name.lower()
        if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                with z.open(name) as f:
                    out[name.split("/")[-1]] = Image.open(f).convert("RGBA")
            except Exception:
                pass
    return out

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

def images_to_pdf_bytes(images: List[Image.Image], page_size_name: str, cards_per_row: int, cards_per_col: int,
                        margin_mm: float = 10.0, gap_mm: float = 4.0) -> bytes:
    # Page size
    if page_size_name == "A4":
        page_w, page_h = A4
    else:
        page_w, page_h = letter

    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=(page_w, page_h))

    margin = margin_mm * mm
    gap = gap_mm * mm

    grid_w = page_w - 2 * margin
    grid_h = page_h - 2 * margin

    cell_w = (grid_w - gap * (cards_per_row - 1)) / cards_per_row
    cell_h = (grid_h - gap * (cards_per_col - 1)) / cards_per_col

    idx = 0
    while idx < len(images):
        for r in range(cards_per_col):
            for col in range(cards_per_row):
                if idx >= len(images):
                    break
                img = images[idx].convert("RGB")
                # place into cell with contain behavior
                iw, ih = img.size
                scale = min(cell_w / iw, cell_h / ih)
                rw, rh = iw * scale, ih * scale
                x = margin + col * (cell_w + gap) + (cell_w - rw) / 2
                y = page_h - margin - (r + 1) * cell_h - r * gap + (cell_h - rh) / 2

                # ReportLab requires file-like or temp file; use in-memory PNG
                tmp = io.BytesIO()
                img.save(tmp, format="PNG")
                tmp.seek(0)
                c.drawImage(ImageReader(tmp), x, y, width=rw, height=rh, preserveAspectRatio=True, mask='auto')
                idx += 1
            if idx >= len(images):
                break
        c.showPage()

    c.save()
    return bio.getvalue()

# ReportLab ImageReader import (kept near usage so it’s obvious)
from reportlab.lib.utils import ImageReader

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="I Have… Who Has… Card Generator", layout="wide")

st.title("🃏 I Have… Who Has… Card Generator")
st.caption("Generate two-part front cards (top/bottom) + a shared backside. Export PNGs + print PDFs.")

tab1, tab2 = st.tabs(["✅ Build from CSV (Deck)", "🧪 Single Card Preview"])

with st.sidebar:
    st.header("Card Settings")

    # Card size
    card_w = st.number_input("Card width (px)", min_value=400, max_value=2000, value=825, step=25)
    card_h = st.number_input("Card height (px)", min_value=400, max_value=2600, value=1125, step=25)

    split_ratio = st.slider("Top section height (%)", min_value=35, max_value=70, value=55) / 100.0
    padding = st.slider("Padding (px)", min_value=0, max_value=80, value=24)

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
    uploaded_ttf = st.file_uploader("Optional: upload your own .ttf font", type=["ttf"], key="ttf")
    uploaded_ttf_bytes = uploaded_ttf.read() if uploaded_ttf else None

    font_size_top = st.slider("Top text size", 16, 120, 56)
    font_size_bottom = st.slider("Bottom text size", 16, 120, 56)

    text_color_top_hex = st.color_picker("Top text color", "#111111")
    text_color_bottom_hex = st.color_picker("Bottom text color", "#111111")

    st.subheader("Divider Line")
    divider_thickness = st.slider("Divider thickness (px)", 0, 20, 6)
    divider_color_hex = st.color_picker("Divider color", "#111111")

    st.subheader("Text Prefixes")
    top_prefix = st.text_input("Top prefix", value="I have ")
    bottom_prefix = st.text_input("Bottom prefix", value="Who has ")

    st.subheader("Card Back")
    back_img_file = st.file_uploader("Upload backside image (used for ALL cards)", type=["png", "jpg", "jpeg", "webp"], key="backimg")

    st.subheader("PDF Export")
    page_size_name = st.selectbox("Page size", ["A4", "Letter"])
    cards_per_row = st.slider("Cards per row", 1, 4, 2)
    cards_per_col = st.slider("Cards per column", 1, 5, 2)

# Load style images
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
    box_round=0,
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

with tab2:
    st.subheader("Single Card Preview")
    c1, c2 = st.columns([1, 1])

    with c1:
        have_text = st.text_input("Top text (have)", value="cat")
        who_text = st.text_input("Bottom text (who has)", value="dog")
        front_img_file_single = st.file_uploader("Optional front image for this preview", type=["png", "jpg", "jpeg", "webp"], key="singlefront")
        front_img_single = load_image(front_img_file_single) if front_img_file_single else None

        if st.button("Generate Preview"):
            front = build_front(have_text, who_text, front_img_single, style)
            back = build_back(back_img, style)

            with c2:
                st.markdown("**Front**")
                st.image(front, use_container_width=True)
                st.markdown("**Back**")
                st.image(back, use_container_width=True)

            # downloads
            front_buf = io.BytesIO()
            back_buf = io.BytesIO()
            front.save(front_buf, format="PNG")
            back.save(back_buf, format="PNG")

            st.download_button("Download Front PNG", data=front_buf.getvalue(), file_name="front_preview.png", mime="image/png")
            st.download_button("Download Back PNG", data=back_buf.getvalue(), file_name="back_preview.png", mime="image/png")

with tab1:
    st.subheader("Build a Deck from CSV")

    st.markdown(
        """
**CSV format (recommended):**
- `have_text` (required)
- `who_text` (required)
- `image_filename` (optional) → filename of image inside your ZIP (e.g., `cat.png`)
- `card_id` (optional)

If you use images, upload a ZIP of images whose filenames match `image_filename`.
        """
    )

    sample = pd.DataFrame(
        {
            "have_text": ["cat", "fish", "train"],
            "who_text": ["dog", "bird", "car"],
            "image_filename": ["cat.png", "fish.png", "train.png"],
            "card_id": [1, 2, 3],
        }
    )
    st.download_button(
        "Download Sample CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_i_have_who_has.csv",
        mime="text/csv",
    )

    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    images_zip = st.file_uploader("Optional: Upload ZIP of card images", type=["zip"])

    if csv_file:
        df = pd.read_csv(csv_file).fillna("")
        required = {"have_text", "who_text"}
        if not required.issubset(set(df.columns)):
            st.error("CSV must include columns: have_text, who_text")
        else:
            st.success(f"Loaded {len(df)} rows.")

            zip_imgs = read_zip_images(images_zip) if images_zip else {}
            if images_zip:
                st.info(f"Loaded {len(zip_imgs)} images from ZIP.")

            # quick preview grid (first 6)
            st.markdown("**Preview (first 6 fronts)**")
            preview_cols = st.columns(3)
            for i in range(min(6, len(df))):
                row = df.iloc[i]
                img = None
                if "image_filename" in df.columns and row["image_filename"]:
                    img = zip_imgs.get(str(row["image_filename"]).strip())
                front = build_front(str(row["have_text"]), str(row["who_text"]), img, style)
                with preview_cols[i % 3]:
                    st.image(front, caption=f"#{i+1}", use_container_width=True)

            if st.button("Generate Deck (PNGs + PDFs)"):
                fronts_out: List[Tuple[str, Image.Image]] = []
                backs_out: List[Tuple[str, Image.Image]] = []
                front_images_for_pdf: List[Image.Image] = []
                back_images_for_pdf: List[Image.Image] = []

                for idx, row in df.iterrows():
                    have = str(row["have_text"])
                    who = str(row["who_text"])
                    card_id = str(row["card_id"]).strip() if "card_id" in df.columns and str(row["card_id"]).strip() else f"{idx+1:03d}"

                    img = None
                    if "image_filename" in df.columns:
                        fn = str(row["image_filename"]).strip()
                        if fn:
                            img = zip_imgs.get(fn)

                    front = build_front(have, who, img, style)
                    back = build_back(back_img, style)

                    front_name = f"card_{card_id}_front.png"
                    back_name = f"card_{card_id}_back.png"

                    fronts_out.append((front_name, front))
                    backs_out.append((back_name, back))
                    front_images_for_pdf.append(front)
                    back_images_for_pdf.append(back)

                # ZIP export
                zbytes = make_zip(fronts_out, backs_out)
                st.download_button(
                    "Download ZIP (fronts + backs PNG)",
                    data=zbytes,
                    file_name="i_have_who_has_cards_png.zip",
                    mime="application/zip",
                )

                # PDF export: create two separate PDFs (fronts, backs)
                # NOTE: backs are identical images per card (still useful for print alignment / count)
                fronts_pdf = export_pdf(front_images_for_pdf, page_size_name, cards_per_row, cards_per_col)
                backs_pdf = export_pdf(back_images_for_pdf, page_size_name, cards_per_row, cards_per_col)

                st.download_button(
                    "Download PDF (Fronts)",
                    data=fronts_pdf,
                    file_name="i_have_who_has_fronts.pdf",
                    mime="application/pdf",
                )
                st.download_button(
                    "Download PDF (Backs)",
                    data=backs_pdf,
                    file_name="i_have_who_has_backs.pdf",
                    mime="application/pdf",
                )

# --- PDF export wrapper (defined after use to keep UI section readable) ---
def export_pdf(images: List[Image.Image], page_size_name: str, cards_per_row: int, cards_per_col: int) -> bytes:
    # We generate PDF with reportlab using in-memory images
    if page_size_name == "A4":
        page_w, page_h = A4
    else:
        page_w, page_h = letter

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

                # convert to RGB for PDF
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
