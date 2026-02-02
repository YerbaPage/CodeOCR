import os
import re
from typing import List, Optional

from PIL import Image as PIL_Image, ImageDraw

from llm_utils import build_folder, get_text_tokens
from tasks.code_clone_detection.data import _split_pair
from text_to_image import (
    COMPRESSION_RATIOS,
    DEFAULT_MARGIN_RATIO,
    prepare_text_for_rendering,
    resize_images_for_compression,
    text_to_image,
    text_to_image_stream,
)


def _find_existing_images(images_dir: str, unique_id: str, ratio: float, separate_mode: bool):
    ratio_str = str(ratio)
    if not os.path.isdir(images_dir):
        return None
    files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")]
    if separate_mode:
        a_pat = re.compile(
            rf"{re.escape(unique_id)}_sep_A_(?:standard|extreme)_ratio{re.escape(ratio_str)}_(\d+)x\1_fs(\d+)_page_(\d+)\.png$"
        )
        b_pat = re.compile(
            rf"{re.escape(unique_id)}_sep_B_(?:standard|extreme)_ratio{re.escape(ratio_str)}_(\d+)x\1_fs(\d+)_page_(\d+)\.png$"
        )
        a_groups = {}
        b_groups = {}
        for p in files:
            bn = os.path.basename(p)
            ma = a_pat.match(bn)
            if ma:
                key = (int(ma.group(1)), int(ma.group(2)))
                a_groups.setdefault(key, []).append((int(ma.group(3)), p))
                continue
            mb = b_pat.match(bn)
            if mb:
                key = (int(mb.group(1)), int(mb.group(2)))
                b_groups.setdefault(key, []).append((int(mb.group(3)), p))
        if len(a_groups) == 1 and len(b_groups) == 1:
            (res_a, fs_a), a_items = next(iter(a_groups.items()))
            (res_b, fs_b), b_items = next(iter(b_groups.items()))
            a_items.sort(key=lambda x: x[0])
            b_items.sort(key=lambda x: x[0])
            paths = [p for _, p in a_items] + [p for _, p in b_items]
            return {
                "paths": paths,
                "res_left": res_a,
                "fs_left": fs_a,
                "res_right": res_b,
                "fs_right": fs_b,
            }
        return None
    pat = re.compile(
        rf"{re.escape(unique_id)}_(?:standard|extreme)_ratio{re.escape(ratio_str)}_(\d+)x\1_fs(\d+)_page_(\d+)\.png$"
    )
    groups = {}
    for p in files:
        bn = os.path.basename(p)
        m = pat.match(bn)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            groups.setdefault(key, []).append((int(m.group(3)), p))
    if len(groups) == 1:
        (res, fs), items = next(iter(groups.items()))
        items.sort(key=lambda x: x[0])
        paths = [p for _, p in items]
        return {"paths": paths, "res": res, "fs": fs}
    return None


def calculate_image_tokens_from_pil_images(pil_images: List[PIL_Image.Image]) -> int:
    """
    Calculate image token count from PIL Image object list.
    
    Args:
        pil_images: List of PIL Image objects
        
    Returns:
        Total token count
    """
    import math

    total_tokens = 0
    base_image_tokens = 170
    patch_size = 14

    for img in pil_images:
        width, height = img.size
        num_patches_w = math.ceil(width / patch_size)
        num_patches_h = math.ceil(height / patch_size)
        total_patches = num_patches_w * num_patches_h
        image_tokens = base_image_tokens + min(total_patches * 2, 100)
        total_tokens += int(image_tokens)

    return total_tokens


def calculate_image_tokens_with_processor_from_pil(
    pil_images: List[PIL_Image.Image], processor=None
) -> Optional[int]:
    """
    Calculate image token count from PIL Image objects using AutoProcessor.
    
    Args:
        pil_images: List of PIL Image objects
        processor: AutoProcessor instance
        
    Returns:
        Total token count, or None if processor is not available
    """
    if processor is None:
        return None

    total_tokens = 0
    for pil_image in pil_images:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image.convert("RGB")},
                        {"type": "text", "text": ""},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            image_tokens = 0

            if "image_grid_thw" in inputs:
                grid_info = inputs["image_grid_thw"]
                height_grid = grid_info[0][1].item()
                width_grid = grid_info[0][2].item()
                image_tokens = height_grid * width_grid
            elif "pixel_values" in inputs:
                image_tokens = inputs["pixel_values"].shape[0]
            else:
                image_tokens = calculate_image_tokens_from_pil_images([pil_image])

            total_tokens += image_tokens

        except Exception as e:
            print(f"  Warning: Error calculating image tokens with Processor: {e}")
            total_tokens += calculate_image_tokens_from_pil_images([pil_image])

    return total_tokens


def _wrap_lines(text: str, font, max_width: int, preserve_newlines: bool, temp_draw) -> List[str]:
    processed = prepare_text_for_rendering(text, preserve_newlines=preserve_newlines)
    lines = []
    current = ""
    current_width = 0
    for ch in processed:
        try:
            ch_w = temp_draw.textlength(ch, font=font)
        except Exception:
            bbox = temp_draw.textbbox((0, 0), ch, font=font)
            ch_w = bbox[2] - bbox[0]
        if preserve_newlines and ch == "\n":
            if current:
                lines.append(current)
            else:
                lines.append("")
            current = ""
            current_width = 0
            continue
        if current_width + ch_w > max_width and current:
            lines.append(current)
            current = ch
            current_width = ch_w
        else:
            current += ch
            current_width += ch_w
    if current:
        lines.append(current)
    return lines


def render_pair_images(
    left_text: str,
    right_text: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: Optional[str],
    preserve_newlines: bool,
    enable_bold: bool = False,
    bg_color: str = "white",
    text_color: str = "black",
    enable_syntax_highlight: bool = False,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    theme: str = "light",
) -> List[PIL_Image.Image]:
    """
    Render two code snippets side-by-side as images.
    
    Args:
        left_text: Left code snippet
        right_text: Right code snippet
        width: Image width
        height: Image height
        font_size: Font size
        line_height: Line height multiplier
        dpi: DPI setting
        font_path: Font path (optional)
        preserve_newlines: Whether to preserve newlines
        enable_bold: Whether to bold text
        bg_color: Background color
        text_color: Text color
        enable_syntax_highlight: Whether to enable syntax highlighting
        language: Programming language
        filename: Filename (for language detection)
        theme: Syntax highlighting theme
        
    Returns:
        List of rendered image pages
    """
    margin_px = int(width * DEFAULT_MARGIN_RATIO)
    col_gap = int(width * DEFAULT_MARGIN_RATIO)

    half_width = (width - 2 * margin_px - col_gap) // 2
    left_imgs = text_to_image(
        left_text,
        width=half_width,
        height=height - 2 * margin_px,
        font_size=font_size,
        line_height=line_height,
        margin_px=0,
        dpi=dpi,
        font_path=font_path,
        bg_color=bg_color,
        text_color=text_color,
        preserve_newlines=preserve_newlines,
        enable_syntax_highlight=enable_syntax_highlight,
        filename=filename,
        language=language,
        should_crop_whitespace=False,
        enable_two_column=False,
        enable_bold=enable_bold,
        theme=theme,
    )
    right_imgs = text_to_image(
        right_text,
        width=half_width,
        height=height - 2 * margin_px,
        font_size=font_size,
        line_height=line_height,
        margin_px=0,
        dpi=dpi,
        font_path=font_path,
        bg_color=bg_color,
        text_color=text_color,
        preserve_newlines=preserve_newlines,
        enable_syntax_highlight=enable_syntax_highlight,
        filename=filename,
        language=language,
        should_crop_whitespace=False,
        enable_two_column=False,
        enable_bold=enable_bold,
        theme=theme,
    )
    pages = []
    max_pages = max(len(left_imgs), len(right_imgs))
    for i in range(max_pages):
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        if i < len(left_imgs):
            img.paste(left_imgs[i], (margin_px, margin_px))
        if i < len(right_imgs):
            img.paste(right_imgs[i], (margin_px + half_width + col_gap, margin_px))
        draw.line(
            [
                (margin_px + half_width + col_gap // 2, margin_px),
                (margin_px + half_width + col_gap // 2, height - margin_px),
            ],
            fill="gray",
            width=1,
        )
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        pages.append(img)
    return pages


def render_pair_images_stream(
    left_text: str,
    right_text: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: Optional[str],
    preserve_newlines: bool,
    enable_bold: bool = False,
    bg_color: str = "white",
    text_color: str = "black",
    enable_syntax_highlight: bool = False,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    theme: str = "light",
):
    margin_px = int(width * DEFAULT_MARGIN_RATIO)
    col_gap = int(width * DEFAULT_MARGIN_RATIO)
    half_width = (width - 2 * margin_px - col_gap) // 2
    left_gen = text_to_image_stream(
        left_text,
        width=half_width,
        height=height - 2 * margin_px,
        font_size=font_size,
        line_height=line_height,
        margin_px=0,
        dpi=dpi,
        font_path=font_path,
        bg_color=bg_color,
        text_color=text_color,
        preserve_newlines=preserve_newlines,
        enable_syntax_highlight=enable_syntax_highlight,
        filename=filename,
        language=language,
        should_crop_whitespace=False,
        enable_two_column=False,
        enable_bold=enable_bold,
        theme=theme,
    )
    right_gen = text_to_image_stream(
        right_text,
        width=half_width,
        height=height - 2 * margin_px,
        font_size=font_size,
        line_height=line_height,
        margin_px=0,
        dpi=dpi,
        font_path=font_path,
        bg_color=bg_color,
        text_color=text_color,
        preserve_newlines=preserve_newlines,
        enable_syntax_highlight=enable_syntax_highlight,
        filename=filename,
        language=language,
        should_crop_whitespace=False,
        enable_two_column=False,
        enable_bold=enable_bold,
        theme=theme,
    )
    import itertools

    for l_img, r_img in itertools.zip_longest(left_gen, right_gen):
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        if l_img is not None:
            img.paste(l_img, (margin_px, margin_px))
        if r_img is not None:
            img.paste(r_img, (margin_px + half_width + col_gap, margin_px))
        draw.line(
            [
                (margin_px + half_width + col_gap // 2, margin_px),
                (margin_px + half_width + col_gap // 2, height - margin_px),
            ],
            fill="gray",
            width=1,
        )
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        yield img


def render_images_for_file(
    file_path: str,
    output_dir: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: Optional[str],
    preserve_newlines: bool,
    enable_bold: bool = False,
    resize_mode: bool = False,
    enable_syntax_highlight: bool = False,
    language: Optional[str] = None,
    separate_mode: bool = False,
) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    split = _split_pair(content)
    image_paths: List[str] = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    lh_str = str(line_height).replace(".", "_")
    common_folder_parts = [
        "code_clone_detection",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
        base_name,
    ]
    common_folder_kwargs = {
        "enable_syntax_highlight": enable_syntax_highlight,
        "preserve_newlines": preserve_newlines,
        "enable_bold": enable_bold,
    }
    if separate_mode:
        left, right = (split if split else (content, ""))
        left_images = text_to_image(
            left,
            width=width,
            height=height,
            font_size=font_size,
            line_height=line_height,
            dpi=dpi,
            font_path=font_path,
            preserve_newlines=preserve_newlines,
            enable_syntax_highlight=enable_syntax_highlight,
            filename=file_path,
            language=language,
            should_crop_whitespace=False,
            enable_two_column=False,
            enable_bold=enable_bold,
        )
        folder = build_folder(output_dir, common_folder_parts, **common_folder_kwargs)
        os.makedirs(folder, exist_ok=True)
        for i, img in enumerate(left_images, 1):
            p = os.path.join(folder, f"sep_A_fs{font_size}_page_{i:03d}.png")
            img.save(p)
            image_paths.append(os.path.abspath(p))
        if right and right.strip():
            right_images = text_to_image(
                right,
                width=width,
                height=height,
                font_size=font_size,
                line_height=line_height,
                dpi=dpi,
                font_path=font_path,
                preserve_newlines=preserve_newlines,
                enable_syntax_highlight=enable_syntax_highlight,
                filename=file_path,
                language=language,
                should_crop_whitespace=False,
                enable_two_column=False,
                enable_bold=enable_bold,
            )
            for i, img in enumerate(right_images, 1):
                p = os.path.join(folder, f"sep_B_fs{font_size}_page_{i:03d}.png")
                img.save(p)
                image_paths.append(os.path.abspath(p))
    else:
        left, right = (split if split else (content, ""))
        pages = render_pair_images(
            left,
            right,
            width,
            height,
            font_size,
            line_height,
            dpi,
            font_path,
            preserve_newlines,
            enable_bold,
            bg_color="white",
            text_color="black",
            enable_syntax_highlight=enable_syntax_highlight,
            language=language,
            filename=file_path,
        )
        folder = build_folder(output_dir, common_folder_parts, **common_folder_kwargs)
        os.makedirs(folder, exist_ok=True)
        for i, img in enumerate(pages, 1):
            p = os.path.join(folder, f"fs{font_size}_page_{i:03d}.png")
            img.save(p)
            image_paths.append(os.path.abspath(p))
    if resize_mode:
        combined_text_tokens = get_text_tokens(left) + get_text_tokens(right)
        resized = resize_images_for_compression(
            pages, combined_text_tokens, COMPRESSION_RATIOS
        )
        for ratio, (res_imgs, target_res) in resized.items():
            sub = os.path.join(folder, f"resize_ratio{ratio}_{target_res}x{target_res}")
            os.makedirs(sub, exist_ok=True)
            for i, img in enumerate(res_imgs, 1):
                rp = os.path.join(sub, f"fs{font_size}_page_{i:03d}.png")
                img.save(rp)
                print(os.path.abspath(rp))
    return image_paths
