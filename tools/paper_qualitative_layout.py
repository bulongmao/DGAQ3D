#!/usr/bin/env python
"""Build a compact paper figure from paired qualitative case rows."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROW_HEADER_HEIGHT = 48
PANEL_TITLE_HEIGHT = 38
SOURCE_PANEL_GAP = 8
SOURCE_PANEL_COUNT = 4
PAPER_PANEL_TITLE_HEIGHT = 68
PAPER_PANEL_TITLE_FONT_SIZE = 38
PAPER_CAPTION_HEIGHT = 82
PAPER_CAPTION_FONT_SIZE = 40
COLORS = {
    'gt': (236, 178, 30),
    'baseline': (25, 159, 199),
    'ours': (36, 166, 86),
    'text': (28, 32, 37),
    'rule': (210, 214, 220),
    'paper': (255, 255, 255),
}


def load_font(size, bold=False):
    names = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
        if bold else
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_font(text, max_width, preferred_size, min_size=18, bold=False):
    """Return the largest font that keeps one-line paper labels in bounds."""
    for size in range(preferred_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        left, _, right, _ = font.getbbox(text)
        if right - left <= max_width:
            return font
    return load_font(min_size, bold=bold)


def draw_centered_label(draw, box, text, preferred_size, bold=True):
    left, top, right, bottom = box
    font = fit_font(
        text, max_width=max(right - left - 24, 1),
        preferred_size=preferred_size, bold=bold)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_height = text_box[3] - text_box[1]
    y = top + (bottom - top - text_height) / 2.0 - text_box[1]
    draw.text((left + 12, y), text, fill=COLORS['text'], font=font)


def parse_case_ranks(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    ranks = [item.strip() for item in str(value).split(',') if item.strip()]
    if not ranks:
        raise ValueError('At least one main-figure case rank is required.')
    return [int(item) for item in ranks]


def case_row_path(source_dir, rank):
    matches = sorted((source_dir / 'cases').glob('{:02d}_*.png'.format(rank)))
    if len(matches) != 1:
        raise FileNotFoundError(
            'Expected one rendered row for rank {}, found {}.'.format(
                rank, len(matches)))
    return matches[0]


def source_panel_width(row):
    content_width = row.width - SOURCE_PANEL_GAP * (SOURCE_PANEL_COUNT - 1)
    if content_width <= 0 or content_width % SOURCE_PANEL_COUNT:
        raise ValueError('Unexpected paired-case row width: {}'.format(row.width))
    return content_width // SOURCE_PANEL_COUNT


def extract_source_panel(row, index):
    panel_width = source_panel_width(row)
    left = index * (panel_width + SOURCE_PANEL_GAP)
    top = ROW_HEADER_HEIGHT + PANEL_TITLE_HEIGHT
    if not 0 <= index < SOURCE_PANEL_COUNT or top >= row.height:
        raise ValueError('Invalid source panel index or row geometry.')
    return row.crop((left, top, left + panel_width, row.height)).convert('RGB')


def scaled_target_rect(case, image_size, reference_size=(1600.0, 640.0)):
    left, top, right, bottom = [float(value) for value in case['target_rect']]
    scale_x = image_size[0] / reference_size[0]
    scale_y = image_size[1] / reference_size[1]
    return (left * scale_x, top * scale_y,
            right * scale_x, bottom * scale_y)


def clipped_focus_box(image_size, target_rect, output_aspect):
    width, height = image_size
    left, top, right, bottom = target_rect
    target_width = max(right - left, 1.0)
    target_height = max(bottom - top, 1.0)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_width = max(target_width * 8.0, width * 0.20)
    crop_height = max(target_height * 6.0, height * 0.36)
    if crop_width / crop_height < output_aspect:
        crop_width = crop_height * output_aspect
    else:
        crop_height = crop_width / output_aspect
    crop_width = min(crop_width, float(width))
    crop_height = min(crop_height, float(height))
    left = min(max(center_x - crop_width / 2.0, 0.0), width - crop_width)
    top = min(max(center_y - crop_height / 2.0, 0.0), height - crop_height)
    return (int(round(left)), int(round(top)),
            int(round(left + crop_width)), int(round(top + crop_height)))


def resize_to_fill(image, size):
    target_width, target_height = size
    source_aspect = image.width / max(float(image.height), 1.0)
    target_aspect = target_width / max(float(target_height), 1.0)
    if source_aspect > target_aspect:
        resized_height = target_height
        resized_width = int(round(resized_height * source_aspect))
    else:
        resized_width = target_width
        resized_height = int(round(resized_width / source_aspect))
    resized = image.resize((resized_width, resized_height), Image.LANCZOS)
    left = max((resized_width - target_width) // 2, 0)
    top = max((resized_height - target_height) // 2, 0)
    return resized.crop((left, top, left + target_width, top + target_height))


def make_labeled_panel(image, size, title, accent):
    title_height = PAPER_PANEL_TITLE_HEIGHT
    panel = Image.new(
        'RGB', (size[0], size[1] + title_height), COLORS['paper'])
    panel.paste(resize_to_fill(image, size), (0, title_height))
    draw = ImageDraw.Draw(panel)
    draw.rectangle(
        (0, title_height, size[0] - 1, title_height + size[1] - 1),
        outline=accent, width=4)
    draw_centered_label(
        draw, (0, 0, size[0], title_height), title,
        PAPER_PANEL_TITLE_FONT_SIZE)
    draw.line((0, title_height - 1, size[0], title_height - 1),
              fill=COLORS['rule'], width=1)
    return panel


def error_label(value):
    return 'miss' if value is None else '{:.2f} m'.format(float(value))


def outcome_text(case):
    status = case['status']
    if status == 'ours_only':
        return 'recovered by DGAQ-3D'
    if status == 'baseline_only':
        return 'failure: DGAQ-3D misses'
    baseline = case.get('baseline_error_m')
    ours = case.get('ours_error_m')
    if baseline is not None and ours is not None:
        return 'localization error {:.2f} -> {:.2f} m'.format(
            float(baseline), float(ours))
    return status.replace('_', ' ')


def build_case_row(case, row_image, case_letter, baseline_name, ours_name,
                   context_width=840, crop_width=588, image_height=336):
    context = extract_source_panel(row_image, 0)
    baseline = extract_source_panel(row_image, 1)
    ours = extract_source_panel(row_image, 2)
    target_rect = scaled_target_rect(case, baseline.size)
    focus_box = clipped_focus_box(
        baseline.size, target_rect, crop_width / float(image_height))
    baseline_focus = baseline.crop(focus_box)
    ours_focus = ours.crop(focus_box)
    panels = (
        make_labeled_panel(context, (context_width, image_height),
                           'Scene | target GT', COLORS['gt']),
        make_labeled_panel(
            baseline_focus, (crop_width, image_height),
            '{} | {}'.format(
                baseline_name, error_label(case.get('baseline_error_m'))),
            COLORS['baseline']),
        make_labeled_panel(
            ours_focus, (crop_width, image_height),
            '{} | {}'.format(
                ours_name, error_label(case.get('ours_error_m'))),
            COLORS['ours']),
    )
    gap = 12
    caption_height = PAPER_CAPTION_HEIGHT
    panel_height = max(panel.height for panel in panels)
    row_width = sum(panel.width for panel in panels) + gap * 2
    row_height = panel_height + caption_height
    output = Image.new('RGB', (row_width, row_height), COLORS['paper'])
    x = 0
    for panel in panels:
        output.paste(panel, (x, 0))
        x += panel.width + gap
    draw = ImageDraw.Draw(output)
    caption = ('({}) {} | {:.1f} m | {:.1f} px | {}').format(
        case_letter, str(case['class_name']).replace('_', ' ').title(),
        float(case['distance_m']), float(case['projected_size_px']),
        outcome_text(case))
    draw.line((0, panel_height, row_width, panel_height),
              fill=COLORS['rule'], width=1)
    draw_centered_label(
        draw, (0, panel_height, row_width, row_height), caption,
        PAPER_CAPTION_FONT_SIZE)
    return output


def build_main_figure(source_dir, output_dir=None, case_ranks=(1, 2, 3),
                      baseline_name='3DPPE', ours_name='DGAQ-3D',
                      context_width=840, crop_width=588, image_height=336):
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = (source_dir if output_dir is None
                  else Path(output_dir).expanduser().resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(source_dir / 'selected_cases.json', 'r') as stream:
        cases = json.load(stream)
    by_rank = {int(case['selection_rank']): case for case in cases}
    requested = parse_case_ranks(case_ranks)
    selected_ranks = [rank for rank in requested if rank in by_rank]
    if len(selected_ranks) < min(3, len(by_rank)):
        selected_ranks.extend(
            rank for rank in sorted(by_rank) if rank not in selected_ranks)
    selected_ranks = selected_ranks[:3]
    if not selected_ranks:
        raise RuntimeError('No rendered cases are available for the main figure.')
    rows = []
    for index, rank in enumerate(selected_ranks):
        with Image.open(case_row_path(source_dir, rank)) as row_image:
            rows.append(build_case_row(
                by_rank[rank], row_image.convert('RGB'),
                chr(ord('a') + index), baseline_name, ours_name,
                context_width=context_width, crop_width=crop_width,
                image_height=image_height))
    margin = 18
    gap = 18
    width = max(row.width for row in rows) + 2 * margin
    height = sum(row.height for row in rows) + gap * (len(rows) - 1) + 2 * margin
    figure = Image.new('RGB', (width, height), COLORS['paper'])
    y = margin
    for row in rows:
        figure.paste(row, (margin, y))
        y += row.height + gap
    png_path = output_dir / 'paired_qualitative_main.png'
    pdf_path = output_dir / 'paired_qualitative_main.pdf'
    figure.save(png_path, dpi=(300, 300), optimize=True)
    figure.save(pdf_path, 'PDF', resolution=300.0)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Repack paired qualitative rows into a paper main figure.')
    parser.add_argument('source_dir')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--case-ranks', default='1,2,3')
    parser.add_argument('--baseline-name', default='3DPPE')
    parser.add_argument('--ours-name', default='DGAQ-3D')
    parser.add_argument('--context-width', type=int, default=840)
    parser.add_argument('--crop-width', type=int, default=588)
    parser.add_argument('--image-height', type=int, default=336)
    return parser.parse_args()


def main():
    args = parse_args()
    png_path, pdf_path = build_main_figure(
        args.source_dir, output_dir=args.out_dir,
        case_ranks=args.case_ranks, baseline_name=args.baseline_name,
        ours_name=args.ours_name, context_width=args.context_width,
        crop_width=args.crop_width, image_height=args.image_height)
    print('[done] PNG: {}'.format(png_path))
    print('[done] PDF: {}'.format(pdf_path))


if __name__ == '__main__':
    main()
