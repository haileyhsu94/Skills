#!/usr/bin/env python3
"""Retro collage / vintage zine photo treatment.

Pipeline: segment subject -> desaturate + film grain -> sticker border
-> composite onto patterned retro background with soft shadow.

Requires: mediapipe==0.10.9 (bundled segmentation model, works offline),
opencv-python, pillow, numpy.
    pip install "mediapipe==0.10.9" opencv-python pillow numpy --break-system-packages

Usage:
    python retro_collage.py INPUT.jpg OUTPUT.jpg [options]
    python retro_collage.py INPUT.jpg OUTPUT.jpg --save-cutout cutout.png

Options (all optional; defaults reproduce the reference style):
    --colors HEX HEX        two background pattern colors (default c8d94a 5c9b95)
    --border-color HEX      sticker border color (default f2e9d8 cream)
    --border 14             sticker border thickness in px
    --pattern stripes|dots|sunburst
    --stripe-width 60       stripe width px (stripes pattern)
    --angle 40              stripe rotation degrees (stripes pattern)
    --grain 14              film grain strength (std dev, 0 disables)
    --duotone HEX           optional: tint shadows toward this color instead of pure B&W
    --margin 80             space around sticker on the background
    --save-cutout PATH      also save the transparent sticker cutout PNG
"""
import argparse
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def hex_rgb(s):
    s = s.lstrip('#')
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


# Named palettes: (color1, color2) for the background pattern.
# Cream border (f2e9d8) works with all of them; a few look better with the
# listed border override. All are 60s-70s print-inspired, mid-contrast pairs.
PALETTES = {
    'classic-mod':          ('c8d94a', '5c9b95'),
    'mustard-ink':          ('d9a441', '3a3934'),
    'avocado-cream':        ('8fae3f', 'f2e9d8'),
    'tangerine-teal':       ('e8743b', '2f6d68'),
    'flamingo-olive':       ('e88fa2', '7a7a3a'),
    'cherry-butter':        ('c23b3b', 'f0d982'),
    'grape-soda':           ('6b4d9e', 'e8c94a'),
    'burnt-orange-navy':    ('c65d21', '233a5c'),
    'bubblegum-mint':       ('e991b8', '8fd4b8'),
    'cocoa-peach':          ('6b4a36', 'f0b990'),
    'denim-daisy':          ('3f5f8a', 'f0d060'),
    'rust-sage':            ('b0532a', 'a3b18a'),
    'lemon-lilac':          ('ede05a', 'a98bc4'),
    'tomato-cream':         ('d94f30', 'f2e9d8'),
    'forest-goldenrod':     ('2f5d3a', 'dba52e'),
    'coral-seafoam':        ('e87461', '9adbc5'),
    'plum-mustard':         ('5c3a56', 'd9a441'),
    'sky-poppy':            ('8ec4d6', 'd9442f'),
    'moss-blush':           ('6a7d3f', 'e8b4a0'),
    'midnight-chartreuse':  ('22303a', 'c8d94a'),
    'sunset-duo':           ('e8a13b', 'c24e6a'),
    'teal-tomato':          ('377f77', 'd95d39'),
}


def detect_flush_sides(mask_u8, frac=0.02):
    """Which photo-frame edges does the subject touch?

    Where the subject is cropped by the original photo frame (an arm leaving
    the shot, a torso filling the bottom), the sticker should bleed flush to
    the background edge on that side -- a floating straight-cut edge with
    margin around it looks wrong. Returns dict of side -> bool.
    """
    h, w = mask_u8.shape
    return {
        'top':    (mask_u8[0, :] > 127).mean() > frac,
        'bottom': (mask_u8[-1, :] > 127).mean() > frac,
        'left':   (mask_u8[:, 0] > 127).mean() > frac,
        'right':  (mask_u8[:, -1] > 127).mean() > frac,
    }


def _finish_mask(m):
    """Keep largest component, smooth edges."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == largest, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.GaussianBlur(m, (5, 5), 0)


def _mediapipe_mask(img_rgb):
    import mediapipe as mp
    seg_mod = mp.solutions.selfie_segmentation
    with seg_mod.SelfieSegmentation(model_selection=1) as seg:
        mask = seg.process(img_rgb).segmentation_mask
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_blur = cv2.GaussianBlur(mask_u8, (7, 7), 0)
    _, mask_thresh = cv2.threshold(mask_blur, 130, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(mask_thresh, (5, 5), 0)


def _colorkey_mask(img_rgb):
    """For near-uniform backgrounds (product shots, white studio walls):
    key out everything close to the border color. Returns None if the
    border is not uniform enough to key on."""
    h, w = img_rgb.shape[:2]
    b = max(4, int(min(h, w) * 0.03))
    ring = np.concatenate([img_rgb[:b].reshape(-1, 3), img_rgb[-b:].reshape(-1, 3),
                           img_rgb[:, :b].reshape(-1, 3), img_rgb[:, -b:].reshape(-1, 3)])
    if ring.std(axis=0).mean() > 22:
        return None
    dist = np.linalg.norm(img_rgb.astype(np.int16) - ring.mean(axis=0), axis=2)
    m = np.where(dist > 30, 255, 0).astype(np.uint8)
    return _finish_mask(m)


def _grabcut_mask(img_rgb, iters=6):
    """Generic fallback: GrabCut seeded with border=background,
    center=probable foreground. Inversion guard: if the resulting mask
    claims the border region, flip it."""
    h, w = img_rgb.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    b = max(2, int(min(h, w) * 0.02))
    gc[:b, :] = cv2.GC_BGD; gc[-b:, :] = cv2.GC_BGD
    gc[:, :b] = cv2.GC_BGD; gc[:, -b:] = cv2.GC_BGD
    gc[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_rgb, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    ring = np.concatenate([m[:b].ravel(), m[-b:].ravel(), m[:, :b].ravel(), m[:, -b:].ravel()])
    if (ring > 127).mean() > 0.5:
        m = 255 - m
    return _finish_mask(m)


def segment_subject(img_rgb):
    """Return a soft-edged uint8 mask (0-255) of the foreground subject.

    Strategy chain -- the person model first (best edges on people, and it
    generalizes passably to pets), color-keying for uniform backgrounds
    (product shots, fruit on white), GrabCut as the generic fallback. A
    strategy's result is trusted only if it found a sanely-sized subject.
    """
    plausible = lambda m: 0.05 < (m > 127).mean() < 0.90
    m = _mediapipe_mask(img_rgb)
    if plausible(m):
        return m
    ck = _colorkey_mask(img_rgb)
    if ck is not None and plausible(ck):
        return ck
    return _grabcut_mask(img_rgb)


def vintage_treatment(img, grain=14, duotone=None, seed=7):
    """Desaturate, boost contrast, add photographic grain. Keeps the photo real."""
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    gray = ImageEnhance.Brightness(gray).enhance(1.03)
    if duotone:
        # map shadows toward the duotone color, highlights toward near-white
        out = ImageOps.colorize(gray, black=duotone, white=(245, 242, 235))
    else:
        out = gray.convert('RGB')
    if grain > 0:
        rng = np.random.default_rng(seed)
        arr = np.array(out).astype(np.int16)
        noise = rng.normal(0, grain, (arr.shape[0], arr.shape[1], 1)).astype(np.int16)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.SMOOTH_MORE)
    return ImageEnhance.Sharpness(out).enhance(1.6)


def build_sticker(photo, mask_u8, border_px=14, border_color=(242, 233, 216), pad=40):
    """Cut out the subject and trace it with a sticker border. Returns RGBA."""
    w, h = photo.size
    subject = Image.new('RGBA', (w, h))
    subject.paste(photo, (0, 0))
    subject.putalpha(Image.fromarray(mask_u8))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px * 2 + 1, border_px * 2 + 1))
    dilated = cv2.GaussianBlur(cv2.dilate(mask_u8, k), (5, 5), 0)

    border_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    border_layer.paste(Image.new('RGBA', (w, h), border_color + (255,)), (0, 0),
                       Image.fromarray(dilated))

    sticker = Image.new('RGBA', (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    sticker.paste(border_layer, (pad, pad), border_layer)
    sticker.paste(subject, (pad, pad), subject)
    return sticker


def make_background(size, pattern='stripes', colors=((200, 217, 74), (92, 155, 149)),
                    stripe_width=60, angle=40, grain=10, seed=3):
    """Patterned retro background with paper grain.

    Stripes are drawn on an oversized square then rotated and center-cropped --
    drawing angled polygons directly leaves uncovered corners.
    """
    bg_w, bg_h = size
    c1, c2 = colors
    if pattern == 'stripes':
        diag = int((bg_w ** 2 + bg_h ** 2) ** 0.5) + 20
        big = Image.new('RGB', (diag, diag))
        d = ImageDraw.Draw(big)
        for i, x in enumerate(range(0, diag, stripe_width)):
            d.rectangle([x, 0, x + stripe_width, diag], fill=c1 if i % 2 == 0 else c2)
        big = big.rotate(angle, expand=False)
        left, top = (diag - bg_w) // 2, (diag - bg_h) // 2
        bg = big.crop((left, top, left + bg_w, top + bg_h))
    elif pattern == 'dots':
        bg = Image.new('RGB', size, c1)
        d = ImageDraw.Draw(bg)
        step, r = stripe_width, stripe_width // 4
        for yy in range(0, bg_h + step, step):
            off = (yy // step % 2) * step // 2
            for xx in range(-step, bg_w + step, step):
                d.ellipse([xx + off - r, yy - r, xx + off + r, yy + r], fill=c2)
    elif pattern == 'sunburst':
        bg = Image.new('RGB', size, c1)
        d = ImageDraw.Draw(bg)
        cx, cy = bg_w // 2, bg_h // 2
        n = 24
        radius = int((bg_w ** 2 + bg_h ** 2) ** 0.5)
        for i in range(n):
            a0, a1 = 360 * i / n, 360 * (i + 0.5) / n
            d.pieslice([cx - radius, cy - radius, cx + radius, cy + radius],
                       a0, a1, fill=c2)
    else:
        raise ValueError(f'unknown pattern: {pattern}')

    if grain > 0:
        rng = np.random.default_rng(seed)
        arr = np.array(bg).astype(np.int16)
        noise = rng.normal(0, grain, (arr.shape[0], arr.shape[1], 1)).astype(np.int16)
        bg = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return bg


def layout(sticker_size, margin=80, pad=40, flush=None, aspect=None, coverage=0.8):
    """Compute background size and sticker position.

    Non-flush sides get `margin` of background around the sticker. Flush
    sides (subject cropped by the photo frame) get the sticker pushed PAST
    the canvas edge by `pad`, so the photo's cut edge lands exactly on the
    background edge and the border/margin on that side is cropped away --
    matching how real collages run cropped subjects off the page.

    With `aspect` (w/h float): the canvas takes that exact ratio and is
    sized so the sticker fills at least `coverage` of the limiting
    dimension -- the subject should dominate the frame, not float in a sea
    of pattern. The sticker is centered on non-flush axes.

    NOTE: sticker_size must be the sticker's CONTENT bounding box (crop the
    sticker to its alpha bbox first). Measuring against the full original
    photo canvas -- which includes empty transparent space around the
    subject -- makes subjects come out far smaller than the coverage target.
    """
    flush = flush or {}
    sw, sh = sticker_size
    fl, fr_ = flush.get('left'), flush.get('right')
    ft, fb = flush.get('top'), flush.get('bottom')

    if aspect is None:
        ml = 0 if fl else margin
        mr = 0 if fr_ else margin
        mt = 0 if ft else margin
        mb = 0 if fb else margin
        return (ml + sw + mr, mt + sh + mb), (ml, mt)

    # two candidates: height-limited and width-limited; both satisfy
    # coverage on their limiting dimension -- pick the SMALLER canvas that
    # still contains the sticker, i.e. the higher overall coverage
    cand = []
    ch = sh / coverage
    cw = ch * aspect
    if cw >= sw:
        cand.append((cw, ch))
    cw2 = sw / coverage
    ch2 = cw2 / aspect
    if ch2 >= sh:
        cand.append((cw2, ch2))
    if not cand:  # degenerate; fall back to height-limited
        cand.append((cw, ch))
    cw, ch = min(cand, key=lambda c: c[0] * c[1])
    bg_size = (int(round(cw)), int(round(ch)))

    if fl and not fr_:
        x = 0
    elif fr_ and not fl:
        x = bg_size[0] - sw
    else:
        x = (bg_size[0] - sw) // 2
    if ft and not fb:
        y = 0
    elif fb and not ft:
        y = bg_size[1] - sh
    else:
        y = (bg_size[1] - sh) // 2
    return bg_size, (x, y)


def composite(bg, sticker, pos=(80, 80)):
    """Paste sticker onto background with a soft, natural drop shadow
    (blurred + low opacity -- avoid hard brutalist offsets).
    Uses paste() rather than alpha_composite() because pos may be negative
    (flush/bleed sides)."""
    x, y = pos
    alpha = sticker.split()[-1]
    shadow = Image.new('RGBA', bg.size, (0, 0, 0, 0))
    layer = Image.new('RGBA', sticker.size, (0, 0, 0, 110))
    layer.putalpha(alpha.point(lambda p: int(p * 0.55)))
    shadow.paste(layer, (x + 10, y + 16), layer)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    out = bg.convert('RGBA')
    out.paste(shadow, (0, 0), shadow)
    out.paste(sticker, (x, y), sticker)
    return out.convert('RGB')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--colors', nargs=2, default=None,
                    help='two hex colors; overrides --palette')
    ap.add_argument('--palette', default='classic-mod', choices=sorted(PALETTES),
                    help='named color set (see --list-palettes)')
    ap.add_argument('--list-palettes', action='store_true')
    ap.add_argument('--no-bleed', action='store_true',
                    help='disable flush-edge bleed; keep margin on all sides')
    ap.add_argument('--border-color', default='f2e9d8')
    ap.add_argument('--border', type=int, default=None)
    ap.add_argument('--pattern', default='stripes', choices=['stripes', 'dots', 'sunburst'])
    ap.add_argument('--stripe-width', type=int, default=None)
    ap.add_argument('--angle', type=int, default=40)
    ap.add_argument('--grain', type=int, default=14)
    ap.add_argument('--duotone', default=None)
    ap.add_argument('--margin', type=int, default=None)
    ap.add_argument('--aspect', default=None,
                    help='output aspect ratio W:H, e.g. 1:1, 2:1, 3:4, 9:16, 4:5. '
                         'The subject fills at least --coverage of the frame.')
    ap.add_argument('--coverage', type=float, default=0.8,
                    help='min fraction of the limiting dimension the subject '
                         'occupies when --aspect is set (default 0.8)')
    ap.add_argument('--save-cutout', default=None)
    args = ap.parse_args()

    if args.list_palettes:
        for name, (a, b) in PALETTES.items():
            print(f'{name:22s} #{a} #{b}')
        return

    colors = args.colors if args.colors else PALETTES[args.palette]

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        sys.exit(f'could not read {args.input}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resolution-proportional defaults: the reference look was tuned on a
    # ~740px image. On larger photos, fixed pixel sizes read too thin, so
    # scale any size the user didn't set explicitly.
    s = max(1.0, max(img_rgb.shape[:2]) / 800)
    border = args.border if args.border is not None else round(14 * s)
    stripe_w = args.stripe_width if args.stripe_width is not None else round(60 * s)
    margin = args.margin if args.margin is not None else round(80 * s)
    pad = round(40 * s)

    mask = segment_subject(img_rgb)
    treated = vintage_treatment(Image.fromarray(img_rgb), grain=args.grain,
                                duotone=hex_rgb(args.duotone) if args.duotone else None)
    sticker = build_sticker(treated, mask, border_px=border,
                            border_color=hex_rgb(args.border_color), pad=pad)
    # crop to actual content: layout sizing must see the subject, not the
    # original photo's full (mostly transparent) canvas
    bbox = sticker.getbbox()
    if bbox:
        sticker = sticker.crop(bbox)
    if args.save_cutout:
        sticker.save(args.save_cutout)

    aspect = None
    if args.aspect:
        try:
            aw, ah = args.aspect.split(':')
            aspect = float(aw) / float(ah)
        except (ValueError, ZeroDivisionError):
            sys.exit(f'bad --aspect {args.aspect!r}; use W:H like 1:1, 3:4, 9:16')

    flush = {} if args.no_bleed else detect_flush_sides(mask)
    bg_size, pos = layout(sticker.size, margin=margin, pad=pad, flush=flush,
                          aspect=aspect, coverage=args.coverage)
    bg = make_background(bg_size, pattern=args.pattern,
                         colors=(hex_rgb(colors[0]), hex_rgb(colors[1])),
                         stripe_width=stripe_w, angle=args.angle)
    final = composite(bg, sticker, pos=pos)
    final.save(args.output, quality=92)
    print(f'saved {args.output} ({final.size[0]}x{final.size[1]})')


if __name__ == '__main__':
    main()
