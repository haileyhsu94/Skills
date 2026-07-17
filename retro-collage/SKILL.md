---
name: retro-collage
description: Apply a retro collage / vintage zine treatment to any photo — automatically extract the subject (person, pet, animal, food, product, or object) from the background, desaturate it with film grain, trace it with a cream sticker-style cutout border, and composite it onto a bold striped/dotted/sunburst retro background at any aspect ratio (1:1, 4:5, 9:16, etc.). Use this whenever the user wants a photo turned into a retro/vintage/mod/60s-70s collage style, a "sticker cutout" look, a magazine-clipping effect, background removal followed by stylized re-compositing, or mentions retro poster/zine/scrapbook aesthetics for an image. Also use it when a user uploads a selfie, pet photo, or product shot and asks to make it look like a vintage cutout or collage.
---

# Retro Collage Photo Treatment

Turn a photograph into a vintage zine collage: the subject is cut out of the
photo, treated like a grainy black-and-white magazine clipping, traced with a
cream "sticker" border, and pasted onto a bold retro-patterned background with
a soft natural shadow.

## Why the style works

The whole trick is contrast between real and flat: the subject stays a genuine
photograph (real texture, lighting, grain — never illustrated or smoothed),
while everything around it is flat and graphic. Keep that contrast intact when
customizing. Shadows should be soft and natural (blurred, low opacity) — hard
offset shadows read as brutalist UI, which is a different aesthetic.

## Quick start

Install dependencies (mediapipe MUST be 0.10.9 — see pitfalls below):

```bash
pip install "mediapipe==0.10.9" opencv-python pillow numpy --break-system-packages
```

Run the bundled script:

```bash
python scripts/retro_collage.py input.jpg output.jpg
```

That produces the reference look: chartreuse/teal diagonal stripes, cream
border, B&W grain. To also keep the transparent cutout (useful for placing the
sticker on a webpage or different background):

```bash
python scripts/retro_collage.py input.jpg output.jpg --save-cutout sticker.png
```

## Customization

All flags are optional; defaults reproduce the reference style.

```bash
python scripts/retro_collage.py in.jpg out.jpg \
  --palette grape-soda \          # named color set (see references/palettes.md)
  --colors d9a441 3a3934 \        # ...or two explicit hex colors (overrides palette)
  --pattern dots \                # stripes | dots | sunburst
  --border-color fff5e6 \         # sticker border color
  --border 20 \                   # border thickness px
  --stripe-width 80 --angle 30 \  # stripe geometry
  --grain 18 \                    # film grain strength (0 = off)
  --duotone 2b3a4a \              # tint shadows toward a color instead of pure B&W
  --aspect 4:5 \                  # output ratio; subject fills >= --coverage of frame
  --coverage 0.8 \                # min subject fraction when --aspect is set
  --margin 100 \                  # space around the sticker (no-aspect mode)
  --no-bleed                      # disable flush-edge bleed (see below)
```

22 named palettes ship with the script (`--list-palettes` to print them);
`references/palettes.md` describes each with mood guidance and how to invent
new pairs. Keep the border cream/off-white so the cutout reads as paper.

## Output aspect ratio & subject coverage

`--aspect W:H` (e.g. `1:1`, `2:1`, `3:4`, `4:5`, `9:16` — any ratio,
landscape or portrait) sets the exact output shape. The canvas is then sized
so the subject fills at least `--coverage` (default 0.8) of the limiting
dimension — the subject should dominate the frame, not float in a sea of
pattern. Without `--aspect`, the output hugs the sticker with a `--margin`
of background on non-flush sides.

Defaults (border, stripe width, margin) auto-scale with image resolution so
the look stays proportional from a 500px selfie to a 4000px portrait;
explicit flags override the scaling.

## Subjects beyond people (pets, food, products)

`segment_subject` runs a strategy chain automatically — no flag needed:

1. **mediapipe person model** — best edges on people; generalizes passably
   to pets and prominent foreground animals.
2. **color-key** — if the person model finds nothing sane and the photo's
   border is near-uniform (product shots, fruit on a white background),
   everything far from the border color becomes the subject.
3. **GrabCut** — generic fallback, seeded border=background /
   center=foreground, with an inversion guard.

A strategy's mask is trusted only if the subject it found covers a sane
fraction of the frame (5–90%). Busy backgrounds with non-person subjects
are the hardest case — expect rougher edges there, and lean on the thick
border to hide them.

## Flush-edge bleed (on by default)

When the subject is cropped by the original photo frame (an arm leaving the
shot, a torso filling the bottom), the script detects which frame edges the
subject touches and runs the sticker flush off the canvas on those sides —
the way real collages run cropped subjects off the page. A straight-cut edge
floating in the middle of the background looks wrong; only fully-contained
subjects should float with margin on all sides. Use `--no-bleed` to disable.

## Environment pitfalls (read before debugging)

- **Use `mediapipe==0.10.9`, not the latest.** Newer versions (0.10.3x) drop
  the legacy `mp.solutions` API and require downloading a model from Google
  Cloud Storage, which is blocked in sandboxed environments. 0.10.9 bundles
  the selfie-segmentation model inside the wheel — fully offline.
- **Don't use `rembg`** in sandboxed environments: it downloads its U²-Net
  model from GitHub releases at runtime, which is typically blocked (403).
  If you're in an environment with open network access and need higher-quality
  edges (wispy hair, fine fingers), rembg/u2net is a reasonable upgrade.
- **Stripes are drawn on an oversized square, then rotated and center-cropped.**
  Drawing angled polygons directly onto the canvas leaves uncovered corners.

## Quality expectations & limitations

- mediapipe's selfie segmentation is tuned for people. It works reasonably on
  pets and prominent foreground objects but edges are approximate — good for
  this style (the thick border hides imperfect edges), not for precision
  masking.
- Subjects cropped by the photo frame stay cropped: if an arm exits the frame,
  the cutout has a straight edge there. The flush-edge bleed (above) handles
  this gracefully by running those edges off the canvas. Best inputs have
  decent subject/background separation.
- Verify the result by viewing the output image (Read tool) before delivering.
  Check: subject cleanly extracted, border traces the silhouette (not a
  rectangle), background pattern covers the full canvas, shadow is soft.

## Adapting beyond the script

For requests the flags don't cover (multiple subjects on one background,
custom illustrated accents layered on top, animated versions), import the
script's functions rather than rewriting the pipeline:

```python
from retro_collage import segment_subject, vintage_treatment, build_sticker, make_background, composite
```

Each stage takes and returns PIL images / numpy masks, so stages can be
swapped independently (e.g., keep the sticker cutout but hand-build the
background in HTML/CSS for a website hero).

## Credits

Created with Claude. Built on open-source components:
MediaPipe (Google, Apache-2.0) for subject segmentation and face landmarks,
OpenCV, Pillow, and NumPy for image processing. The visual style draws on
1960s–70s mod/psychedelic print design and the contemporary "vintage sticker
collage" aesthetic popular in editorial and social design.
