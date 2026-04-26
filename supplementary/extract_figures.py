"""Crop the three figures from rendered pages for inclusion as PNGs."""
from PIL import Image
from pathlib import Path

REVIEW = Path(__file__).parent.parent / "review_pages"
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

# Figure 1 (pipeline) is on page 3, spans full width near top
p3 = Image.open(REVIEW / "page_03.png")
w, h = p3.size
fig1 = p3.crop((80, 100, w-80, 430))
fig1.save(OUT / "fig1_pipeline.png")

# Figure 2 (score trajectory) is on page 4 left column
p4 = Image.open(REVIEW / "page_04.png")
fig2 = p4.crop((70, 320, 620, 650))
fig2.save(OUT / "fig2_score_trajectory.png")

# Figure 3 (HPWL vs v_rel scatter) is on page 4 right column
fig3 = p4.crop((620, 320, 1200, 720))
fig3.save(OUT / "fig3_hpwl_vrel_scatter.png")

print("Extracted figures:")
for p in sorted(OUT.glob("fig*.png")):
    im = Image.open(p)
    print(f"  {p.name}: {im.size[0]}x{im.size[1]}")
