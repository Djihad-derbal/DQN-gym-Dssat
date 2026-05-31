"""Crop the right-side LLM panel from screenshots 02, 03, 05."""
from pathlib import Path
from PIL import Image

SRC = Path("screenshots")
OUT = SRC  # write cropped files alongside originals

# Right-column LLM panel bounds on a 1440x900 capture.
# Includes the tab bar (State / Decision / Advisor) at the top
# and the chat input row at the bottom.
LEFT, TOP, RIGHT, BOTTOM = 1135, 50, 1425, 690

for name in ["02_state_explainer.png",
             "03_decision_narrator.png",
             "05_advisor_chat.png"]:
    im = Image.open(SRC / name)
    cropped = im.crop((LEFT, TOP, RIGHT, BOTTOM))
    out_name = name.replace(".png", "_cropped.png")
    cropped.save(OUT / out_name)
    print(f"wrote {out_name}  ({cropped.size[0]}x{cropped.size[1]})")
