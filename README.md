# Real-Time-Edge-Detection
Equalisation not helping

What is Histogram Equalization?
It’s a technique to improve the contrast of an image.
It spreads out the most frequent pixel intensities, making dark areas brighter and bright areas darker — resulting in more detail in both.
But traditional histogram equalization works on the entire image, and that can lead to:

Over-amplification of noise
Unrealistic lighting
⚡️ What is CLAHE?
CLAHE is a smarter version that:

Divides the image into small tiles (in our case, 8×8 regions),
Applies histogram equalization independently to each tile,
Then blends the tiles together using bilinear interpolation (so you don't see tile borders),
And limits contrast amplification using clipLimit to avoid over-enhancing noise.
🔧 Parameters Explained:
clipLimit=2.0

Controls how much contrast enhancement happens.
Higher values = more contrast, but may also bring out noise.
Typical range: 1.0 to 4.0
2.0 is a safe, balanced value.
tileGridSize=(8, 8)

Image is divided into 8×8 tiles (or regions).
Smaller tiles → more localized contrast adjustment (good for uneven lighting).
Larger tiles → more global adjustment (like traditional histogram equalization).