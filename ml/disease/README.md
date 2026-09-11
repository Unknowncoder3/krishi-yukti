# Crop Disease Detection

This module will contain the computer-vision pipeline for crop/leaf disease detection.

Planned flow:

```text
Image -> preprocessing -> trained classifier -> class probabilities -> guidance response
```

The model API must return both the predicted class and confidence. A low-confidence result should be surfaced for human review instead of being presented as certain.

Training artifacts and large datasets must remain outside Git and be downloaded through a documented dataset pipeline.
