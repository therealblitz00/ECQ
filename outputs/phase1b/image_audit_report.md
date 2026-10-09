# Phase 1b: image audit

- Items with an image: 9,457
- Masks: 9,096 by white-background threshold, 272 by the detector (OWLv2)
- Embeddings: `data/embeddings/image_clip.npy` (openai/clip-vit-base-patch32, 512 numbers per image)

| Flag | Items |
|---|---|
| `IMG_NOT_PACKSHOT` | 272 |
| `VIS_TYPE_DOUBT` | 189 |
| `VIS_COLOUR_MISMATCH` | 132 |
| `IMG_NEAR_DUPLICATE` | 54 |
| `VIS_TARGET_NOT_FOUND` | 46 |
| `VIS_TYPE_MISMATCH` | 24 |

- VIS_TYPE_MISMATCH: contact sheet `outputs/phase1b/vis_type_mismatch.jpg` (first 60)
- VIS_TYPE_DOUBT: contact sheet `outputs/phase1b/vis_type_doubt.jpg` (first 60)
- VIS_COLOUR_MISMATCH: contact sheet `outputs/phase1b/vis_colour_mismatch.jpg` (first 60)
- VIS_TARGET_NOT_FOUND: contact sheet `outputs/phase1b/vis_target_not_found.jpg` (first 60)
- IMG_NEAR_DUPLICATE: contact sheet `outputs/phase1b/img_near_duplicate.jpg` (first 60)
- IMG_NOT_PACKSHOT: contact sheet `outputs/phase1b/img_not_packshot.jpg` (first 60)
