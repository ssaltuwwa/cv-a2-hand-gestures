## Results (test set, 97 real images)
|                               |   Accuracy |   Precision |   Recall |   Macro-F1 |
|:------------------------------|-----------:|------------:|---------:|-----------:|
| Baseline CNN (A1)             |      0.557 |       0.53  |    0.52  |      0.493 |
| MobileNetV2 real (A1)         |      0.897 |       0.907 |    0.88  |      0.89  |
| MobileNetV2 + synthetic (A2)  |      0.856 |       0.877 |    0.845 |      0.856 |
| Ablation: + synthetic, no aug |      0.918 |       0.925 |    0.914 |      0.918 |

## Accuracy under simulated conditions
|                               |   clean |   dark |   overexposed |   motion_blur |   noise |   low_res |   rotated |   occlusion |
|:------------------------------|--------:|-------:|--------------:|--------------:|--------:|----------:|----------:|------------:|
| Baseline CNN (A1)             |   0.557 |  0.536 |         0.536 |         0.608 |   0.351 |     0.608 |     0.588 |       0.567 |
| MobileNetV2 real (A1)         |   0.897 |  0.907 |         0.887 |         0.742 |   0.814 |     0.485 |     0.835 |       0.722 |
| MobileNetV2 + synthetic (A2)  |   0.856 |  0.876 |         0.742 |         0.784 |   0.773 |     0.557 |     0.773 |       0.711 |
| Ablation: + synthetic, no aug |   0.918 |  0.907 |         0.845 |         0.701 |   0.753 |     0.567 |     0.784 |       0.794 |

## Held-out synthetic clips (per-frame accuracy)
| env          |   MobileNetV2 real (A1) |   MobileNetV2 + synthetic (A2) |   frames |
|:-------------|------------------------:|-------------------------------:|---------:|
| classroom    |                   0.125 |                          0     |        8 |
| dark_room    |                   1     |                          1     |        8 |
| kitchen      |                   0.875 |                          1     |       16 |
| night_neon   |                   0.938 |                          0.938 |       16 |
| plain_wall   |                   0.875 |                          1     |        8 |
| sunny_street |                   0     |                          0.625 |       16 |
| ALL          |                   0.625 |                          0.792 |       72 |

## Latency
| Model                        | Device   |   Mean ms |   p95 ms |   FPS |   Params |
|:-----------------------------|:---------|----------:|---------:|------:|---------:|
| Baseline CNN (A1)            | CPU      |      35.2 |     42.2 |  28.4 |   110147 |
| Baseline CNN (A1)            | GPU      |      15   |     16.2 |  66.9 |   110147 |
| MobileNetV2 + synthetic (A2) | CPU      |     282.9 |    354   |   3.5 |  2261827 |
| MobileNetV2 + synthetic (A2) | GPU      |     173.2 |    226.6 |   5.8 |  2261827 |

Grad-CAM: 301.7 ms/image

## Dataset
|            |   Real train |   Real val |   Real test |   Synthetic train |   Synthetic stress |
|:-----------|-------------:|-----------:|------------:|------------------:|-------------------:|
| peace-sign |          134 |         28 |          29 |                72 |                 24 |
| thumbs-up  |          199 |         43 |          43 |                72 |                 24 |
| stop       |          117 |         26 |          25 |                72 |                 24 |
| Total      |          450 |         97 |          97 |               216 |                 72 |

Synthetic share of training set: 32.4%