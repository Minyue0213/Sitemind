# External data manifest

SiteMind does not mirror the complete GOOSE / GOOSE-Ex datasets or raw ROS bags. This manifest records the exact external inputs retained during development so they can be downloaded again and verified.

## Official inputs

| Local file | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `data/raw/gooseEx_2d_val.zip` | [GOOSE-Ex 2D validation archive](https://goose-dataset.de/storage/gooseEx_2d_val.zip) | 1,461,287,553 | `39238c4213425da2c60d9e283a652fafcfd75d0e22943e81ea0a514aa8bfeeb1` |
| `data/raw/alice_example_bagfile.zip` | [Official ALICE reference bag archive](https://goose-dataset.de/storage/alice_example_bagfile.zip) | 12,902,823,631 | `253a1407acb81b4ebee55489938f9383952bbc9b21405b6754fda1bd60a35edd` |
| `data/raw/alice_scenario06_sequence02_sensors.bag` | Extracted from `alice_example_bagfile.zip` | 22,340,147,392 | `e3221ac2058fa18ca8682c79a427d88073dc8439b6ee05fcbe3bc53ad7cb7f82` |
| `models/ppliteseg_category_512.pth` | [Official PP-LiteSeg 12-category checkpoint](https://goose-dataset.de/models/ppliteseg_category_512.pth) | 98,180,357 | `87bf55a1ae994a4bdf3a4d30c8f0dec2916994b37d433f0b5b694d63157ee0ab` |
| `models/ppliteseg_class_512.pth` | [Official PP-LiteSeg 64-class checkpoint](https://goose-dataset.de/models/ppliteseg_class_512.pth) | 98,208,249 | `6dd412c0c99115e359896c4cab43a8e6bce9e09b843e7fa885fe597b0a6121cd` |

The extracted `data/processed/gooseEx_2d_val/` directory is derived entirely from the validation archive and does not need a separate backup.

## SiteMind checkpoints

The two project checkpoints are preserved in the [`v0.1.0` GitHub Release](https://github.com/Minyue0213/Sitemind/releases/tag/v0.1.0). Their hashes and intended use are documented in the main README.

## Verification

On macOS or Linux, verify a downloaded file with:

```bash
shasum -a 256 /path/to/file
```

GOOSE data and derived material remain subject to the upstream CC BY-SA 4.0 terms. This manifest does not transfer the SiteMind source-code MIT license to third-party data.
