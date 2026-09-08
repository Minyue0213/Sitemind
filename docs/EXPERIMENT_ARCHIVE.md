# Experiment archive

The generated outputs retained for the SiteMind `v0.1.x` run are published as a GitHub Release asset rather than committed to Git.

## Download

- Release: [`v0.1.1`](https://github.com/Minyue0213/Sitemind/releases/tag/v0.1.1)
- Asset: [`sitemind-experiment-artifacts-v0.1.1.tar.gz`](https://github.com/Minyue0213/Sitemind/releases/download/v0.1.1/sitemind-experiment-artifacts-v0.1.1.tar.gz)
- SHA-256: `c031d334ac91b79f29ceaa429bd2271b6abd6f421199bca2fb2947e44acd8051`

The archive contains the `artifacts/` tree with generated predictions, calibrated fusion layers, temporal sequence outputs, evaluation summaries, audit reports and final GIF/JSON demonstrations. The two 140 MiB SiteMind checkpoints are intentionally excluded because they are already published separately in `v0.1.0`.

## Restore

Run from the repository root:

```bash
curl -L https://github.com/Minyue0213/Sitemind/releases/download/v0.1.1/sitemind-experiment-artifacts-v0.1.1.tar.gz \
  -o sitemind-experiment-artifacts-v0.1.1.tar.gz
shasum -a 256 sitemind-experiment-artifacts-v0.1.1.tar.gz
tar -xzf sitemind-experiment-artifacts-v0.1.1.tar.gz
```

The archive includes selected camera frames and derived visualizations originating from GOOSE / GOOSE-Ex. Preserve upstream attribution and CC BY-SA 4.0 terms when redistributing it.
