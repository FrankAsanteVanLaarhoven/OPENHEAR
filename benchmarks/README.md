# Feasibility data directory

Do **not** include private participant audio, personal speech transcripts or model weights in a public repository.

Create 20 empty trial rows per direction:

```sh
python -m openhear.cli benchmark init --dataset benchmarks/utterances.csv
```

`utterances.csv` is **not** shipped with generated reference texts or synthetic evidence. Point `audio_path` at a consented WAV relative to the CSV location, and fill `reference_transcript`, `reference_translation`, `speaker_id`, `accent` and `condition`.

Execute the local, model-backed benchmark after the records are complete:

```sh
python -m openhear.cli benchmark run --dataset benchmarks/utterances.csv \
  --models-dir models --report benchmark-001.json
```

Use a new report filename for each run. Output includes sensitive source and translated text. A 20-per-direction feasibility set is not a confirmatory sample or proof of SOTA.
