# Voice Arena Entity Accented English Benchmark

This is the scorer for the Voice Arena Entity Accented English Benchmark on Hugging Face. Scoring uses a static map of verified entity renderings (`data/entity_map_strict.json` and `data/entity_map_relaxed.json`) together with the Hugging Face open_asr_leaderboard normaliser. The maps were built by LLM verification of ASR model predictions.

```bash
pip install pandas jiwer rapidfuzz regex
python score.py --benchmark benchmark.csv --preds my_preds.csv
```

`my_preds.csv` needs an `id` column (e.g. `S0013_us`) and one transcript column per model.

The maps will be updated regularly, and contributions are welcome.
