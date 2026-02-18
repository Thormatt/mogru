# CWRU Bearing Dataset

The CWRU Bearing Data Center dataset is required for `bearing_benchmark.py`.

## Download

1. Visit the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
2. Download the 12kHz drive-end (DE) accelerometer `.mat` files for:
   - Normal baseline (0 HP load)
   - Inner race faults: 7, 14, 21 mil diameter
   - Ball faults: 7, 14, 21 mil diameter
   - Outer race faults: 7, 14, 21 mil diameter (centered @6)
3. Place the `.mat` files in this directory
4. Run the bearing benchmark preprocessing, which converts `.mat` to `.npz`:

```bash
python -m mogru.bearing_benchmark
```

The script will automatically detect and convert the raw data files.

## File Format

After preprocessing, each `.npz` file contains:
- Vibration waveform arrays sampled at 12 kHz
- File naming: `{rpm}_{fault_type}.npz` (e.g., `1797_Normal.npz`, `1797_B007.npz`)

## Citation

```
K.A. Loparo, "Case Western Reserve University Bearing Data Center"
https://engineering.case.edu/bearingdatacenter
```
