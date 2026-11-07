"""Train the CNN digit reader and the logistic-regression baseline, calibrate, evaluate, build pools.

Steps (seeded with --seed, default config.SEED):
1. Train the CNN on the train split (digits 1-9); validation accuracy printed per epoch.
2. Fit a temperature T for each model on the VALIDATION split (temperature scaling).
3. Train multinomial logistic regression on raw pixels (seeded subset of the train split).
4. Evaluate on the untouched MNIST test split (digits 1-9): accuracy, NLL, ECE (15 bins), before and
   after temperature scaling -> results/nn/metrics.csv, confusion matrices, reliability diagrams.
5. Save models to models/ (cnn.pt, cnn_meta.json, logreg.joblib).
6. Run the reader (CNN + T) on the MNIST test split and save its probabilities with the hard /
   misread / clean pool masks to data/processed/digit_pools.npz (see DATA_DICTIONARY.md section 5).

Usage:  python train.py [--seed 7] [--quick]
--quick trains on a small subset for 1 epoch and writes to models/quick, results/nn/quick and
data/processed/digit_pools_quick.npz, so it never overwrites the real models.
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from sudoku import config
from sudoku.data import injection, mnist
from sudoku.nn import baseline, calibration, report
from sudoku.nn.model import predict_logits
from sudoku.nn.reader import DigitReader, save_cnn
from sudoku.nn.training import train_cnn


def evaluate(name: str, logits, labels, temperature: float) -> dict:
    """Accuracy, NLL and ECE of one model at one temperature on one split."""
    probs = calibration.softmax(logits, temperature)
    return {
        "model": name,
        "temperature": round(temperature, 4),
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "nll": calibration.negative_log_likelihood(logits, labels, temperature),
        "ece": calibration.expected_calibration_error(probs, labels),
    }


def pool_size_table(probs, digits) -> pd.DataFrame:
    """Images per digit in the hard, misread and clean pools."""
    masks = {"hard": injection.hard_mask(probs, digits),
             "misread": injection.misread_mask(probs, digits),
             "clean": injection.clean_mask(probs, digits)}
    rows = [{"digit": str(d), **{k: int((m & (digits == d)).sum()) for k, m in masks.items()}}
            for d in config.DIGITS]
    rows.append({"digit": "all", **{k: int(m.sum()) for k, m in masks.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    args = parser.parse_args()
    config.set_seed(args.seed)

    model_dir = config.MODELS / "quick" if args.quick else config.MODELS
    out_dir = config.NN_RESULTS / "quick" if args.quick else config.NN_RESULTS
    pools_file = config.POOLS_QUICK_FILE if args.quick else config.POOLS_FILE
    epochs = config.QUICK_EPOCHS if args.quick else config.CNN_EPOCHS
    lr_subset = config.QUICK_LOGREG_SUBSET if args.quick else config.LOGREG_TRAIN_SUBSET
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = mnist.load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    if args.quick:
        keep = config.make_rng(args.seed).permutation(len(train))[: config.QUICK_TRAIN_SUBSET]
        train = mnist.DigitSplit(train.images[keep], train.digits[keep], train.source_idx[keep])
    print(f"train {len(train)}  val {len(val)}  test {len(test)} images (digits 1-9)")

    print(f"[1/6] CNN: {epochs} epoch(s)")
    start = time.perf_counter()
    cnn, history = train_cnn(train.images, train.classes, val.images, val.classes, epochs, args.seed)
    cnn_seconds = time.perf_counter() - start

    print("[2/6] temperature scaling on validation")
    cnn_val, cnn_test = predict_logits(cnn, val.images), predict_logits(cnn, test.images)
    t_cnn = calibration.fit_temperature(cnn_val, val.classes)
    print(f"  CNN T = {t_cnn:.3f}")

    print(f"[3/6] logistic regression on {min(lr_subset, len(train))} raw-pixel images")
    start = time.perf_counter()
    logreg = baseline.train_logreg(train.images, train.classes, lr_subset, args.seed)
    lr_seconds = time.perf_counter() - start
    lr_val, lr_test = baseline.logreg_logits(logreg, val.images), baseline.logreg_logits(logreg, test.images)
    t_lr = calibration.fit_temperature(lr_val, val.classes)
    print(f"  logreg T = {t_lr:.3f}  ({lr_seconds:.1f}s)")

    print("[4/6] test-set evaluation (MNIST test, digits 1-9)")
    y = test.classes
    metrics = pd.DataFrame([
        evaluate("cnn", cnn_test, y, 1.0),
        evaluate("cnn+temperature", cnn_test, y, t_cnn),
        evaluate("logreg", lr_test, y, 1.0),
        evaluate("logreg+temperature", lr_test, y, t_lr),
    ])
    metrics["train_seconds"] = [cnn_seconds, cnn_seconds, lr_seconds, lr_seconds]
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pd.DataFrame(history).to_csv(out_dir / "cnn_history.csv", index=False)
    probs = {name: calibration.softmax(z, t) for name, z, t in [
        ("CNN", cnn_test, 1.0), ("CNN + temperature", cnn_test, t_cnn),
        ("LogReg", lr_test, 1.0), ("LogReg + temperature", lr_test, t_lr)]}
    report.plot_reliability(probs, y, out_dir / "reliability.png")
    for name, key in (("cnn", "CNN"), ("logreg", "LogReg")):
        matrix = report.digit_confusion(probs[key], y)
        pd.DataFrame(matrix, index=config.DIGITS, columns=config.DIGITS).to_csv(
            out_dir / f"confusion_{name}.csv")
        acc = metrics.loc[metrics.model == name, "accuracy"].item()
        report.plot_confusion(matrix, f"{key} on MNIST test (acc {acc:.4f})",
                              out_dir / f"confusion_{name}.png")

    print(f"[5/6] saving models to {model_dir.relative_to(config.ROOT)}")
    save_cnn(cnn, {"temperature": t_cnn, "seed": args.seed, "epochs": epochs,
                   "train_size": len(train), "history": history,
                   "test_accuracy": metrics.accuracy[0]}, model_dir)
    joblib.dump({"model": logreg, "temperature": t_lr}, model_dir / config.LOGREG_FILE)

    print("[6/6] hard-digit pools from the reader's test probabilities")
    reader = DigitReader(cnn, t_cnn if config.USE_TEMPERATURE else 1.0)
    reader_probs = reader.predict_proba(test.images)
    injection.save_pool_probs(pools_file, reader_probs, test.digits, reader.temperature)
    sizes = pool_size_table(reader_probs, test.digits)
    sizes.to_csv(out_dir / "pool_sizes.csv", index=False)
    print(sizes.to_string(index=False))
    empty = [d for d in config.DIGITS if not sizes.loc[sizes.digit == str(d), "misread"].item()]
    if empty:
        print(f"  WARNING: no misread image for digit(s) {empty}; they are never injected")
    print(f"  saved {pools_file.relative_to(config.ROOT)} and {out_dir.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
