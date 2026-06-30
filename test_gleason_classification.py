import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset.gleason_cls import build_gleason_classification_loaders
from engine.gleason_classification import (
    binary_scores_for_grade_threshold,
    decode_predictions,
    evaluate_with_scores,
    main_logits,
    select_binary_thresholds,
    threshold_metrics_for_scores,
    unpack_batch,
)
from main_gleason_classification import get_args_parser
from models.build_gleason_classification import build_gleason_model
from util.gleason_config import resolve_adversarial_config
from util.misc import load_trusted_checkpoint


def resolve_experiment_dirs(args):
    args.crop_spatial_size = tuple(args.crop_spatial_size)
    resolve_adversarial_config(args)
    if args.task_type == "binary":
        definition_name = (
            args.binary_label_col
            if args.binary_label_col
            else f"{args.label_col}_ge_{args.binary_positive_min}"
        )
    else:
        definition_name = f"{args.label_col}_ordinal_{args.ordinal_levels}levels"

    adversarial_name = "ruleout_none"
    if args.adversarial_specs_resolved:
        adversarial_name = f"ruleout_{next(iter(args.adversarial_specs_resolved))}"

    args.log_dir = os.path.join(
        args.log_dir,
        "gleason",
        args.task_type,
        definition_name,
        adversarial_name,
        args.model,
        args.train,
        str(args.seed),
    )
    args.output_dir = os.path.join(
        args.output_dir,
        "gleason",
        args.task_type,
        definition_name,
        adversarial_name,
        args.model,
        args.train,
        str(args.seed),
    )
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        "Test Gleason classification checkpoint",
        parents=[get_args_parser()],
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Checkpoint path. Defaults to <resolved output_dir>/best.pth.tar.",
    )
    parser.add_argument(
        "--metrics_output",
        default=None,
        type=str,
        help="Metrics JSON path. Defaults to <resolved output_dir>/test_metrics.json.",
    )
    parser.add_argument(
        "--drop_adversarial_head",
        action="store_true",
        help=(
            "Evaluate an adversarially trained checkpoint as an MRI-only Gleason "
            "model by removing adversarial-head weights before loading."
        ),
    )
    parser.add_argument(
        "--predictions_output",
        default=None,
        type=str,
        help=(
            "Per-scan test prediction CSV path. Defaults to the metrics output "
            "path with '_predictions.csv' appended to the stem."
        ),
    )
    return parser.parse_args()


def clear_adversarial_model_config(args):
    args.adversarial_specs = ""
    args.adversarial_variable = None
    args.adversarial_column = None
    args.adversarial_observed_column = None
    args.adversarial_num_classes = None
    args.adversarial_specs_resolved = {}
    args.adversarial_columns = {}
    args.adversarial_observed_columns = {}
    args.adversarial_definition = "none"
    return args


def state_dict_without_adversarial_heads(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("adversarial_heads.")
    }


def default_predictions_output(metrics_output):
    path = Path(metrics_output)
    return str(path.with_name(f"{path.stem}_predictions.csv"))


def prediction_metadata_columns(dataset):
    preferred = [
        "person_id",
        "new_id",
        "pseudo_study_uid",
        "procedure_date",
        "image_npy_path",
        "grade_group",
        "gleason_binary_cs",
        "csPCa",
        "psa_value",
        "psa_value_observed",
        "psa_group",
        "psa_group_code",
        "age",
        "age_observed",
        "age_group",
        "age_group_code",
        "bmi",
        "bmi_observed",
        "bmi_group",
        "bmi_group_code",
        "cardio_any",
        "cardio_any_observed",
        "diabetes",
        "diabetes_observed",
        "renal_metabolic_any",
        "renal_metabolic_any_observed",
    ]
    df = getattr(dataset, "df", None)
    if df is None:
        return []
    columns = df.columns
    return [column for column in preferred if column in columns]


def clean_metadata_value(value):
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return value


def collect_prediction_rows(model, data_loader, device, args, thresholds=None):
    model.eval()
    rows = []
    dataset = data_loader.dataset
    metadata_columns = prediction_metadata_columns(dataset)

    with torch.no_grad():
        for batch in data_loader:
            img, labels, _, _, dataidx = unpack_batch(batch)
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = main_logits(model(img)).detach().cpu()
            batch_labels = labels.detach().cpu().numpy().astype(int)
            batch_indices = dataidx.detach().cpu().numpy().astype(int)
            preds, scores = decode_predictions(logits, args)

            for row_position, sample_index in enumerate(batch_indices):
                source_row = dataset.df.iloc[int(sample_index)]
                output_row = {
                    "split": getattr(dataset, "phase", "test"),
                    "row_index": int(sample_index),
                    "label": int(batch_labels[row_position]),
                    "pred_default_0_5": int(preds[row_position]),
                }
                for column in metadata_columns:
                    output_row[column] = clean_metadata_value(source_row[column])

                if args.task_type == "binary":
                    score = float(scores[row_position])
                    output_row["score"] = score
                    output_row["binary_label"] = int(batch_labels[row_position])
                    output_row["binary_score"] = score
                else:
                    for score_index, score in enumerate(scores[row_position]):
                        output_row[f"score_{score_index}"] = float(score)
                    binary_labels, binary_scores = binary_scores_for_grade_threshold(
                        np.asarray([batch_labels[row_position]]),
                        np.asarray([scores[row_position]]),
                        args,
                    )
                    output_row["binary_label"] = int(binary_labels[0])
                    if binary_scores is not None:
                        output_row["binary_score"] = float(binary_scores[0])

                if thresholds and "binary_score" in output_row:
                    binary_score = float(output_row["binary_score"])
                    for name, threshold in thresholds.items():
                        output_row[f"pred_threshold_{name}"] = int(
                            binary_score >= float(threshold)
                        )
                        output_row[f"threshold_{name}"] = float(threshold)

                rows.append(output_row)
    return rows


def write_prediction_rows(rows, output_path):
    if not rows:
        raise ValueError("No prediction rows to write")

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    args = resolve_experiment_dirs(args)
    checkpoint_adversarial_definition = args.adversarial_definition
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best.pth.tar")
    metrics_output = args.metrics_output or os.path.join(
        args.output_dir, "test_metrics.json"
    )
    predictions_output = args.predictions_output or default_predictions_output(
        metrics_output
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if args.drop_adversarial_head:
        args = clear_adversarial_model_config(args)

    _, data_loader_val, data_loader_test = build_gleason_classification_loaders(args)
    print("{}".format(args).replace(", ", ",\n"))
    print(f"label definition: {args.label_definition}")
    print(f"checkpoint adversarial definition: {checkpoint_adversarial_definition}")
    print(f"evaluation adversarial definition: {args.adversarial_definition}")
    print(f"test batches: {len(data_loader_test)}")
    print(f"checkpoint: {checkpoint_path}")

    model, _ = build_gleason_model(args=args, device=device)
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]
    if args.drop_adversarial_head:
        state_dict = state_dict_without_adversarial_heads(state_dict)
    model.load_state_dict(state_dict)

    val_stats, val_labels, val_scores = evaluate_with_scores(
        model=model,
        data_loader=data_loader_val,
        device=device,
        args=args,
        phase="val_threshold_source",
    )
    test_stats, test_labels, test_scores = evaluate_with_scores(
        model=model,
        data_loader=data_loader_test,
        device=device,
        args=args,
        phase="test",
    )
    val_binary_labels, val_binary_scores = binary_scores_for_grade_threshold(
        val_labels,
        val_scores,
        args,
    )
    test_binary_labels, test_binary_scores = binary_scores_for_grade_threshold(
        test_labels,
        test_scores,
        args,
    )
    threshold_output = {}
    if val_binary_scores is not None and test_binary_scores is not None:
        thresholds = select_binary_thresholds(val_binary_labels, val_binary_scores)
        threshold_output["thresholds_selected_on_val"] = thresholds
        threshold_output.update(
            {
                f"val_threshold_{key}": value
                for key, value in threshold_metrics_for_scores(
                    val_binary_labels,
                    val_binary_scores,
                    thresholds,
                ).items()
            }
        )
        threshold_output.update(
            {
                f"test_threshold_{key}": value
                for key, value in threshold_metrics_for_scores(
                    test_binary_labels,
                    test_binary_scores,
                    thresholds,
                ).items()
            }
        )
    output = {
        "checkpoint": checkpoint_path,
        "label_definition": args.label_definition,
        "checkpoint_adversarial_definition": checkpoint_adversarial_definition,
        "evaluation_adversarial_definition": args.adversarial_definition,
        "drop_adversarial_head": bool(args.drop_adversarial_head),
        **{f"val_{key}": value for key, value in val_stats.items()},
        **{f"test_{key}": value for key, value in test_stats.items()},
        **threshold_output,
    }

    prediction_rows = collect_prediction_rows(
        model=model,
        data_loader=data_loader_test,
        device=device,
        args=args,
        thresholds=threshold_output.get("thresholds_selected_on_val"),
    )
    write_prediction_rows(prediction_rows, predictions_output)

    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w", encoding="utf-8") as handle:
        json.dump(output, handle)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote test metrics: {metrics_output}")
    print(f"Wrote test predictions: {predictions_output}")


if __name__ == "__main__":
    main(parse_args())
