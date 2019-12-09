import matplotlib
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from multiprocessing import Pool

# matplotlib: use agg backend
matplotlib.use("agg")


def load_synonyms_dict(synonyms_dict_fn):
    print("INFO: loading synonyms dictionary")
    with open(synonyms_dict_fn, "r") as f:
        synonyms_dict = eval(f.readlines()[0])

    # filter dict to contain only
    # distinct pairs (no relations with itself)
    # and confidences over 0.0
    synonyms_dict = {
            k: v for k, v in synonyms_dict.items()
            if len(k) == 2 and v > 0.0
            }
    return synonyms_dict


def load_gold_synonyms_uri(synonyms_uri_fn):
    print("INFO: loading gold synonyms")
    with open(synonyms_uri_fn, "r") as f:
        lines = f.readlines()[1:]
        lines = list(map(lambda x: x.split(), lines))
        synonyms_gold = set(map(lambda x: frozenset(x), lines))
    return synonyms_gold


def load_relation_mapping(relation2id_fn):
    print("INFO: loading relation mapping")
    with open(relation2id_fn, "r") as f:
        lines = f.readlines()[1:]
        lines = list(map(lambda x: x.split(), lines))
        relation2id = {
                relation_uri: int(relation_id)
                for relation_uri, relation_id in lines
                }
        relation2uri = {
                int(relation_id): relation_uri
                for relation_uri, relation_id in lines
                }
    return relation2id, relation2uri


def build_similarity_matrix(synonyms_dict, relation2id):
    similarity_matrix = np.eye(len(relation2id), dtype=float)
    for synonym in synonyms_dict:
        it = iter(synonym)
        r1 = relation2id[next(it)]
        r2 = relation2id[next(it)]
        similarity_matrix[r1][r2] = synonyms_dict[synonym]
        similarity_matrix[r2][r1] = synonyms_dict[synonym]
    return similarity_matrix


def zscore(observed_value, similarity_mean, similarity_std):
    return (
            ((observed_value - similarity_mean) / similarity_std)
            if abs(similarity_std) > 0.0 else 0.0
            )


def zscore_threshold_chebyshevs_ineq(max_population_percent):
    return np.sqrt(1 / (max_population_percent / 2))


def classify_synonyms_simple(
        synonyms_dict,
        min_threshold
        ):
    return set(filter(
        lambda x: synonyms_dict[x] >= min_threshold, synonyms_dict
        ))


def classify_synonyms_outlier(
        similarity_matrix,
        min_threshold,
        relation2uri
        ):
    min_threshold_zscore = zscore_threshold_chebyshevs_ineq(min_threshold)
    synonyms = set()
    for r1_idx, r2_similarities in enumerate(similarity_matrix):
        r2_similarities_mean = r2_similarities.mean()
        r2_similarities_std = r2_similarities.std()
        r2_similarities_zscore = zscore(
                r2_similarities,
                r2_similarities_mean,
                r2_similarities_std
                )
        r2_synonyms = list(
                filter(
                    lambda r2_idx: (
                        r2_similarities_zscore[r2_idx] >= min_threshold_zscore
                        ),
                    range(0, len(r2_similarities_zscore))
                    )
                )
        for r2_idx in r2_synonyms:
            if r1_idx != r2_idx:
                r1_uri = relation2uri[r1_idx]
                r2_uri = relation2uri[r2_idx]
                synonyms.add(frozenset([r1_uri, r2_uri]))
    return synonyms


def precision_recall(synonyms_gold, classified_synonyms):
    true_positives = set.intersection(synonyms_gold, classified_synonyms)
    precision = (
            len(true_positives) / len(classified_synonyms)
            if len(classified_synonyms) > 0.0 else 1.0
            )
    recall = (
            len(true_positives) / len(synonyms_gold)
            if len(synonyms_gold) > 0.0 else 1.0
            )
    return precision, recall


def precision_topk(synonyms_gold, classified_synonyms):
    true_positives = set.intersection(synonyms_gold, classified_synonyms)
    precision = (
            len(true_positives) / len(classified_synonyms)
            if len(classified_synonyms) > 0.0 else 1.0
            )
    topk = len(classified_synonyms)
    return precision, topk


def evaluate_simple(
        synonyms_gold,
        synonyms_dict,
        threshold_range,
        precision_func,
        relation2id=None,
        relation2uri=None
        ):
    precision_x_values = {}
    for threshold in threshold_range:
        classified_synonyms = classify_synonyms_simple(
                synonyms_dict,
                threshold
                )
        precision_x_values[threshold] = precision_func(
                synonyms_gold,
                classified_synonyms
                )
    return precision_x_values


def evaluate_outlier(
        synonyms_gold,
        synonyms_dict,
        threshold_range,
        precision_func,
        relation2id,
        relation2uri
        ):
    similarity_matrix = build_similarity_matrix(synonyms_dict, relation2id)
    precision_x_values = {}
    for threshold in threshold_range:
        classified_synonyms = classify_synonyms_outlier(
                similarity_matrix,
                threshold,
                relation2uri
                )
        precision_x_values[threshold] = precision_func(
                synonyms_gold,
                classified_synonyms
                )
    return precision_x_values


def evaluate_affinity(
        synonyms_gold,
        synonyms_dict,
        threshold_range,
        precision_func,
        relation2id,
        relation2uri
        ):
    raise NotImplementedError


def evaluate_spectral(
        synonyms_gold,
        synonyms_dict,
        threshold_range,
        precision_func,
        relation2id,
        relation2uri
        ):
    raise NotImplementedError


def evaluate_dbscan(
        synonyms_gold,
        synonyms_dict,
        threshold_range,
        precision_func,
        relation2id,
        relation2uri
        ):
    raise NotImplementedError


def plot_prepare(topk=None, title=""):
    print("INFO: prepare plotting")
    plt.clf()
    plt.xlim((0.0, 1.0) if not topk else (0, topk))
    plt.ylim((0.0, 1.0))
    plt.yticks(list(map(lambda x: x / 10.0, range(0, 11))))
    if topk:
        plt.xticks(range(0, topk + 1, 100))
    else:
        plt.xticks(list(map(lambda x: x / 10.0, range(0, 11))))
    plt.xlabel("RECALL" if not topk else "TOP K")
    plt.ylabel("PRECISION")
    plt.title(title)
    plt.grid()


def plot_finish(plot_fn):
    print("INFO: finish plotting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.savefig(plot_fn, bbox_inches="tight")


def plot_precision(precision_x_values, topk=None, title=""):
    print("INFO: plotting precision values")
    precision_values = list(map(lambda x: x[0], precision_x_values))
    x_values = list(map(lambda x: x[1], precision_x_values))
    plt.plot(x_values, precision_values, label="RuleAlign", color="r")


def plot_baseline(baseline_fn, label, color):
    print("INFO: plotting baseline \"{0}\"".format(label))
    if os.path.isfile(baseline_fn):
        with open(baseline_fn, "r") as f:
            lines = f.readlines()
            values = list(map(lambda x: x.split(), lines))
        y_values = list(map(lambda x: float(x[0]), values))
        x_values = list(map(lambda x: float(x[1]), values))
        plt.plot(x_values, y_values, label=label, color=color)
    else:
        print("WARNING: baseline file \"{0}\" not found".format(baseline_fn))


CLASSIFIER_CHOICES = {
        "simple": evaluate_simple,
        "outlier": evaluate_outlier,
        "affinity": evaluate_affinity,
        "spectral": evaluate_spectral,
        "dbscan": evaluate_dbscan
        }


PRECISION_FUNC_CHOICES = {
        "precision_recall": precision_recall,
        "precision_topk": precision_topk
        }


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
            "-r", "--relation2id",
            required=True, type=str,
            help=(
                "The relation2id.txt mapping file needed "
                "for some classifier."
                )
            )
    parser.add_argument(
            "-g", "--gold-synonyms-uri",
            required=True, type=str,
            help="The gold standard synonyms list (URIs)."
            )
    parser.add_argument(
            "-s", "--synonyms-dict",
            required=True, type=str,
            help="The synonyms dictionary to evaluate."
            )
    parser.add_argument(
            "-c", "--classifier", type=str,
            choices=CLASSIFIER_CHOICES.keys(), default="simple",
            help="The classifier to use. (Default: simple)."
            )
    parser.add_argument(
            "-f", "--precision-func", type=str,
            choices=PRECISION_FUNC_CHOICES.keys(), default="precision_recall",
            help="The precision function to use. (Default: precision_recall)."
            )
    parser.add_argument(
            "-b", "--baseline-dir", type=str,
            help="The opt. baseline dir containing all baseline.txt files."
            )
    parser.add_argument(
            "-p", "--processes", type=int, default=4,
            help=(
                "The number of processes to use "
                "for classification. (Default: 4)."
                )
            )
    args = parser.parse_args()

    # check filenames
    if not os.path.isfile(args.relation2id):
        sys.exit(
                "ERROR: specified relation2id.txt mapping "
                "file does not exist"
                )
    if not os.path.isfile(args.gold_synonyms_uri):
        sys.exit("ERROR: specified gold synonyms list file does not exist")
    if not os.path.isfile(args.synonyms_dict):
        sys.exit(
                "ERROR: specified synonyms dictionary "
                "file to evaluate does not exist"
                )
    plot_fn = os.path.join(os.path.dirname(args.synonyms_dict), "{0}_{1}.pdf")

    # load relation2id mapping
    relation2id, relation2uri = load_relation_mapping(args.relation2id)

    # load synonyms
    gold_synonyms_uri = load_gold_synonyms_uri(
            args.gold_synonyms_uri
            )
    synonyms_dict = load_synonyms_dict(
            args.synonyms_dict
            )

    # determine precision-recall or precision-topk for plotting
    if args.precision_func == "precision_recall":
        topk = None
    if args.precision_func == "precision_topk":
        topk = len(gold_synonyms_uri)

    # defining threshold range
    threshold_range = list(map(lambda x: x / 1000.0, range(1, 1001)))

    # evaluate
    print(
            "INFO: evaluating synonyms using "
            "classifier \"{0}\"".format(args.classifier)
            )

    # multiprocessing
    pool = Pool(processes=args.processes)
    results = []
    for process in range(0, args.processes):
        threshold_range_p = [
                threshold_range[i]
                for i in range(
                    process,
                    len(threshold_range),
                    args.processes
                    )
                ]
        results.append(
                pool.apply_async(
                    CLASSIFIER_CHOICES[args.classifier],
                    [
                        gold_synonyms_uri,
                        synonyms_dict,
                        threshold_range_p,
                        PRECISION_FUNC_CHOICES[args.precision_func],
                        relation2id,
                        relation2uri
                        ]
                    )
                )
    pool.close()
    pool.join()

    # combine results
    results = list(map(lambda x: x.get(), results))
    precision_dict = {
            k: v
            for result in results
            for k, v in result.items()
            }
    precision_x_values = list(map(
        lambda x: precision_dict[x], sorted(precision_dict)
        ))

    # prepare plotting
    plot_prepare(topk)

    # plot baselines
    if args.baseline_dir and os.path.isdir(args.baseline_dir):
        # set plotting range for topk to 500
        # as the baselines only went to top 500
        plt.xlim((0.0, 1.0) if not topk else (0, 500))

        # get filenames of precision values
        transh_fn = os.path.join(args.baseline_dir, "transh.txt")
        transd_fn = os.path.join(args.baseline_dir, "transd.txt")
        distmult_fn = os.path.join(args.baseline_dir, "distmult.txt")
        hole_fn = os.path.join(args.baseline_dir, "hole.txt")
        complex_fn = os.path.join(args.baseline_dir, "complex.txt")
        analogy_fn = os.path.join(args.baseline_dir, "analogy.txt")
        freqitemset_fn = os.path.join(args.baseline_dir, "freqitemset.txt")

        # plot baselines
        plot_baseline(transh_fn, "TransH", "#ffb24d")
        plot_baseline(transd_fn, "TransD", "#b0ff78")
        plot_baseline(distmult_fn, "DistMult", "#78ffd2")
        plot_baseline(hole_fn, "HolE", "#78acff")
        plot_baseline(complex_fn, "ComplEx", "#ce78ff")
        plot_baseline(analogy_fn, "Analogy", "#ff78be")
        plot_baseline(freqitemset_fn, "Freq. Item Set", "#000000")

    # plot precision
    plot_precision(precision_x_values)

    # save plot
    plot_fn = plot_fn.format(args.precision_func, args.classifier)
    plot_finish(plot_fn)


if __name__ == "__main__":
    main()
