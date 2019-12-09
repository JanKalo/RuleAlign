import sys
import os

from argparse import ArgumentParser
from amie_rule_wrapper import NoAMIERuleInLineError, AMIERule
from multiprocessing import Pool


NORMALIZATION_CHOICES = [
        "standard",
        "min-max",
        "jaccard"
        ]


class Config(object):

    def __init__(
            self,
            relation2id_fn,
            amie_output_fn,
            min_std,
            weight_std,
            weight_pca,
            weight_hc,
            max_diff_std_hc,
            normalization,
            processes
            ):
        self._relation2id_fn = relation2id_fn
        self._amie_output_fn = amie_output_fn
        self._min_std = min_std
        self._weight_std = weight_std
        self._weight_pca = weight_pca
        self._weight_hc = weight_hc
        self._max_diff_std_hc = max_diff_std_hc
        if normalization and normalization not in NORMALIZATION_CHOICES:
            raise AttributeError(
                    "ERROR: unknown normalization method"
                    )
        self._normalization = normalization
        self._processes = processes

    @property
    def relation2id_fn(self):
        return self._relation2id_fn

    @property
    def amie_output_fn(self):
        return self._amie_output_fn

    @property
    def min_std(self):
        return self._min_std

    @property
    def weight_std(self):
        return self._weight_std

    @property
    def weight_pca(self):
        return self._weight_pca

    @property
    def weight_hc(self):
        return self._weight_hc

    @property
    def max_diff_std_hc(self):
        return self._max_diff_std_hc

    @property
    def normalization(self):
        return self._normalization

    @property
    def processes(self):
        return self._processes

    def get_output_dir_basename(self):
        # get weight name
        weight_name = "weight"
        if self._weight_std:
            weight_name += "-std"
        if self._weight_pca:
            weight_name += "-pca"
        if self._weight_hc:
            weight_name += "-hc"
        if (
                not self._weight_std and
                not self._weight_pca and
                not self._weight_hc
                ):
            weight_name += "none"

        # get normalization name
        normalization_name = "norm-"
        normalization_name += str(self._normalization).lower()

        # concat
        return "{0}_{1}".format(weight_name, normalization_name)

    def get_output_dir(self):
        output_dir = os.path.join(
                os.path.dirname(self._amie_output_fn),
                self.get_output_dir_basename()
                )
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        return output_dir

    def get_output_fn(self):
        output_fn = os.path.join(
                self.get_output_dir(),
                os.path.basename(self._amie_output_fn)
                ) + ".synonyms"
        return output_fn


# load mapping2id file
def loadMapping2uri(mapping2id_fn):
    # load mapping as dict
    with open(mapping2id_fn, "r") as f:
        lines = f.readlines()[1:]
        lines = list(map(lambda x: x.split(), lines))
        mapping2id = {int(y): x for x, y in lines}
    return mapping2id


# save benchmark as dataset (default: *.nt formatting)
def saveAsDataset(
        train2id_fn,
        entity2id_fn,
        relation2id_fn,
        output_fn,
        line_formatting="<{0}> <{1}> <{2}> .\n"):
    # load mappings
    entity2id = loadMapping2uri(entity2id_fn)
    relation2id = loadMapping2uri(relation2id_fn)

    # load train2id file
    with open(train2id_fn, "r") as f:
        lines = f.readlines()[1:]
        train2id = list(map(
            lambda x: tuple(map(lambda y: int(y), x.split())),
            lines
            ))

    # save as *.nt file
    with open(output_fn, "w") as f:
        for triple2id in train2id:
            f.write(line_formatting.format(
                entity2id[triple2id[0]],
                relation2id[triple2id[2]],
                entity2id[triple2id[1]]
                ))


# parse amie rules
def parseAMIERules(config):
    amie_rules = set()
    with open(config.amie_output_fn, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                amie_rule = AMIERule(line)
                if amie_rule.std_confidence >= config.min_std:
                    amie_rules.add(amie_rule)
            except NoAMIERuleInLineError:
                # this line doesn't have a rule, just skip it
                continue
    return amie_rules


# check if body isomorphism of both rules is synonym
def isBodyIsomorphismSynonym(rule1, rule2):
    # apply isomorphism algorithm on both rules
    dgm = rule1.getBodyIsomorphism(rule2)

    # stop here if bodies aren't even isomorphic
    if not dgm.is_isomorphic():
        return False

    # if heads subject & object of rule1 map
    # to the heads subject & object of rule2,
    # the rules body isomorphism is synonym
    return (dgm.mapping[rule1.head.s] == rule2.head.s
            and dgm.mapping[rule1.head.o] == rule2.head.o)


# check if body isomorphism of both rules is inverse
def isBodyIsomorphismInverse(rule1, rule2):
    # apply isomorphism algorithm on both rules
    dgm = rule1.getBodyIsomorphism(rule2)

    # stop here if bodies aren't even isomorphic
    if not dgm.is_isomorphic():
        return False

    # if heads subject & object of rule1 map
    # to the heads object & subject of rule2,
    # the rules body isomorphism is inverse
    return (dgm.mapping[rule1.head.s] == rule2.head.o
            and dgm.mapping[rule1.head.o] == rule2.head.s)


# check if rule is closed for two relations
def isRuleClosed(rule, relation1, relation2):
    # rule is considered closed
    # if the body length is 1
    # and the properties of the head and body triple
    # are relation1 vs. relation2 and vice versa
    return (
            len(rule.body) == 1
            and
            (
                (
                    next(iter(rule.body)).p == relation1
                    and
                    rule.head.p == relation2
                    )
                or
                (
                    next(iter(rule.body)).p == relation2
                    and
                    rule.head.p == relation1
                    )
                )
            )


# process function for building relation rules dictionary
def buildRelationRulesDict_process(
        relations_p,
        amie_rules
        ):
    relation_rules = {
            relation: set(filter(
                lambda x: x.head.p == relation, amie_rules
                ))
            for relation in relations_p
            }
    return relation_rules


# process function for getting synonym relations
def getSynonymRelations_process(
        config,
        relations_p,
        relations,
        relation_rules,
        amie_rules
        ):
    # get confidences for each relation pair
    relation_pair_confidence = {}
    for relation1 in relations_p:
        relation1_rules = relation_rules[relation1]
        for relation2 in relations:
            relation2_rules = relation_rules[relation2]
            total_confidence = 0.0
            total_weight = 1.0
            skip_pair = False
            for rule1 in relation1_rules:
                # check if we have to skip this pair
                if skip_pair:
                    break

                # if rule1 is closed
                # for relation1 and relation2,
                # check the diff of its std and hc
                # and skip the whole relation pair if it's too high
                # to prevent subproperty relations
                if isRuleClosed(rule1, relation1, relation2):
                    if abs(
                            rule1.std_confidence - rule1.head_coverage
                            ) > config.max_diff_std_hc:
                        skip_pair = True
                        break

                # continue for this rule with all other rules
                for rule2 in relation2_rules:
                    # do the same check as for rule1:
                    # if rule2 is closed
                    # for relation1 and relation2,
                    # check the diff of its std and hc
                    # and skip the whole relation pair if it's too high
                    # to prevent subproperty relations
                    if isRuleClosed(rule2, relation1, relation2):
                        if abs(
                                rule2.std_confidence - rule2.head_coverage
                                ) > config.max_diff_std_hc:
                            skip_pair = True
                            break

                    # perform confidence calculation
                    if rule1.areBodyRelationsEqual(rule2):
                        # get confidence as 1 or 0 (isomorphic or not)
                        if isBodyIsomorphismSynonym(rule1, rule2):
                            # get minimum weight and weight confidence with it
                            confidence = 1.0
                            min_weight = 1.0
                            if config.weight_std:
                                min_weight = min(
                                        min_weight,
                                        min(
                                            rule1.std_confidence,
                                            rule2.std_confidence
                                            )
                                        )
                            if config.weight_pca:
                                min_weight = min(
                                        min_weight,
                                        min(
                                            rule1.pca_confidence,
                                            rule2.pca_confidence
                                            )
                                        )
                            if config.weight_hc:
                                min_weight = min(
                                        min_weight,
                                        min(
                                            rule1.head_coverage,
                                            rule2.head_coverage
                                            )
                                        )
                            confidence *= min_weight

                            # add to total confidence and weight of this pair
                            total_confidence += confidence
                            total_weight *= min_weight

            # check if we have to apply jaccard
            if config.normalization == "jaccard":
                total_len_rules = (
                        len(relation1_rules)
                        +
                        len(relation2_rules)
                        )
                total_len_rules *= total_weight
                jaccard = (
                        total_len_rules
                        -
                        total_confidence
                        )
                if total_confidence > 0.0:
                    if jaccard > 0.0:
                        total_confidence /= jaccard
                    else:
                        # should not happen
                        total_confidence = 1.0

            # only add relation pairs to result with positive confidence
            # and ignore skipped ones
            if not skip_pair and total_confidence > 0.0:
                relation_pair_confidence[
                        frozenset([relation1, relation2])
                        ] = total_confidence

    # normalize confidences
    if config.normalization == "standard":
        raise NotImplementedError
    elif config.normalization == "min-max":
        c_max = max(relation_pair_confidence.values())
        c_min = min(relation_pair_confidence.values())
        c_max_min_diff = c_max - c_min
        relation_pair_confidence = {
                k: (v - c_min) / c_max_min_diff
                for k, v in relation_pair_confidence.items()
                }

    # done
    return relation_pair_confidence


# get synonym relations
def getSynonymRelations(config):
    # loading mapping and amie rules
    print("INFO: loading relation2id mapping")
    relations = loadMapping2uri(config.relation2id_fn)
    relations = list(map(lambda x: relations[x], relations))
    print("INFO: parsing AMIE rules")
    amie_rules = parseAMIERules(config)

    # building relation_rules dictionary
    print("INFO: building relation_rules dictionary")
    pool = Pool(processes=config.processes)
    results = []
    for p in range(0, config.processes):
        relations_p = [
                relations[i]
                for i in range(p, len(relations), config.processes)
                ]
        results.append(
                pool.apply_async(
                    buildRelationRulesDict_process,
                    [
                        set(relations_p),
                        amie_rules
                        ]
                    )
                )
    pool.close()
    pool.join()
    results = list(map(lambda x: x.get(), results))
    relation_rules = {
            k: v
            for result in results
            for k, v in result.items()
            }

    # get relation pairs and their confidences
    print("INFO: getting synonymous relation pairs with their confidences")
    pool = Pool(processes=config.processes)
    results = []
    for p in range(0, config.processes):
        relations_p = [
                relations[i]
                for i in range(p, len(relations), config.processes)
                ]
        results.append(
                pool.apply_async(
                    getSynonymRelations_process,
                    [
                        config,
                        set(relations_p),
                        set(relations),
                        relation_rules,
                        amie_rules
                        ]
                    )
                )
    pool.close()
    pool.join()
    results = list(map(lambda x: x.get(), results))
    relation_pair_confidences = {
            k: v
            for result in results
            for k, v in result.items()
            }

    # write relation_pair_confidences dictionary directly
    print(
            "INFO: writing relation_pair_confidences "
            "dictionary to file"
            )
    with open(config.get_output_fn() + ".dict", "w") as f:
        f.write(str(relation_pair_confidences))

    # write human readable relation_pair_confidences
    print(
            "INFO: writing relation_pair_confidences "
            "in a human readable format to file"
            )
    with open(config.get_output_fn() + ".txt", "w") as f:
        for relation_pair in sorted(
                relation_pair_confidences,
                key=relation_pair_confidences.get,
                reverse=True
                ):
            if len(relation_pair) == 2:
                it = iter(relation_pair)
                relation1 = next(it)
                relation2 = next(it)
                confidence = relation_pair_confidences[relation_pair]
                f.write("{0}\t{1}\t{2}\n".format(
                    relation1,
                    relation2,
                    confidence
                    ))

    # done
    return relation_pair_confidences


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
            "-r", "--relation2id",
            type=str, required=True,
            help="The relation2id.txt mapping file of the dataset."
            )
    parser.add_argument(
            "-a", "--amie-output",
            type=str, required=True,
            help="The amie output containing the mined rules of the dataset."
            )
    parser.add_argument(
            "--min-std",
            type=float, default=0.2,
            help=(
                "The minimum std confidence score to filter AMIE rules with. "
                "(Default: 0.2)."
                )
            )
    parser.add_argument(
            "--weight-std", action="store_true",
            help="Weight confidence scores with rules' std confidence."
            )
    parser.add_argument(
            "--weight-pca", action="store_true",
            help="Weight confidence scores with rules' pca confidence."
            )
    parser.add_argument(
            "--weight-hc", action="store_true",
            help="Weight confidence scores with rules' head coverage."
            )
    parser.add_argument(
            "--max-diff-std-hc",
            type=float, default=0.5,
            help=(
                "The maximal allowed difference of std confidence "
                "and head coverage for a closed rule of two relations. "
                "(Default: 0.5)."
                )
            )
    parser.add_argument(
            "-n", "--normalization",
            type=str, choices=NORMALIZATION_CHOICES, default=None,
            help=(
                "The normalization method for the confidence "
                "scores to use. (Default: None)."
                )
            )
    parser.add_argument(
            "-p", "--processes",
            type=int, default=4,
            help="The number of processes to use. (Default: 4)."
            )
    args = parser.parse_args()

    # check filenames
    if not os.path.isfile(args.relation2id):
        sys.exit("ERROR: specified relation2id mapping file does not exist")
    if not os.path.isfile(args.amie_output):
        sys.exit("ERROR: specified amie output file does not exist")

    # create config
    config = Config(
            args.relation2id,
            args.amie_output,
            args.min_std,
            args.weight_std,
            args.weight_pca,
            args.weight_hc,
            args.max_diff_std_hc,
            args.normalization,
            args.processes
            )

    # do work
    getSynonymRelations(config)


if __name__ == "__main__":
    main()
