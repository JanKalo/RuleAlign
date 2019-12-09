import re

import networkx as nx
import networkx.algorithms.isomorphism as nxiso

from enum import Enum, unique


class Triple(object):
    """This class represents a triple."""

    def __init__(self, s, p, o):
        """
        Constructor.

        Parameters
        ----------
        s : str
            The subject / head of the triple.
        p : str
            The predicate / relation of the triple.
        o : str
            The object / tail of the triple.
        """

        self._s = s
        self._p = p
        self._o = o

    def __str__(self):
        """String representation of this instance."""

        return "Triple: {0} {1} {2}".format(self.s, self.p, self.o)

    def __repr__(self):
        """Representation of this instance."""

        return self.__str__()

    def isHeadEqual(self, triple):
        """
        Determines if this triples' head
        is equal to the head of another triple.

        Parameters
        ----------
        triple : Triple
            The triple whose head to test with this triples' head.

        Returns
        -------
        bool
            Result.
        """

        return self._s == triple.s

    def isRelationEqual(self, triple):
        """
        Determines if this triples' relation
        is equal to the relation of another triple.

        Parameters
        ----------
        triple : Triple
            The triple whose relation to test with this triples' relation.

        Returns
        -------
        bool
            Result.
        """

        return self._p == triple.p

    def isTailEqual(self, triple):
        """
        Determines if this triples' tail
        is equal to the tail of another triple.

        Parameters
        ----------
        triple : Triple
            The triple whose tail to test with this triples' tail.

        Returns
        -------
        bool
            Result.
        """

        return self._o == triple.o

    def isDomainRangeEqual(self, triple):
        """
        Determines if this triples' domain and range
        is equal to those of another triple.

        Parameters
        ----------
        triple : Triple
            The triple whose domain and range to test with this triples' ones.

        Returns
        -------
        bool
            Result.
        """

        return self._s == triple.s and self._o == triple.o

    def isDomainRangeInverse(self, triple):
        """
        Determines if this triples' domain and range
        is inverse to those of another triple.

        Parameters
        ----------
        triple : Triple
            The triple whose domain and range to test with this triples' ones.

        Returns
        -------
        bool
            Result.
        """

        return self._s == triple.o and self._o == triple.s

    @property
    def s(self):
        """The subject / head of this triple."""

        return self._s

    @property
    def p(self):
        """The predicate / relation of this triple."""

        return self._p

    @property
    def o(self):
        """The object / tail of this triple."""

        return self._o


class NoAMIERuleInLineError(Exception):
    """
    This class represents the exception
    that no AMIE rule was found in a given line.
    """

    def __init__(self, amie_rule_line):
        """
        Constructor.

        Parameters
        ----------
        amie_rule_line : str
            The line of the generated AMIE output causing this error.
        """

        self._amie_rule_line = amie_rule_line

    def __str__(self):
        """String representation of this instance."""

        return (
                "No AMIE rule found in the given AMIE rule line.\n\n"
                "AMIE rule line: \"{0}\"".format(self._amie_rule_line)
                )

    def __repr__(self):
        """Representation of this instance."""

        return "NoAMIERuleInLineError: {0}".format(self.__str__())

    @property
    def amie_rule_line(self):
        """The line of the generated AMIE output causing this error."""

        return self._amie_rule_line


class AMIERuleParseError(Exception):
    """
    This class represents the exception
    that a parsing error has occured for a given line.
    """

    def __init__(self, amie_rule_line, message, inner_exception):
        """
        Constructor.

        Parameters
        ----------
        amie_rule_line : str
            The line of the generated AMIE output causing this error.
        message : str
            An optional error message.
        inner_exception : Exception
            An optional inner exception that has occured while parsing.
        """

        self._amie_rule_line
        self._message = message
        self._inner_exception = inner_exception

    def __str__(self):
        """String representation of this instance."""

        return (
                "An error occured while parsing "
                "the AMIE rule in the given line.\n\n"
                "{0}\n\nAMIE rule line: "
                "\"{1}\"\n\nInner Exception: \"{2}\"".format(
                    self._message,
                    self._amie_rule_line,
                    self._inner_exception
                    )
                )

    def __repr__(self):
        """Representation of this instance."""

        return "AMIERuleParseError: {0}".format(self.__str__())

    @property
    def amie_rule_line(self):
        """The line of the generated AMIE output causing this error."""

        return self._amie_rule_line

    @property
    def message(self):
        """The error message."""

        return self._message

    @property
    def inner_exception(self):
        """The inner exception that has occured while parsing."""

        return self._inner_exception


@unique
class AMIERuleType(Enum):
    """This enumeration covers all relevant AMIE rule types."""

    UNKNOWN = "UNKNOWN AMIERule"
    EQUIVALENT = "EQUIVALENT AMIERule"
    INVERSE = "INVERSE AMIERule"
    CHAIN = "CHAIN AMIERule"


class AMIERule(object):
    """This class acts as a wrapper representing an AMIE generated rule."""

    def __init__(self, amie_rule_line):
        """
        Constructor.

        Parameters
        ----------
        amie_rule_line : str
            A line of the generated AMIE output containing a valid rule.
        """

        self._body = set()
        self._head = None
        self._head_coverage = None
        self._std_confidence = None
        self._pca_confidence = None
        self._positive_examples = None
        self._body_size = None
        self._pca_body_size = None
        self._functional_variable = None
        self._std_lower_bound = None
        self._pca_lower_bound = None
        self._pca_conf_estimation = None
        self._rule_type = AMIERuleType.UNKNOWN
        self.__parse_amie_rule_line(amie_rule_line)

    def __parse_amie_rule_line(self, amie_rule_line):
        """
        Parses an AMIE rule line and fills this instance with its data.

        Parameters
        ----------
        amie_rule_line : str
            A line of the generated AMIE output containing a valid rule.

        Raises
        ------
        AMIERuleParseError
            If the rule type can't be identified or a general error occured.
        """

        # parse amie rule line and get regex match
        body_size = AMIERule.__getRuleBodySize(amie_rule_line)
        match = AMIERule.__getRuleMatch(amie_rule_line, body_size)
        rule_dict = match.groupdict()

        # get head
        head = Triple(
                rule_dict["head_s"],
                rule_dict["head_p"],
                rule_dict["head_o"]
                )
        self._head = head

        # get body
        for i in range(0, body_size):
            body = Triple(
                    rule_dict["body{0}_s".format(i)],
                    rule_dict["body{0}_p".format(i)],
                    rule_dict["body{0}_o".format(i)]
                    )
            self._body.add(body)

        # get other metrics
        try:
            self._head_coverage = float(
                    rule_dict["head_coverage"].replace(",", ".")
                    )
            self._std_confidence = float(
                    rule_dict["std_confidence"].replace(",", ".")
                    )
            self._pca_confidence = float(
                    rule_dict["pca_confidence"].replace(",", ".")
                    )
            self._positive_examples = int(
                    rule_dict["positive_examples"]
                    )
            self._body_size = int(
                    rule_dict["body_size"]
                    )
            self._pca_body_size = int(
                    rule_dict["pca_body_size"]
                    )
            self._functional_variable = str(
                    rule_dict["functional_variable"]
                    )
            self._std_lower_bound = float(
                    rule_dict["std_lower_bound"].replace(",", ".")
                    )
            self._pca_lower_bound = float(
                    rule_dict["pca_lower_bound"].replace(",", ".")
                    )
            self._pca_conf_estimation = float(
                    rule_dict["pca_conf_estimation"].replace(",", ".")
                    )
        except Exception as ex:
            raise AMIERuleParseError(
                    amie_rule_line,
                    "ERROR: an inner exception occured.",
                    ex
                    )

        # get rule type
        if len(self._body) > 1:
            self._rule_type = AMIERuleType.CHAIN
        else:
            # size of body can only be 1 at this point
            body_triple = next(iter(self._body))
            if self._head.isDomainRangeEqual(body_triple):
                self._rule_type = AMIERuleType.EQUIVALENT
            elif self._head.isDomainRangeInverse(body_triple):
                self._rule_type = AMIERuleType.INVERSE
            else:
                # should not happen
                raise AMIERuleParseError(
                        amie_rule_line,
                        "ERROR: inconclusive rule type",
                        None
                        )

    def __str__(self):
        """String representation of this instance."""

        result = "{0}: ".format(self._rule_type.value)
        for body_triple in self._body:
            result += "{0} {1} {2} AND ".format(
                    body_triple.s,
                    body_triple.p,
                    body_triple.o
                    )
        result = result[0:-4]
        result += "=> {0} {1} {2}".format(
                self._head.s,
                self._head.p,
                self._head.o
                )
        return result

    def __repr__(self):
        """Representation of this instance."""

        return self.__str__()

    def isHeadEqual(self, rule):
        """
        Determines if the heads of this rule
        and another rule are equal.

        Parameters
        ----------
        rule : AMIERule
            The other rule to compare.

        Returns
        -------
        bool
            Result.
        """

        return self._head.p == rule.head.p

    def areBodyRelationsEqual(self, rule):
        """
        Determines if the bodies' relations of this rule
        and another rule are equal.

        Parameters
        ----------
        rule : AMIERule
            The other rule to compare.

        Returns
        -------
        bool
            Result.
        """

        self_predicates = set(map(lambda x: x.p, self._body))
        other_predicates = set(map(lambda x: x.p, rule.body))
        return self_predicates == other_predicates

    def getBodyIsomorphism(self, rule):
        """
        Applies the isomorphism algorithm on the graphs
        of this and another rules' bodies.

        Parameters
        ----------
        rule : AMIERule
            The other rule to compare.

        Returns
        -------
        nxiso.DiGraphMatcher
            The DiGraphMatcher instance of the applied isomorphism algorithm.
        """

        # build graphs for the body of each rule and test for isomorphism
        self_graph = nx.DiGraph()
        for body_triple in self._body:
            self_graph.add_node(body_triple.s)
            self_graph.add_node(body_triple.o)
            self_graph.add_edge(
                    body_triple.s,
                    body_triple.o,
                    label=body_triple.p
                    )
        other_graph = nx.DiGraph()
        for body_triple in rule.body:
            other_graph.add_node(body_triple.s)
            other_graph.add_node(body_triple.o)
            other_graph.add_edge(
                    body_triple.s,
                    body_triple.o,
                    label=body_triple.p
                    )
        nm = nxiso.categorical_edge_match("label", "empty")
        dgm = nxiso.DiGraphMatcher(self_graph, other_graph, edge_match=nm)
        return dgm

    @property
    def body(self):
        """The body of this rule."""

        return self._body

    @property
    def head(self):
        """The head of this rule."""

        return self._head

    @property
    def head_coverage(self):
        """The head coverage of this rule."""

        return self._head_coverage

    @property
    def std_confidence(self):
        """The standard confidence of this rule."""

        return self._std_confidence

    @property
    def pca_confidence(self):
        """The PCA confidence of this rule."""

        return self._pca_confidence

    @property
    def positive_examples(self):
        """The number of positive examples of this rule."""

        return self._positive_examples

    @property
    def body_size(self):
        """The body size of this rule."""

        return self._body_size

    @property
    def pca_body_size(self):
        """The PCA body size of this rule."""

        return self._pca_body_size

    @property
    def functional_variable(self):
        """The functional variable of this rule."""

        return self._functional_variable

    @property
    def std_lower_bound(self):
        """The standard lower bound of this rule."""

        return self._std_lower_bound

    @property
    def pca_lower_bound(self):
        """The PCA lower bound of this rule."""

        return self._pca_lower_bound

    @property
    def pca_conf_estimation(self):
        """The PCA confidence estimation of this rule."""

        return self._pca_conf_estimation

    @property
    def rule_type(self):
        """The rule type of this rule."""

        return self._rule_type

    @staticmethod
    def __getRuleBodySize(amie_rule_line):
        """
        This static method parses the number of triples
        in the body of the rule in a given line.

        Parameters
        ----------
        amie_rule_line : str
            A line of the generated AMIE output containing a valid rule.

        Returns
        -------
        int
            The body size.

        Raises
        ------
        AMIERuleParseError
            If the AMIE rule line is invalid
            (no body, not found or general parse error).
        """

        # pattern for matching a body triple
        pattern = re.compile(
                r"(?P<s>\?[a-z])\s+(?P<p>[^\s]+)\s+(?P<o>\?[a-z])"
                )

        # cut amie rule line at => and match only the string before that token
        # which represents the body of the rule
        if "=>" in amie_rule_line:
            try:
                matches = re.findall(pattern, amie_rule_line.split("=>")[0])
            except Exception as ex:
                raise AMIERuleParseError(
                        amie_rule_line,
                        "ERROR: an inner exception occured.", ex
                        )
            if matches and len(matches) > 0:
                return len(matches)
            else:
                raise AMIERuleParseError(
                        amie_rule_line,
                        "ERROR: invalid AMIE rule line: no body found",
                        None
                        )
        else:
            raise NoAMIERuleInLineError(amie_rule_line)

    @staticmethod
    def __getRuleMatch(amie_rule_line, body_size):
        """
        This static method parses the whole rule in a given line.

        Parameters
        ----------
        amie_rule_line : str
            A line of the generated AMIE output containing a valid rule.
        body_size : int
            The body size / number of triples
            in the body of the rule in the given line.

        Returns
        -------
        re.Match
            The match instance containing the rule.

        Raises
        ------
        NoAMIERuleInLineError
            If no AMIE rule was found in the given line.
        """

        # assert positive body size
        assert body_size > 0

        # craft the regex pattern to match the whole amie rule line
        # expand "pattern_rule" with "pattern_body_triple" "body_size" times
        pattern_body_triple = (
                r"(?P<body{0}_s>\?[a-z])\s+"
                r"(?P<body{0}_p>[^\s]+)\s+"
                r"(?P<body{0}_o>\?[a-z])\s+"
                )
        pattern_rule = (
                r"^{0}=>"
                r"\s+(?P<head_s>\?[a-z])"
                r"\s+(?P<head_p>[^\s]+)"
                r"\s+(?P<head_o>\?[a-z])\s+"
                r"(?P<head_coverage>[^\s]+)\s+"
                r"(?P<std_confidence>[^\s]+)\s+"
                r"(?P<pca_confidence>[^\s]+)\s+"
                r"(?P<positive_examples>[^\s]+)\s+"
                r"(?P<body_size>[^\s]+)\s+"
                r"(?P<pca_body_size>[^\s]+)\s+"
                r"(?P<functional_variable>[^\s]+)\s+"
                r"(?P<std_lower_bound>[^\s]+)\s+"
                r"(?P<pca_lower_bound>[^\s]+)\s+"
                r"(?P<pca_conf_estimation>[^\s]+)$"
                )
        pattern_expansion = r""
        for i in range(0, body_size):
            pattern_expansion += pattern_body_triple.format(i)
        pattern_result = pattern_rule.format(pattern_expansion)
        pattern = re.compile(pattern_result)

        # return match
        match = re.match(pattern, amie_rule_line)
        if match:
            return match
        else:
            raise NoAMIERuleInLineError(amie_rule_line)
