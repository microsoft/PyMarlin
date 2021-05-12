import os
import sys
import json
import csv
from abc import abstractmethod

from transformers import InputExample

from pymarlin.core.data_interface import DataProcessor
from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

from metric_utils import simple_accuracy, mcc, pearson_and_spearman, acc_and_f1


class GLUEBaseProcessor(DataProcessor):
    """Base Processor for all GLUE tasks"""
    def __init__(self, args):
        self.args = args

    def _read_tsv(self, input_file, quotechar=None, encoding='utf-8'):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, encoding) for cell in line)
                lines.append(line)
            return lines

    @abstractmethod
    def _create_examples(self):
        """Implement this method based on GLUE task"""

    def process(self, input_filepath):
        self.examples = self._create_examples(self._read_tsv(input_filepath), self.args.set_type)
        return self.examples

    def analyze(self):
        logger.info(f"# of examples created = {len(self.examples)}")

class MrpcProcessor(GLUEBaseProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[3]
            text_b = line[4]
            label = line[0] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(GLUEBaseProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[8]
            text_b = line[9]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class ColaProcessor(GLUEBaseProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and set_type == "test":
                continue
            guid = i
            text_a = line[3] if set_type != "test" else line[1]
            label = line[1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(GLUEBaseProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[0] if set_type != "test" else line[1]
            label = line[1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(GLUEBaseProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[7]
            text_b = line[8]
            label = line[-1] if set_type != "test" else "0.0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(GLUEBaseProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            try:
                text_a = line[3] if set_type != "test" else line[1]
                text_b = line[4] if set_type != "test" else line[2]
                label = line[5] if set_type != "test" else "0"
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(GLUEBaseProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(GLUEBaseProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)
        
    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WnliProcessor(GLUEBaseProcessor):
    """Processor for the WNLI data set (GLUE version)."""
    def __init__(self, args):
        super().__init__(args)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_dict = {
    "cola": {"processor": ColaProcessor, "output_mode": "classification", "metric": mcc},
    "mnli": {"processor": MnliProcessor, "output_mode": "classification", "metric": simple_accuracy},
    "mrpc": {"processor": MrpcProcessor, "output_mode": "classification", "metric": acc_and_f1},
    "sst-2": {"processor": Sst2Processor, "output_mode": "classification", "metric": simple_accuracy},
    "sts-b": {"processor": StsbProcessor, "output_mode": "regression", "metric": pearson_and_spearman},
    "qqp": {"processor": QqpProcessor, "output_mode": "classification", "metric": acc_and_f1},
    "qnli": {"processor": QnliProcessor, "output_mode": "classification", "metric": simple_accuracy},
    "rte": {"processor": RteProcessor, "output_mode": "classification", "metric": simple_accuracy},
    "wnli": {"processor": WnliProcessor, "output_mode": "classification", "metric": simple_accuracy},
}