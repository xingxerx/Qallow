from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import NvRules
from MetricAggregation import ByOpcodeMetricAggregate, ByOpcodeMetricAggregator


class TableBuilder(ABC):
    """Interface to construct inputs to the NvRules.IFrontend.generate_table method."""

    @abstractmethod
    def build(self) -> tuple[list[str], list[list[Any]], dict[str, Any] | None]:
        """Build the inputs to the NvRules.IFrontend.generate_table method.

        Returns:
            Header, data and (optional) configuration for the table.
        """
        pass


class AggregatedMetricByOpcodeTableBuilder(TableBuilder):
    """Implementation to build a table from a metric that is aggregated by opcodes.

    The table will display the top N source locations (and their associated source code)
    that have the highest aggregated value for the the given metric and opcodes.

    The table data will look as follows:

    | Location                    | Source     | Opcode | Aggregated value |
    | --------------------------- | ---------- | ------ | ---------------- |
    | file_name.cpp, line 12      | c = a + b; | FADD   | 1283847          |
    | ...                         | ...        | ...    | ...              |

    The "Source" column will only be displayed if there is any imported source code
    in the report.

    Attributes:
        _sources: A class cache to store source file contents by workload name.
        _sources_by_line: A class cache to store source file contents split into lines.
        _source_unknown_placeholder: A placeholder string to use when some source code
            is not found. This will only be used if source code was imported in the report.

    Args:
        workload: The NvRules.IAction associated with the metric.
        metric: The metric to aggregate and display.
        opcodes: The opcode(s) to aggregate by.
        group_aggregation: Whether to aggregate individually by each opcode or
            to have one aggregate including all opcodes.
            In the latter case, the third column will be called "Opcodes",
            and show a list of all opcodes for which metric values contributed to
            the aggregate.
    """

    _sources: dict[str, NvRules.map_string_string] = dict()  # workload name -> source files
    _sources_by_line: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)  # workload name -> file path -> lines
    _source_unknown_placeholder = "???"

    def __init__(
        self,
        workload: NvRules.IAction,
        metric: NvRules.IMetric,
        opcodes: str | list[str],
        group_aggregation: bool,
    ) -> None:
        self.workload = workload
        self.metric = metric
        self.opcodes = opcodes
        self.group_aggregation = group_aggregation

        # Cache source code by workload name
        self.has_source = False
        if self._sources.get(workload.name()) is None:
            sources = self.workload.source_files()
            if sources:
                self.has_source = True
                self._sources[workload.name()] = sources
        else:
            self.has_source = True

    def build(
        self,
        title: str,
        description: str,
        metric_column_name: str,
        metric_column_tooltip: str,
        top_n: int,
    ) -> tuple[list[str], list[list[Any]], dict[str, Any]]:
        """Build the inputs to the NvRules.IFrontend.generate_table method.

        Args:
            title: The title of the table.
            description: The description of the table.
            metric_column_name: The name of the column that displays the aggregated values.
            metric_column_tooltip: The tooltip for the metric column.
            top_n: The top N highest aggregated values to display.

        Returns:
            Header, data and configuration for the table.
        """
        opcode_column_name = "Opcode"
        opcode_column_tooltip = "Types of SASS instruction"
        if self.group_aggregation:
            opcode_column_name = "Opcodes"
            opcode_column_tooltip = "Types of SASS instructions"

        config = {
            "title": title,
            "description": description,
            "sort_by": {"column": metric_column_name, "order": "descending"},
        }

        if self.has_source:
            header = ["Location", "Source", opcode_column_name, metric_column_name]
            column_config = {
                "per_column_configs": {
                    "Location": {
                        "tooltip": "Location in the source file",
                        "relative_width": 0.2,
                    },
                    "Source": {
                        "tooltip": "Source code line",
                        "relative_width": 0.4,
                    },
                    opcode_column_name: {
                        "tooltip": opcode_column_tooltip,
                        "relative_width": 0.2,
                    },
                    metric_column_name: {
                        "tooltip": metric_column_tooltip,
                        "relative_width": 0.2,
                    },
                },
            }
        else:
            header = ["Location", opcode_column_name, metric_column_name]
            column_config = {
                "per_column_configs": {
                    "Location": {
                        "tooltip": "Location in the source file",
                        "relative_width": 0.4,
                    },
                    opcode_column_name: {
                        "tooltip": opcode_column_tooltip,
                        "relative_width": 0.3,
                    },
                    metric_column_name: {
                        "tooltip": metric_column_tooltip,
                        "relative_width": 0.3,
                    },
                },
            }

        config.update(column_config)
        data = self._generate_data(top_n)

        return header, data, config

    def get_aggregates(self) -> list[ByOpcodeMetricAggregate]:
        """Get the list of aggregates that was generated in the last call to build."""
        return self.aggregates

    def _generate_data(self, top_n: int) -> list[list[Any]]:
        aggregator = ByOpcodeMetricAggregator(self.workload, self.metric)
        self.aggregates = aggregator.get_aggregates(
            self.opcodes, top_n, self.group_aggregation
        )

        data = list()

        for result in self.aggregates:
            source_link = self._get_source_link(result)
            opcodes = self._get_opcodes(result)

            if self.has_source:
                source_code = self._get_source_code(result)
                row = [source_link, source_code, opcodes, result.value]
            else:
                row = [source_link, opcodes, result.value]

            data.append(row)

        return data

    def _get_source_link(self, result: ByOpcodeMetricAggregate) -> str:
        file_name = Path(result.source_location.path).name
        line = result.source_location.line
        line_to_navigate = line - 1

        link_text = f"{file_name}, line {line}"
        source_link = f"@source:{file_name}:{line_to_navigate}:{link_text}@"

        return source_link

    def _get_source_code(self, result: ByOpcodeMetricAggregate) -> str:
        file_path = result.source_location.path
        line_to_navigate = result.source_location.line - 1

        # split source file content by lines and cache
        workload_name = self.workload.name()
        if self._sources_by_line[workload_name].get(file_path) is None:
            if not self._sources[workload_name].has_key(file_path):
                return self._source_unknown_placeholder

            self._sources_by_line[workload_name][file_path] = self._sources[
                workload_name
            ][file_path].splitlines()

        try:
            source_code = self._sources_by_line[workload_name][file_path][
                line_to_navigate
            ].strip()
        except IndexError:
            return self._source_unknown_placeholder

        return source_code

    def _get_opcodes(self, result: ByOpcodeMetricAggregate) -> str:
        if isinstance(result.opcodes, str):
            return result.opcodes
        else:
            return ", ".join(result.opcodes)


class OpcodeTableBuilder(TableBuilder):
    """Build a table for a source-correlated executed instructions metric,
    individually aggregated by opcodes.

    Example Table:

    | Location                | Source       | Opcode | Executed Instructions |
    | ----------------------- | ----------   | ------ | --------------------- |
    | some_file.cpp, line 12  | c = a + b;   | FADD   | 1283847               |
    | other_file.cpp, line 42 | d *= 3;      | FMUL   | 938280                |
    | some_file.cpp, line 78  | z = a + 3*c; | FFMA   | 4800                  |

    By default, `build` will display the top 3 locations with the highest number of
    executed instructions.

    The "Source" column will only be displayed if there is any imported source code
    in the report.

    Args:
        workload: The NvRules.IAction associated with the metric.
        instruction_metric: The NvRules.IMetric used for aggregation.
        opcodes: The opcode(s) to aggregate by (individually).
    """

    def __init__(
        self,
        workload: NvRules.IAction,
        instruction_metric: NvRules.IMetric,
        opcodes: str | list[str],
    ) -> None:
        self.tableBuilder = AggregatedMetricByOpcodeTableBuilder(
            workload=workload,
            metric=instruction_metric,
            opcodes=opcodes,
            group_aggregation=False,
        )

    def build(
        self,
        title: str,
        description: str,
        top_n: int = 3,
    ) -> tuple[list[str], list[list[Any]], dict[str, Any]]:
        return self.tableBuilder.build(
            title=title,
            description=description,
            metric_column_name="Executed Instructions",
            metric_column_tooltip="Number of executed instructions of type Opcode",
            top_n=top_n,
        )

    def get_aggregates(self) -> list[ByOpcodeMetricAggregate]:
        return self.tableBuilder.get_aggregates()


class PipelineTableBuilder(TableBuilder):
    """Build a table for a source-correlated executed instructions metric,
    aggregated by all opcodes.

    Example Table:

    | Location           | Source        | Opcodes           | Executed Instructions |
    | ------------------ | ------------- | ----------------- | --------------------- |
    | file_1.py, line 12 | d = (a-b) / c | DFMA, DADD, MOV   | 1283847               |
    | file_2.py, line 42 | s = a / b;    | DFMA, DMUL, FFMA  | 938280                |
    | file_2.py, line 78 | z = a - b     | DADD              | 4800                  |

    By default, `build` will display the top 3 locations with the highest number of
    executed instructions.

    The "Source" column will only be displayed if there is any imported source code
    in the report.

    Args:
        workload: The NvRules.IAction associated with the metric.
        instruction_metric: The NvRules.IMetric used for aggregation.
        opcodes: The group of opcode(s) to aggregate by.
    """

    def __init__(
        self,
        workload: NvRules.IAction,
        instruction_metric: NvRules.IMetric,
        opcodes: str | list[str],
    ) -> None:
        self.tableBuilder = AggregatedMetricByOpcodeTableBuilder(
            workload=workload,
            metric=instruction_metric,
            opcodes=opcodes,
            group_aggregation=True,
        )

    def build(
        self,
        title: str,
        description: str,
        top_n: int = 3,
    ) -> tuple[list[str], list[list[Any]], dict[str, Any]]:
        return self.tableBuilder.build(
            title=title,
            description=description,
            metric_column_name="Executed Instructions",
            metric_column_tooltip="Number of executed instructions for any type in Opcodes",
            top_n=top_n,
        )

    def get_aggregates(self) -> list[ByOpcodeMetricAggregate]:
        return self.tableBuilder.get_aggregates()
