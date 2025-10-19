from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from re import compile as re_compile

import NvRules


@dataclass(frozen=True)
class MetricAggregate:
    """A dataclass to store aggregated metric values.

    Attributes:
        name: The name of the metric.
        value: The aggregated value of the metric.
    """

    name: str
    value: float


class MetricAggregator(ABC):
    """Interface to aggregate metric values.

    Args:
        workload: The NvRules.IAction associated with the metric.
        metric: The metric to be aggregated.
    """

    @abstractmethod
    def __init__(
        self,
        workload: NvRules.IAction,
        metric: NvRules.IMetric,
        *args,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def get_aggregates(self, top_n: int | None = None) -> list[MetricAggregate]:
        """Aggregate metric values.

        Returns:
            A list of the top-N MetricAggregate objects by value.
            If top_n is None, return all.
        """
        pass


@dataclass(frozen=True)
class SourceLocation:
    """A dataclass to store high-level source location information.

    Attributes:
        path: The full path to the source file.
        line: The 1-based high-level source line.
    """

    path: str
    line: int


@dataclass(frozen=True)
class ByOpcodeMetricAggregate(MetricAggregate):
    """A dataclass for metrics aggregated by opcode(s), and their associate source location.

    Attributes:
        source_location: Location of the high-level source code for which
            instructions where aggregated.
        opcodes: The type(s) of instructions for which the metric was aggregated.
    """

    source_location: SourceLocation
    opcodes: str | list[str]


class PcCache:
    """A cache to store mappings from and to PCs for a given IAction.

    Lookup of SASS instructions and high-level source locations from PCs is costly.
    However, within a given IAction (profile result), these mappings are immutable
    and lookup only needs to be done once per PC.

    This class provides caches for the following mappings:

    1. Opcode to PCs
    2. PC to high-level source location
    3. PC to opcode

    It uses the SWIG generated __hash__ function of NvRules.IAction
    to store cached values in dicts.
    """
    PC = int
    OpcodeToPcs = defaultdict[str, set[PC]]
    PcToLocation = dict[PC, SourceLocation]
    PcToOpcode = dict[PC, str]

    opcode_to_pcs: defaultdict[NvRules.IAction, OpcodeToPcs] = defaultdict(lambda: defaultdict(set))
    pc_to_location: defaultdict[NvRules.IAction, PcToLocation] = defaultdict(lambda: defaultdict(SourceLocation))
    pc_to_opcode: defaultdict[NvRules.IAction, PcToOpcode] = defaultdict(lambda: defaultdict(str))

    def __init__(self):
        pass

    def get_opcode_to_pcs(self, action: NvRules.IAction) -> OpcodeToPcs:
        return self.opcode_to_pcs[action]

    def get_pc_to_location(self, action: NvRules.IAction) -> PcToLocation:
        return self.pc_to_location[action]

    def get_pc_to_opcode(self, action: NvRules.IAction) -> PcToOpcode:
        return self.pc_to_opcode[action]


class ByOpcodeMetricAggregator(MetricAggregator):
    """Implementation to aggregate metric values by opcode(s).

    Attributes:
        _opcode_regex: A regex pattern to match opcodes in SASS code.
        _opcode_to_pcs: Map opcodes to the set of PCs where they occur.
        _pc_to_location: Map PCs to high-level source locations.
        _pc_to_opcode: Map PCs to the instruction type they represent.
        _pc_to_metric_value: PC to metric value mapping for self.metric.

    Args:
        workload: The NvRules.IAction associated with the metric.
        metric: The metric to be aggregated.
    """

    _opcode_regex = re_compile(r"\s*(?:@\!?P\d\s+)?([\w\.]+)\s+.*")

    def __init__(
        self,
        workload: NvRules.IAction,
        metric: NvRules.IMetric,
    ) -> None:
        self.action = workload
        self.metric = metric

        # Cache all mappings from and to PCs on the level of IActions
        # so the costly SASS/source correlation lookup is done only once
        # per profile result
        pc_cache = PcCache()
        self._opcode_to_pcs = pc_cache.get_opcode_to_pcs(workload)
        self._pc_to_location = pc_cache.get_pc_to_location(workload)
        self._pc_to_opcode = pc_cache.get_pc_to_opcode(workload)

        self._pc_to_metric_value: dict[int, float] = {}

    def get_aggregates(
        self,
        opcodes: str | list[str],
        top_n: int | None = None,
        group_aggregation: bool = False,
    ) -> list[ByOpcodeMetricAggregate]:
        """Aggregate metric values by opcode(s).

        Generate/update the caches before aggregating the metric values.
        Return the top-N aggregated metric values (or all).
        """
        opcodes = [opcodes] if isinstance(opcodes, str) else opcodes

        self._update_pc_caches()
        self._get_metric_values_by_pc()
        metric_aggregates = self._aggregate_by_opcodes(opcodes, group_aggregation)

        return metric_aggregates[:top_n] if top_n is not None else metric_aggregates

    def _update_pc_caches(self) -> None:
        """Update caches for PC to opcode and location mappings.

        For all (yet unknown) PCs that the metric is correlated to:

        1. Get the opcode by calling IAction.sass_by_pc, and regex matching
        2. Get the high-level source location by calling IAction.source_info
        """
        num_instances = self.metric.num_instances()
        pcs = self.metric.correlation_ids()

        for instance in range(num_instances):
            pc = pcs.value(instance)

            if pc in self._pc_to_opcode:
                continue

            sass = self.action.sass_by_pc(pc)
            regex_match = self._opcode_regex.match(sass)
            if regex_match:
                opcode = regex_match.group(1)
            else:
                continue

            source_info = self.action.source_info(pc)
            if source_info is None:
                # NOTE: Some PCs do not have high-level source line correlation,
                # even though source correlation is available for others.
                # We will skip those instructions here, as we are only interested
                # in high-level source locations.
                # This also, implicitly, handles the case where no source correlation
                # is available at all.
                continue

            self._opcode_to_pcs[opcode].add(pc)
            self._pc_to_opcode[pc] = opcode
            self._pc_to_location[pc] = SourceLocation(
                source_info.file_name(), source_info.line()
            )

    def _get_metric_values_by_pc(self) -> None:
        """Get metric values for all PCs that the metric is correlated to."""
        num_instances = self.metric.num_instances()
        pcs = self.metric.correlation_ids()

        for instance in range(num_instances):
            metric_value = self.metric.value(instance)
            pc = pcs.value(instance)
            self._pc_to_metric_value[pc] = metric_value

    def _aggregate_by_opcodes(
        self,
        opcodes: list[str],
        group_aggregation: bool,
    ) -> list[ByOpcodeMetricAggregate]:
        """Aggregate metric by opcodes:

        Algorithm:

        1. Get all relevant PCs for the given opcodes
        2. Aggregate the metric values for these PCs by opcode, then location
        3. Sort by aggregate value in descending order
        """
        relevant_pcs = set()
        for opcode in opcodes:
            # only consider PCs of the given opcodes that also have a metric value
            pcs_for_opcode = self._opcode_to_pcs.get(opcode, set())
            for pc in pcs_for_opcode:
                if pc in self._pc_to_metric_value.keys():
                    relevant_pcs.add(pc)

        if group_aggregation:
            metric_aggregates = self._aggregate_in_group(relevant_pcs)
        else:
            metric_aggregates = self._aggregate_individually(relevant_pcs)

        metric_aggregates.sort(key=lambda x: x.value, reverse=True)

        return metric_aggregates

    def _aggregate_in_group(
        self,
        pcs: set[int],
    ) -> list[ByOpcodeMetricAggregate]:
        # track aggregates by high-level location, and all opcodes associated
        # with this location
        aggregates: defaultdict[SourceLocation, float] = defaultdict(float)
        opcodes_by_location: defaultdict[SourceLocation, set[str]] = defaultdict(set)

        for pc in pcs:
            location = self._pc_to_location.get(pc)
            opcode = self._pc_to_opcode.get(pc)
            aggregates[location] += self._pc_to_metric_value[pc]
            opcodes_by_location[location].add(opcode)

        metric_aggregates = list()
        for location, value in aggregates.items():
            metric_aggregates.append(
                ByOpcodeMetricAggregate(
                    name=self.metric.name(),
                    value=value,
                    source_location=location,
                    opcodes=opcodes_by_location[location],
                )
            )

        return metric_aggregates

    def _aggregate_individually(
        self,
        pcs: set[int],
    ) -> list[ByOpcodeMetricAggregate]:
        # track aggregates by opcode and high-level location
        aggregates: defaultdict[str, defaultdict[SourceLocation, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for pc in pcs:
            location = self._pc_to_location.get(pc)
            opcode = self._pc_to_opcode.get(pc)
            aggregates[opcode][location] += self._pc_to_metric_value[pc]

        metric_aggregates = list()
        for opcode, locations_dict in aggregates.items():
            for location, value in locations_dict.items():
                metric_aggregates.append(
                    ByOpcodeMetricAggregate(
                        name=self.metric.name(),
                        value=value,
                        source_location=location,
                        opcodes=opcode,
                    )
                )

        return metric_aggregates
