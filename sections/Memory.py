# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # Redistribution and use in source and binary forms, with or without
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # modification, are permitted provided that the following conditions
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # are met:
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #  * Redistributions of source code must retain the above copyright
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #    notice, this list of conditions and the following disclaimer.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #  * Redistributions in binary form must reproduce the above copyright
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #    notice, this list of conditions and the following disclaimer in the
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #    documentation and/or other materials provided with the distribution.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #  * Neither the name of NVIDIA CORPORATION nor the names of its
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #    contributors may be used to endorse or promote products derived
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #    from this software without specific prior written permission.
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] #
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # [REVIEWED] # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


requested_metrics = [
    MetricRequest(
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        None,
        Importance.OPTIONAL,
        None,
    ),
]


def get_identifier():
    return "Memory"


def get_name():
    return "Memory Bottleneck"


def get_description():
    return "Detect bottlenecks arising from memory capabilities"


def get_section_identifier():
    return "SpeedOfLight"


def get_parent_rules_identifiers():
    return ["SOLBottleneck"]


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    speedup_metrics = dict()
    for name, metric in metrics.items():
        if metric is not None:
            speedup_metrics[name] = metric.value()

    fe.send_dict_to_children(speedup_metrics)
