# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("smsp__pcsamp_sample_count"),
    MetricRequest("smsp__pcsamp_interval_cycles"),
    MetricRequest("gpc__cycles_elapsed.max"),
]


def get_identifier():
    return "PCSamplingData"


def get_name():
    return "PC sampling data"


def get_description():
    return "PC sampling data"


def get_section_identifier():
    return "SourceCounters"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    if action.workload_type() != NvRules.IAction.WorkloadType_KERNEL:
        return

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    sample_count = metrics["smsp__pcsamp_sample_count"].value()
    interval_cycles = metrics["smsp__pcsamp_interval_cycles"].value()
    kernel_duration = metrics["gpc__cycles_elapsed.max"].value()

    if sample_count == 0:
        message = "Sampling metrics were enabled, but no samples could be collected for this kernel."

        if interval_cycles >= kernel_duration:
            message += "Note that the kernel duration is shorter than the sampling interval."
        fe.message(NvRules.MsgType.WARNING, message, "No Samples")
