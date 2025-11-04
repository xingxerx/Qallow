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
