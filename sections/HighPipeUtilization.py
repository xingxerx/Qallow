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
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser
from TableBuilder import PipelineTableBuilder

requested_metrics = [
    MetricRequest("device__attribute_compute_capability_major", "cc_major"),
    MetricRequest("device__attribute_compute_capability_minor", "cc_minor"),


    # Active cycles pipelines
    MetricRequest("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),

    # Instruction executed pipelines
    MetricRequest("sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fp64_op_dmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_fp64_op_fp64.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tc.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_subpipe_dmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tensor_op_imma_v2.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tma.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_tmem.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_workid.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    MetricRequest("sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed", None, Importance.OPTIONAL, None, False),
    # Additional metrics
    MetricRequest("smsp__issue_active.avg.per_cycle_active", "issue_active", Importance.OPTIONAL, None, False),
]


def get_identifier():
    return "HighPipeUtilization"

def get_name():
    return "High Pipe Utilization"

def get_description():
    return "High pipe utilization bottleneck analysis"

def get_section_identifier():
    return "ComputeWorkloadAnalysis"

def get_parent_rules_identifiers():
    return ["Compute"]

def get_estimated_speedup(max_utilization_ac):
    improvement_local = 1 - (max_utilization_ac / 100)

    speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
    improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent

def get_max_pipeline(pipelines, metrics):
    max_utilization = 0.0
    max_pipe = None

    cc = metrics["cc_major"].value() * 10 + metrics["cc_minor"].value()

    for pipe in pipelines:
        if pipe.cc_min is not None and cc < pipe.cc_min:
            continue
        if pipe.cc_max is not None and cc > pipe.cc_max:
            continue
        metric_name = pipe.metric
        if metrics[metric_name] is not None:
            value = metrics[metric_name].value()
            if value > max_utilization:
                max_utilization = value
                max_pipe = pipe

    return (max_pipe, max_utilization)


class Pipeline:
    def __init__(self, name, cc_min, cc_max, metric, description = None):
        self.name = name
        self.cc_min = cc_min
        self.cc_max = cc_max
        self.metric = metric + ".avg.pct_of_peak_sustained_elapsed"
        self.description = description

    def get_description(self, metrics):
        return self.description


class CompositePipeline(Pipeline):
    def __init__(self, name,  cc_min, cc_max, metric, description, sub_pipelines):
        super().__init__(name,  cc_min, cc_max, metric, description)
        self.sub_pipelines = sub_pipelines

    def get_description(self, metrics):
        description = self.description

        max_pipe, _ = get_max_pipeline(self.sub_pipelines, metrics)
        if max_pipe is not None:
            description += ". It's dominated by its {} sub-pipeline".format(max_pipe.name)

        return description


class SharedPipeline(CompositePipeline):
    def __init__(self, name,  cc_min, cc_max, metric, sub_pipelines):
        super().__init__(name,  cc_min, cc_max, metric, None, sub_pipelines)

    def get_description(self, metrics):
        cc = metrics["cc_major"].value() * 10 + metrics["cc_minor"].value()

        descriptions = {
            75 : ". It executes 16-bit floating point and tensor operations",
            80 : ". It executes 64-bit floating point and tensor operations",
            90 : ". It executes 64-bit floating point and tensor operations",
            100 : ". It executes 64-bit floating point and tensor operations",
            110 : ". It executes 64-bit floating point and tensor operations",
            120 : ". It executes 64-bit floating point and tensor operations",
            121 : ". It executes 64-bit floating point and tensor operations",
        }

        description = "is the logical sum of several other pipelines which can't achieve full utilization on their own"

        if cc in descriptions:
            description += descriptions[cc]

        self.description = description
        description = super().get_description(metrics)

        return description




def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    cc = metrics["cc_major"].value() * 10 + metrics["cc_minor"].value()

    fe.send_dict_to_children({
        "fp32_pipeline_utilization_pct": metrics["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"].value(),
        "fp64_pipeline_utilization_pct": metrics["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"].value(),
    })

    # Special handling for Ada GPU meric names in PerfWorks
    sm__pipe_tensor_cycles_active = "sm__pipe_tensor_cycles_active"
    sm__pipe_tensor_op_hmma_cycles_active = "sm__pipe_tensor_op_hmma_cycles_active"
    sm__pipe_tensor_op_imma_cycles_active = "sm__pipe_tensor_op_imma_cycles_active"
    sm__inst_executed_pipe_tensor_op_hmma = "sm__inst_executed_pipe_tensor_op_hmma"
    sm__inst_executed_pipe_tensor_op_imma = "sm__inst_executed_pipe_tensor_op_imma"
    if cc == 89:
        sm__pipe_tensor_cycles_active = "sm__pipe_tensor_cycles_active_v2"
        sm__pipe_tensor_op_hmma_cycles_active = "sm__pipe_tensor_op_hmma_cycles_active_v2"
        sm__pipe_tensor_op_imma_cycles_active = "sm__pipe_tensor_op_imma_cycles_active_v2"
        sm__inst_executed_pipe_tensor_op_hmma = "sm__inst_executed_pipe_tensor_op_hmma_v2"
        sm__inst_executed_pipe_tensor_op_imma = "sm__inst_executed_pipe_tensor_op_imma_v2"

    # Active cycles pipelines
    # These are based on the number of cycles the pipeline was active.
    # They take the rates of different instructions executing on the pipeline into account.
    # We use these to categorize the overall compute pipeline utilization.
    ac_pipelines = {
        Pipeline("ALU",                     None, None, "sm__pipe_alu_cycles_active", "executes integer and logic operations"),
        Pipeline("FMA",                     None, None, "sm__pipe_fma_cycles_active", "executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations"),
        Pipeline("FP64",                    None, None, "sm__pipe_fp64_cycles_active", "executes 64-bit floating point operations"),
        SharedPipeline("Shared",            None, None, "sm__pipe_shared_cycles_active",
            [
                Pipeline("FP64",            None, None, "sm__pipe_fp64_cycles_active"),
                Pipeline("Tensor (FP)",      100, None, "sm__pipe_tensor_subpipe_hmma_cycles_active"),
                Pipeline("Tensor (INT)",     100, None, "sm__pipe_tensor_subpipe_imma_cycles_active"),
                Pipeline("Tensor (DP)",      100,  100, "sm__pipe_tensor_subpipe_dmma_cycles_active"),
                Pipeline("Tensor (FP)",     None,   90, sm__pipe_tensor_op_hmma_cycles_active),
                Pipeline("Tensor (INT)",      72,   90, sm__pipe_tensor_op_imma_cycles_active),
                Pipeline("Tensor (DP)",       80,   80, "sm__pipe_tensor_op_dmma_cycles_active"),
                Pipeline("Tensor (DP)",       90,   90, "sm__pipe_tensor_op_dmma_cycles_active"),
            ]),
        Pipeline("TC",                       100,  100, "sm__pipe_tc_cycles_active", "executes Tensor Core (UTCBAR, UTCCP, UTC*MMA, UTCSHIFT and UTC*SWS) operations"),
        Pipeline("TC",                       110,  110, "sm__pipe_tc_cycles_active", "executes Tensor Core (UTCBAR, UTCCP, UTC*MMA, UTCSHIFT and UTC*SWS) operations"),
        CompositePipeline("Tensor",         None, None, sm__pipe_tensor_cycles_active, "is the logical aggregation of individual tensor pipelines",
            [
                Pipeline("Tensor (FP)",      100, None, "sm__pipe_tensor_subpipe_hmma_cycles_active"),
                Pipeline("Tensor (INT)",     100, None, "sm__pipe_tensor_subpipe_imma_cycles_active"),
                Pipeline("Tensor (DP)",      100,  100, "sm__pipe_tensor_subpipe_dmma_cycles_active"),
                Pipeline("Tensor (FP)",     None,   90, sm__pipe_tensor_op_hmma_cycles_active),
                Pipeline("Tensor (INT)",      72,   90, sm__pipe_tensor_op_imma_cycles_active),
                Pipeline("Tensor (DP)",       80,   80, "sm__pipe_tensor_op_dmma_cycles_active"),
                Pipeline("Tensor (DP)",       90,   90, "sm__pipe_tensor_op_dmma_cycles_active"),
            ]
        ),
        Pipeline("TMA",                       90, None, "sm__pipe_tma_cycles_active", "executes Tensor Memory Accelerator (TMA) operations"),
        Pipeline("TMEM (Tensor Memory)",     100,  100, "sm__mem_tensor_cycles_active", "increments for LDT(M), STT(M), UTCCP, UTCMMA and UTCSHIFT operations"),
        Pipeline("TMEM (Tensor Memory)",     110,  110, "sm__mem_tensor_cycles_active", "increments for LDT(M), STT(M), UTCCP, UTCMMA and UTCSHIFT operations"),
    }

    # Instruction executed pipelines
    # They do not account for any variation in instruction latencies for this pipeline.
    # We use these to understand the active cycles results in more detail.
    inst_pipelines = {
        Pipeline("ADU",                     None, None, "sm__inst_executed_pipe_adu"),
        Pipeline("ALU",                     None, None, "sm__inst_executed_pipe_alu", "executes integer and logic operations"),
        Pipeline("CBU",                     None, None, "sm__inst_executed_pipe_cbu"),
        Pipeline("FMA",                     None, None, "sm__inst_executed_pipe_fma", "executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations"),
        Pipeline("FP16",                    None,   80, "sm__inst_executed_pipe_fp16", "executes 16-bit floating point operations"),
        Pipeline("FMA (FP16)",                86, None, "sm__inst_executed_pipe_fma_type_fp16", "executes 16-bit floating point operations"),
        Pipeline("FP64",                    None, None, "sm__inst_executed_pipe_fp64", "executes 64-bit floating point operations"),
        Pipeline("FP64 (DMMA)",               86, None, "sm__inst_executed_pipe_fp64_op_dmma", "executes DMMA operations"),
        Pipeline("FP64 (FP64)",               86, None, "sm__inst_executed_pipe_fp64_op_fp64", "executes non-DMMA 64-bit floating point operations"),
        Pipeline("LSU",                     None, None, "sm__inst_executed_pipe_lsu", "executes load/store memory operations"),
        Pipeline("TC",                       100,  100, "sm__inst_executed_pipe_tc", "executes Tensor Core (UTCBAR, UTCCP, UTC*MMA, UTCSHIFT and UTC*SWS) operations"),
        Pipeline("TC",                       110,  110, "sm__inst_executed_pipe_tc", "executes Tensor Core (UTCBAR, UTCCP, UTC*MMA, UTCSHIFT and UTC*SWS) operations"),
        Pipeline("Tensor (FP)",              100, None, "sm__inst_executed_pipe_tensor_subpipe_hmma", "executes 16-bit floating point tensor operations"),
        Pipeline("Tensor (INT)",             100, None, "sm__inst_executed_pipe_tensor_subpipe_imma", "executes 4/8-bit integer tensor operations"),
        Pipeline("Tensor (DP)",              100,  100, "sm__inst_executed_pipe_tensor_subpipe_dmma", "executes 64-bit floating point tensor operations"),
        Pipeline("Tensor (DP)",               80,   80, "sm__inst_executed_pipe_tensor_op_dmma", "executes 64-bit floating point tensor operations"),
        Pipeline("Tensor (DP)",               90,   90, "sm__inst_executed_pipe_tensor_op_dmma", "executes 64-bit floating point tensor operations"),
        Pipeline("Tensor (Warp Group)",       90,   90, "sm__inst_executed_pipe_tensor_op_gmma", "executes warp group tensor operations"),
        Pipeline("Tensor (FP)",             None,   90, sm__inst_executed_pipe_tensor_op_hmma, "executes 16-bit floating point tensor operations"),
        Pipeline("Tensor (INT)",              72,   90, sm__inst_executed_pipe_tensor_op_imma, "executes 4/8-bit integer tensor operations"),
#endif
        Pipeline("TEX",                     None, None, "sm__inst_executed_pipe_tex", "executes texture/surface operations"),
        Pipeline("TMA",                       90, None, "sm__inst_executed_pipe_tma", "executes Tensor Memory Accelerator (TMA) operations"),
        Pipeline("TMEM (Tensor Memory)",    None, None, "sm__inst_executed_pipe_tmem", "executes Tensor Memory (FENCE.VIEW.ASYNC.T, LDT(M) and STT(M)) operations"),
        Pipeline("Uniform",                   75, None, "sm__inst_executed_pipe_uniform"),
        Pipeline("WorkID",                   100, None, "sm__inst_executed_pipe_workid", "executes UGETNEXTWORKID operations"),
        Pipeline("XU",                      None, None, "sm__inst_executed_pipe_xu"),
    }

    # several thresholds used to provide guidance
    low_utilization_threshold = 20
    high_utilization_threshold = 60
    bottleneck_utilization_threshold = 80

    # get the dominant active cycles-based pipeline metric
    (max_pipe_ac, max_utilization_ac) = get_max_pipeline(ac_pipelines, metrics)
    if max_pipe_ac is not None:
        doc_msg = " See the @url:Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder@ or hover over the pipeline name to understand the workloads handled by each pipeline."
        inst_section_msg = " The @section:InstructionStats:Instruction Statistics@ section shows the mix of executed instructions for this workload."

        stall_msg = ""
        if metrics["issue_active"] is not None:
            issue_active = metrics["issue_active"].value()
            if issue_active < 0.8:
                stall_msg = " Check the @section:WarpStateStats:Warp State Statistics@ section for which reasons cause warps to stall."

        # compare the active cycles-based pipeline utilization against various thresholds to categorize the performance and provide guidance
        if max_utilization_ac < low_utilization_threshold:
            message = "All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps per scheduler."
            message += " Check the @section:LaunchStats:Launch Statistics@ and @section:SchedulerStats:Scheduler Statistics@ sections for further details."
            msg_id = fe.message(NvRules.MsgType.OPTIMIZATION, message, "Low Utilization")

            speedup_type, speedup_value = get_estimated_speedup(max_utilization_ac)
            fe.speedup(msg_id, speedup_type, speedup_value)

            fe.focus_metric(
                msg_id,
                max_pipe_ac.metric,
                max_utilization_ac,
                NvRules.IFrontend.Severity_SEVERITY_HIGH,
                "Increase the utilization of the busiest pipeline (currently: {})".format(max_pipe_ac.name),
            )
        else:
            # descriptive info about the max active cycles pipe
            message = "{} is the highest-utilized pipeline ({:.1f}%) based on elapsed cycles in the workload, taking into account the rates of its different instructions.".format(max_pipe_ac.name, max_utilization_ac)
            pipe_info = max_pipe_ac.get_description(metrics)
            if pipe_info is not None:
                message += " It " + pipe_info + "."

            if max_utilization_ac < high_utilization_threshold:
                message_name = "Balanced"
                message += " It is well-utilized, but should not be a bottleneck."
                fe.message(NvRules.MsgType.OK, message, message_name)
            else:
                if max_utilization_ac < bottleneck_utilization_threshold:
                    message_name = "High Utilization"
                    message += " The pipeline is well-utilized, but might become a bottleneck if more work is added."
                else:
                    message_name = "Very High Utilization"
                    message += " The pipeline is over-utilized and likely a performance bottleneck."

                # get the dominant instruction executed-based pipeline, too
                (max_pipe_inst, max_utilization_inst) = get_max_pipeline(inst_pipelines, metrics)
                if max_pipe_inst is not None:
                    # descriptive info about the max instruction executed pipe
                    message += " Based on the number of executed instructions, the highest utilized pipeline ({:.1f}%) is {}.".format(max_utilization_inst, max_pipe_inst.name)
                    pipe_info_inst = max_pipe_inst.get_description(metrics)
                    if pipe_info_inst is not None:
                        message += " It " + pipe_info_inst + "."

                    # compare its utilization to the active cycles metric
                    utilization_diff = max_utilization_inst / max_utilization_ac
                    if utilization_diff < 0.3:
                        message += " Comparing the two, the overall pipeline utilization appears to be caused by high-latency instructions."
                    elif utilization_diff > 0.7:
                        message += " Comparing the two, the overall pipeline utilization appears to be caused by frequent, low-latency instructions."

                message += doc_msg + inst_section_msg + stall_msg
                msg_id = fe.message(NvRules.MsgType.OPTIMIZATION, message, message_name)


                fe.focus_metric(
                    msg_id,
                    max_pipe_ac.metric,
                    max_utilization_ac,
                    NvRules.IFrontend.Severity_SEVERITY_DEFAULT,
                    "Try to decrease the utilization of the busiest pipeline (currently: {})".format(max_pipe_ac.name),
                )
