# MLPerf Tiny benchmark v1.1

## Automation and reproducibility study

The [MLCommons Task Force on Automation and Reproducibility](https://github.com/mlcommons/ck/blob/master/docs/taskforce.md),
led by the [cTuning foundation](https://cTuning.org) and [cKnowledge.org](https://cKnowledge.org),
is developing a [free and open-source collaboration platform (MLCommons CK)](https://github.com/mlcommons/ck).

Our goal is to help users run all MLPerf benchmarks out of the box on any software and hardware from any vendor,
automate repetitive tasks and optimization experiments, reproduce results, and synthesize Pareto-optimal end-to-end applications
based on MLPerf results and user requirements including accuracy, performance, energy, size and costs.

This study demonstrates our work in progress to automate and reproduce one of the past 
[MLPerf Tiny submissions]( https://github.com/mlcommons/tiny_results_v1.0/blob/main/closed/OctoML ).

You can rerun our experiments following [this tutorial](https://github.com/mlcommons/ck/blob/master/docs/tutorials/reproduce-mlperf-tiny.md) 
in an automated and reproducible way with the help of the [MLCommons CM automation language](https://github.com/mlcommons/ck) 
and the [MLCommons CK playground](https://access.cKnowledge.org).
You can also reuse CM automations to setup EEMBC Energy runner, MLPerf Tiny benchmark and related data sets on your platform
using this [automation tutorial](https://github.com/mlcommons/ck/blob/master/docs/tutorials/automate-mlperf-tiny.md).

You can visualize and compare all MLPerf Tiny results using this [public CK dashboard](https://x.cknowledge.org/playground/?action=experiments&tags=mlperf-tiny)
or run this platform locally and privately to collaborate with your partners and users before submission. 
The simple GUI also allows you to set constraints and add derived metrics such as power efficiency and energy vs voltage
as shown in the following example for [image classification]( https://cKnowledge.org/mlperf-tiny-gui ):

![image](https://github.com/mlcommons/submissions_tiny_v1.1/assets/4791823/650b0aa3-17a9-4153-bfe4-547427bb240b)

## The next steps

This is an on-going community effort - join our [Discord server](https://discord.gg/JjWNWXKxwT) for the MLCommons task force on automation and reproducibility.
You will get free help to add support for your software, hardware, models and data sets to the MLCommons CM automation language.
This, in turn, will help you join the [upcoming challenges](https://access.cknowledge.org/playground/?action=challenges)
for external researchers, students and companies to reproduce benchmarks and optimize applications 
for your software/hardware stacks.
