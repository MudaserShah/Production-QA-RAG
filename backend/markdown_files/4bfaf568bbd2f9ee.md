<!-- page:1 -->

Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning
Ruoyu Qin†♢ Weiran He† Weixiao Huang† Yangkun Zhang† Yikai Zhao†
Bo Pang† Xinran Xu† Yingdi Shan♢ Yongwei Wu♢ Mingxing Zhang♢1
†Moonshot AI ♢Tsinghua University
Abstract
Reinforcement Learning (RL) has emerged as a critical tech-
nique for advancing modern Large Language Models (LLMs),
yet existing synchronous RL systems face severe performance
bottlenecks. The rollout phase, which dominates end-to-end
iteration time, suffers from substantial long-tail latency and
poor resource utilization due to inherent workload imbalance.
We present SEER, a novel context learning RL system that ad-
dresses these challenges through a key observation: requests
sharing the same prompt exhibit strong similarities in output
lengths and response patterns. Leveraging this insight, SEER
introduces three coordinated techniques: (1) divided rollout
for dynamic load balancing, (2) context-aware scheduling to
mitigate long-tail request delays, and (3) adaptive grouped
speculative decoding to accelerate generation. These mecha-
nisms work in concert to markedly reduce long-tail latency
and improve resource efficiency during rollout. Evaluations
on production-grade RL workloads demonstrate that SEER
achieves up to 2.04× end-to-end rollout throughput improve-
ment compared to the state-of-the-art synchronous RL sys-
tems, while notably reducing long-tail latency by 72–94%.
1 Introduction
Reinforcement Learning (RL) has become a cornerstone in
the development of state-of-the-art Large Language Mod-
els (LLMs), enabling significant breakthroughs in complex
reasoning and problem-solving capabilities [9, 37, 38]. The it-
erative RL training process alternates between arollout phase
for data generation and atraining phasefor updating model
parameters. However, the rollout phase consistently emerges
as the dominant bottleneck, consuming approximately 80%
of the total iteration time (see Table 1). Therefore, improving
the efficiency of the rollout phase represents one of the most
pressing challenges in modern LLM development.
The primary challenge in RL rollout arises from severe re-
source inefficiency driven by the increasing demand for long-
1 Corresponding to zhang_mingxing@mail.tsinghua.edu.cn.
Table 1: Time distribution across RL training phases for dif-
ferent workloads. Detailed configurations given in §4.1.
Rollout Training Weight Update
Moonlight [24] 84% 14% 2%
Qwen2-VL-72B [42] 63% 31% 6%
Kimi-K2 [37] 87% 10% 3%
generation capabilities, especially in tasks requiring complex
chain-of-thought (CoT) reasoning. These workloads create
two fundamental bottlenecks. First, long-generation requests
exhibit highlyunpredictable and rapidly increasing mem-
ory footprints: a CoT request may begin with only a few
hundred megabytes of KVCache usage but expand to tens
of gigabytes as decoding progresses. This volatility forces
the system to shrink batch sizes dynamically or to preempt
running requests; both outcomes reduce hardware efficiency,
and preemptions are particularly costly because they trigger
expensive re-prefills, ultimately degrading rollout throughput.
Second, long-generation requests produce aheavy-tailed
distribution of output lengths, resulting in pronounced load
imbalance. Toward the end of a rollout iteration, only a small
number of disproportionately long-running requests remain
active, leaving most accelerators underutilized. The ineffi-
ciency is so severe that attempting to use idle nodes to accel-
erate these long-tail requests often yields little benefit and can
even harm performance due to increased inter-node commu-
nication overhead.
To improve hardware utilization, recent works have ex-
plored asynchronous rollout systems [5, 10, 11, 28, 32, 50],
which overlap the rollout and training phases. While this ap-
proach can reduce end-to-end iteration time, it comes at the
cost of algorithmic fidelity. By nature, these systems intro-
duce a degree of off-policy learning, as data generated with
model parameters from step i may be used to train the model
at step i+1 or beyond. Furthermore, asynchronous or non-
strictly synchronous [8, 52] RL systems often suffer from
distributional skew, where faster-to-generate short samples
1
arXiv:2511.14617v3  [cs.DC]  3 Apr 2026

---

<!-- page:2 -->

4FFS0OMJOF$POUFYU-FBSOJOH'JOFHSBJOFE4DIFEVMJOH
Inter-instance ImbalanceIntra-instance Imbalance
Instance 1Instance 2Instance 3
Timeline
Inter-instance Balance
Instance 1Instance 2Instance 3
TimelineKV-usageIntra-instance Balance
Context Sched.Dynamic Load Balance
Grouped SDTimeline
KV-usage100%
Preemption & Recompute
Divided Rollout
Timeline
100%
OOM
Saved Time
Request Group 1Request Group 2Request Group 3
Low Util.
Req. Length
Batch Size
(a) Group-level rollout.
4FFS0OMJOF$POUFYU-FBSOJOH'JOFHSBJOFE4DIFEVMJOH
Inter-instance ImbalanceIntra-instance Imbalance
Instance 1Instance 2Instance 3
Timeline
Inter-instance Balance
Instance 1Instance 2Instance 3
TimelineKV-usageIntra-instance Balance
Context Sched.Dynamic Load Balance
Grouped SDTimeline
KV-usage100%
Preemption & Recompute
Divided Rollout
Timeline
100%
OOM
Saved Time
Request Group 1Request Group 2Request Group 3
Low Util.
Req. Length
Batch Size
(b) SEER’s rollout.
Figure 1: Challenges and SEER’s solution for long-generation
rollout.
disproportionately populate early training batches [39]. These
off-policy effects can diminish final model performance and
complicate debugging and reproducibility [48]. Consequently,
synchronous (or “on-policy”) rollout remains critical in many
settings for ensuring methodical evaluation, reproducibility,
and strict adherence to the underlying algorithm’s assump-
tions. This paper, therefore, focuses on optimizing the syn-
chronous case, although we note that our techniques can also
be adapted to improve asynchronous settings.
Speculative decoding [17] (SD) presents another promising
direction for accelerating memory-bound generation, as its
parallel verification mechanism can utilize more computa-
tional resources to accelerate individual requests. However,
conventional SD methods struggle to simultaneously achieve
high draft accuracy and low draft overhead in RL settings,
where both the request workload and the target model itself
evolve dynamically. This motivates the need for adaptive SD
techniques tailored to the unique characteristics of RL rollout.
To address these challenges, we introduce SEER, a novel
system for synchronous RL rollout that dynamically exploits
contextual information within the workload to maximize re-
source efficiency. SEERis built upon a key observation: popu-
lar RL algorithms such as Group Relative Policy Optimiza-
tion (GRPO) [30] generate G (typically 8–16) responses per
prompt, andresponses within a group tend to exhibit sim-
ilar length profiles and recurring local token patterns,
which represent a rich structure that existing schedulers and
inference engines leave untapped. For instance, generating an
early “probe” response per prompt provides a strong, online
estimator of that prompt group’s remaining work (i.e., ex-
pected output length and KVCache footprint), enabling more
informed scheduling decisions than static heuristics. SEER
leverages this latent intra-group context to make three key
contributions:
1) Divided Rollout with Global KVCache: SEERdeparts
from conventional group-level scheduling, where all requests
within a prompt group are treated as a single, inseparable unit.
This traditional approach causes pronounced inter-instance
and intra-instance load imbalance. As illustrated in Figure 1,
SEERinstead performsDivided Rollout, splitting each group
not only into G independent requests but further into smaller
chunks that are scheduled incrementally. This fine-grained
decomposition lets the scheduler pack many short requests
together early in rollout to fully utilize VRAM, and later,
when long requests dominate, adjust concurrency based
on KVCache budgets to avoid preemptions. SEER’s global
scheduler continuously monitors KVCache usage across
instances and can migrate a request to a less-loaded worker
when its next chunk is scheduled. This migration is efficient
because SEERuses a global KVCache pool adapted from
Mooncake [29] and shared across all instances, eliminating
the cost of prefill recomputation.
2) Context-Aware Scheduling:SEERleverages a“speculative
request”from each GRPO group to estimate that group’s
remaining workload—specifically its likely generation length
and KVCache footprint. These lightweight online estimates
allow SEERto approximate a longest-job-first scheduling
policy that pairs long requests with short ones to maintain
dense batches throughout rollout. Experiments in §4.4.1 show
that this approach significantly reduces the time spent in the
long-tail phase by 89%.
3) Adaptive Grouped Speculative Decoding: To leverage
speculative decoding for rollout acceleration while overcom-
ing the acceptance rate collapse of traditional SD, SEER
introduces a context-learning-based speculative mechanism.
SEERdeploys a Distributed Grouped Draft Server (DGDS)
that maintains a Compressed Suffix Tree [44] (CST) for each
group, aggregating token sequences from all requests within
the same group. This approach creates a highly accurate,
dynamic “draft model” that is inherently synchronized
with the target model. Additionally, DGDS introduces a
Marginal-Benefit-Aware Adaptive Speculation policy that
balances overall throughput with the latency of high-priority
requests. Our ablation experiments in §4.4.2 demonstrate the
effectiveness of the SD components. Our adaptive grouped
speculative decoding outperforms multiple SD strategies,
achieving up to 1.3× performance improvement. Compared
to vanilla CST-based SD without group context, it increases
the mean acceptance length by 0.22.
We have implemented SEERand conducted an extensive
evaluation on 256 H800 GPUs across multiple production-
grade RL workloads, encompassing models ranging from tens
2

---

<!-- page:3 -->

0 20k 40k 60k 80k 100k
Output Length
0.0
0.2
0.4
0.6
0.8
1.0CDF Moonlight (mean: 22386)
Qwen2-VL-72B (mean: 7615)
Kimi-K2 (mean: 38959)
Figure 2: Distribution of output lengths during rollout across
three reasoning tasks.
of billions to over one trillion parameters. Our experiments
demonstrate that SEERconsistently improves end-to-end roll-
out throughput by 44–104% and cuts long-tail latency by
72–94% relative to a highly optimized synchronous baseline.
Through detailed ablations, we isolate the contributions of di-
vided rollout, context-aware scheduling, and adaptive grouped
speculative decoding, showing that each component provides
substantial and complementary gains. We further compare
SEERagainst state-of-the-art rollout optimization approaches,
including a non-strictly synchronous system [52], as well
as multiple speculative decoding strategies [17, 22, 27]. We
find that SEERachieves the best overall performance while
preserving strict on-policy training semantics.
2 Background and Motivation
Reinforcement learning (RL) for LLMs progresses through a
repeated iterative loop. In each iteration, the model generates
responses for a batch of prompts (rollout), evaluators assign
quality scores to those responses (reward computation), these
scores are transformed into supervision signals (experience
construction), and the model is updated accordingly (train-
ing). The updated parameters are then distributed to inference
workers to initiate the next iteration (weight update). A se-
ries of recent systems aim to accelerate this loop by improv-
ing coordination across stages. veRL [33] colocates rollout
and training to reduce data movement, RLHFuse [51] over-
laps reward computation with experience construction, and
Kimi-K2 [37] further optimizes checkpoint conversion and
weight distribution. However, Table 1 makes clear that such
non-rollout phase optimizations are insufficient: across three
representative production workloads, rollout alone occupies
63–87% of total iteration time, overwhelmingly dominating
all other stages.
This imbalance is structural and rooted in the nature of
training reasoning models. RL increasingly relies on chain-
of-thought (CoT) supervision, which encourages models to
produce longer and more detailed reasoning traces. As a re-
sult, rollout exhibits two defining characteristics: long average
output lengths and extremely high variance across requests.
Figure 2 shows that generations range from a few hundred
0 10 20 30 40 50 60 70
Time (min)
0
25
50
75
100KVCache Usage (%)
16 Instances (KVCache Usage) Avg Running Requests
0
100
200
Requests Per Instance
(a) KVCache utilization and average number of running requests
across instances.
0 10 20 30 40 50 60 70
Time (min)
0
50
100
150
200Number of Preemptions
(b) Request preemption count.
Figure 3: KVCache utilization, number of running requests,
and preemption count during a synchronous rollout phase of
the Qwen2-VL-72B task. In the early stage of rollout, insuffi-
cient KVCache capacity causes frequent request preemptions;
in the later stage, a small number of extremely long request
groups contribute to a long-tail period that accounts for nearly
half of the total rollout time.
tokens to as many as 96k tokens. Long average outputs exert
heavy pressure on memory because KVCache consumption
scales with request length, while the extreme variance pro-
duces severe long-tail effects: a small number of exceptionally
long requests monopolize GPU resources near the end of roll-
out. Figure 3 illustrates how these properties cause substantial
resource underutilization and imbalance, motivating the two
challenges discussed next.
2.1 Challenge #1: Request Scheduling
Dilemma
During the rollout process of long-CoT models, the KVCache
memory of requests undergoes dramatic changes, from neg-
ligible in the early phase to several gigabytes per request in
the final stage. This varying memory consumption creates
a dilemma in concurrency management. If concurrency is
not controlled, the increasing request length leads to memory
exhaustion, resulting in massiverequest preemption, where
the KVCache of some running requests is evicted to allow
a few requests to complete inference. This introduces huge
overhead for KVCache recomputation. Conversely, if con-
currency is controlled to avoid preemption, requests in their
early generation phase (with only a few hundred tokens) may
occupy the entire memory space for hundreds of seconds, lead-
ing to severeresource underutilization. Furthermore, this
3

---

<!-- page:4 -->

memory footprint is amplified by a factor of G in GRPO-like
algorithms, where G responses are generated for each prompt
within the same group, further exacerbating the inter-instance
imbalance.
To address the inter-instance imbalance caused by group-
level scheduling, a concurrent work Roll Flash [25] proposes
prompt replication, which splits request groups into inde-
pendent requests for separate scheduling, thereby resolving
the amplified imbalance caused by scheduling entire request
groups. However, this does not address the dilemma ofintra-
instanceconcurrent scheduling, and still suffers from uneven
request distribution across instances leading to inter-instance
imbalance. The root cause of these issues is treating individ-
ual (or groups of) requests as indivisible monolithic units,
even though rollout scenarios impose no strict latency con-
straints on individual requests. This persistent scheduling
dilemma motivates our design ofdivided rollout(detailed in
§3.2), which schedules requests at the chunk level with little
overhead, enabling fine-grained scheduling and dynamic load
balancing.
2.2 Challenge #2: Severe Long-tail Effect
The long-tail problem is another critical issue in rollout,
widely noted in prior work [8, 11, 52]. As shown in Figure 3a,
the long-tail phase can account for nearly 50% of the total
rollout time. This phenomenon arises from two factors: (1)
In GRPO-like algorithms, requests within the same group
tend to have similar lengths. Groups with extremely long aver-
age lengths form “monolithic” batches that cause severe load
imbalance across instances when scheduling is performed
at the group granularity. (2) Under memory constraints, re-
quests may be preempted or delayed, causing extremely long
requests to be blocked and further exacerbating tail latency.
To mitigate the long-tail effect, recent works [5, 10, 11, 32,
50] have proposedasynchronous RL. Unlike synchronous
methods, where all training experiences must originate from
the current policy iteration, asynchronous methods allow train-
ing on stale rollout data from previous policy iterations, intro-
ducingoff-policyness. Other works [8, 52], while nominally
maintaining synchrony, allow deferring a portion of requests
to subsequent iterations, thereby sacrificing iteration-level
consistency compared to strict synchronous RL systems. Al-
though asynchronous or non-strictly synchronous RL can
reduce the long-tail impact, many RL practitioners still prefer
synchronous training to achieve the best convergence and
maintain reward stability.
Speculative decoding (SD) [2, 6, 14, 17–20, 22, 27] offers
a promising approach to accelerate the long-tail phase while
preserving synchronous RL training guarantees. SD consists
of two stages:DraftandVerification. In the draft stage, a
draft model generates draft tokens, which are then verified in
parallel by the target model (i.e., the policy LLM) in the veri-
fication stage. Since parallel verification of n tokens by the
0 10 20 30 40 50 60 70 80 90
Prompt Index
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15 Response Index
 10000
20000
30000
40000
Output T oken Count
Figure 4: Length correlation within response groups. Each
column represents a prompt group in GRPO rollout, and each
cell corresponds to an individual response. The color intensity
indicates output length.
LLM is faster than serial generation ofn tokens due to reduced
memory access, SD can improve the generation speed for indi-
vidual requests. However, RL rollout scenarios present unique
challenges: batch sizes fluctuate dramatically, and the target
LLM undergoes continuous updates, causing rapid “model
drift”. Existing SD strategies suffer from either high draft
overhead or low draft token accuracy, resulting in marginal or
even negative performance gains in rollout scenarios.
2.3 Opportunity: Shared Contextual Informa-
tion in Group Sampling
In LLM reinforcement learning practice, the most widely
adopted algorithm is Group Relative Policy Optimization
(GRPO) [30]. GRPO performs group-based preference opti-
mization by sampling G candidate responses for each prompt,
computing rewards for these responses, and then normaliz-
ing the rewardswithinthe group to obtain advantages. Sev-
eral recent works [47, 49] refine GRPO to improve general-
ization, but they all preserve the core principle of generat-
ing multiple responses for the same prompt. Other efforts,
such as BroRL [12] and Knapsack RL [21], even scale G to
512 or more to enhance exploration. As a result, GRPO and
its variants establishgroup samplingas a dominant training
paradigm, in which each prompt is associated with a set of
responses generated by the same policy.
This group sampling paradigm naturally introduces strong
and stable contextual signals during rollout. Since all re-
sponses in a group are conditioned on the same prompt and
produced by the same model, they exhibit notable similar-
ity in semantics, structural templates, and generation length.
However, existing RL systems typically treat each prompt
group as a monolithic unit during rollout, overlooking the
potential to externalize and exploit such intra-group similarity
asshared contextual information. Our statistical analysis
shows that this shared context is sufficiently rich to support
both scheduling and inference optimizations.
First, at thelengthlevel, responses within the same group
have highly correlated generation lengths. Prior work [3,7,50]
has shown that response length is predictable from prompt
and model characteristics. Our measurements on real RL
workloads, summarized in Figure 4, further confirm that most
4

---

<!-- page:5 -->

Table 2: Mean acceptance length in n-gram speculative de-
coding with grouped pattern references under different draft
strategies. Linear generates one draft sequence per step, while
multi-path generates multiple candidate sequences with top-k
branching. Values represent mean acceptance length (includ-
ing bonus token).
Ref. Count Linear Multi-Path (k=2) Multi-Path (k=4)
n=0 (baseline) 1.70 1.77 1.85
n=1 2.04 2.14 2.25
n=5 2.32 2.44 2.59
n=15 2.53 2.69 2.85
prompt groups in GRPO rollout exhibit strong length correla-
tion. We observe that responses within the same group tend to
have similar lengths, forming visually consistent columns.
This length context can be obtained cheaply via specula-
tive sampling and used to inform global scheduling policies,
such as approximate longest-first scheduling, which prior-
itizes groups likely to produce very long responses. Such
length- and memory-aware scheduling is a promising direc-
tion for mitigating the long-tail effect while respecting device
memory constraints.
Second, at thepatternlevel, grouped responses share recur-
ring semantic and syntactic structures that can be exploited
to accelerate inference. Instead of relying solely on each re-
quest’s own history, we can aggregate previously generated
tokens from other responses in the same group into a shared
pattern dictionary (e.g., via compressed suffix trees) and use
it as an n-gram reference for speculative decoding. To quan-
tify this opportunity, we sample 20 prompt groups from a
Qwen2-VL-72B task and simulate n-gram speculative decod-
ing under different draft strategies. Table 2 reports the mean
acceptance length (including bonus tokens) for varying num-
bers of grouped references n and different drafting modes.
Compared to the self-referencing baseline with no grouped
references (n=0 ), incorporating even a small number of intra-
group references (e.g., n=1 or n=5 ) consistently increases
the mean acceptance length, and using more references (e.g.,
n=15 ) together with multi-path drafting yields the largest
gains. In the later stage of rollout, speculative decoding with
full grouped pattern references can improve the number of
accepted draft tokens by up to 119% compared to a baseline
that only uses per-request history.
In summary, group sampling in GRPO-like algorithms pro-
vides rich shared contextual information, both in terms of
length correlation and pattern similarity, that is currently un-
derutilized by existing RL systems. Our statistical data indi-
cates that this contextual signal can be harnessed to enable
length- and memory-aware scheduling and grouped specula-
tive decoding, offering a promising opportunity to mitigate
long-tail inefficiency and improve rollout throughput in syn-
chronous LLM RL training.
Request Buffer
…
CPU Draft ClientKVCachePooling with Hierarchical Storage
GPU
Context Manager
Context-aware Scheduler (§4.3)
Request FlowContext Flow
Grouped Draft Server (§4.4)
Async Batch API for ClientsDivided Rollout Scheduler (§4.2)Req. States
Scheduling
Inference Engine Pool
Spec. Req.Completed ChunkIn-flight ChunkUnscheduled ChunkAppend
Speculate…Instance N
KVCacheLoad/Offload
Sync
Draft Client
Instance 1DividedRequestsStreamResponses
Draft Client
Instance 2
Draft Client
Instance 3
………
Figure 5: The overview of SEER.
3 Design of SEER
3.1 Overview
SEERis a synchronous, colocated RL system designed to
substantially reduce rollout long-tail latency and improve
overall system throughput without compromising algorith-
mic fidelity. The broader pipeline remains standard: train-
ing uses Megatron [34] for distributed optimization, rollout
uses vLLM [16] for generation and an asynchronous reward
computation backend, and Moonshot Checkpoint Engine [1]
ensures rapid movement of updated model weights. SEER’s
contributions therefore layer onto a conventional architecture
without imposing additional integration overhead.
Within rollout, as motivated by the workload analysis in
§2.3, SEERintroduces a unifying idea that drives its three
innovations: Group-Aware Context Learning. SEERcontinu-
ously learns the intra-group shared properties and uses them
to guide scheduling and decoding. Length similarities be-
come signals for estimating whether upcoming generations
are likely to be long-running; pattern similarities reveal op-
portunities to construct group-level draft predictions that ac-
celerate decoding without a separate draft model. In addition
to these optimizations, SEERalso incorporates efficient mem-
ory management, load balancing, and asynchronous reward
computation.
To support this context-driven approach at scale, the roll-
out subsystem is organized into three tightly connected com-
ponents as illustrated in Figure 5. An Inference Engine
Pool hosts multiple model-serving instances backed by a dis-
tributed KVCache pool, enabling KVCache migration across
nodes without recomputation. A Request Buffer provides
a global view of all pending and in-progress rollout work, al-
lowing the system to coordinate concurrency precisely. Atop
these components, a logically centralized Context Manager
maintains group-level contextual views and drives scheduling
and draft generation decisions.
On this foundation, SEERintroduces three techniques
that address the underlying challenges of long-CoT rollout.
Divided Rollout (§3.2) resolves the concurrency–memory
5

---

<!-- page:6 -->

dilemma by breaking each request into smaller schedulable
units and enabling fine-grained placement across instances.
Context-Aware Scheduling (§3.3) then leverages learned
length context to approximate a longest-first policy that re-
duces tail latency. Finally, Adaptive Grouped Speculative
Decoding (§3.4) exploits shared token-pattern structure to
build speculative decoding at the group level and dynami-
cally adjusts draft lengths to maximize throughput. Despite
these performance enhancements, SEERmaintains logical
consistency across all stages of the synchronous RL pipeline,
thereby achieving an algorithmically lossless reinforcement
learning process.
3.2 Divided Rollout
As illustrated in Figure 1, conventional group-level request
scheduling mechanisms lead to severe load imbalance both
within and across inference instances. Divided Rollout ad-
dresses this core tension by rethinking what constitutes a
schedulable unit. Rather than binding any request group to
a single execution slot, SEERnot only decomposes it into
individual requests but further divides each request into a se-
quence of generation chunks, each representing a bounded
segment of progress. A request that would previously monop-
olize an instance for thousands of tokens becomes a pipeline
of shorter units that can be interleaved and redistributed as
the system’s load evolves. This change in granularity enables
continuous rebalancing: after each chunk completes, the next
chunk can be scheduled on whichever instance currently has
the most available memory and compute.
However, divided rollout poses significant challenges to the
KVCache system. When requests are rescheduled at the chunk
level, all KVCache will be recomputed, and this overhead can
even negate the benefits of load balancing. Although exist-
ing inference frameworks [4, 40] support offloading request
KVCache to DRAM within instances, inference instances
can generate tens of terabytes of KVCache during a single
rollout iteration, far exceeding the DRAM capacity of individ-
ual instances. Furthermore, due to the dynamic load balanc-
ing scheduling policy, requests may be rescheduled to other
instances, leading to cache misses. Overall, divided rollout
requires a globally shared, high-capacity cache system to sup-
port chunk-level request scheduling with minimal overhead.
To address this challenge, SEERbuilds upon Moon-
cake [29] to construct a globally shared KVCache pool dis-
tributed across inference nodes. The KVCache of all active
requests is stored in a hierarchical global storage spanning
DRAM and SSD, with RDMA enabling rapid KVCache trans-
fer between nodes. From the perspective of the upper-level
scheduling system, request chunk scheduling can be treated
as stateless, eliminating the need to consider KVCache distri-
bution.
What distinguishes Divided Rollout from prior offloading
or preemption-based approaches is that cache movement is
no longer a reactive measure taken only when memory pres-
sure becomes intolerable. Instead, since chunk-level work
units have well-defined memory footprints, the scheduler can
proactively place and move them to optimize global system
balance rather than merely avoiding out-of-memory events.
This shift from reactive eviction to proactive, mobility-aware
scheduling is the key novelty that allows SEERto resolve
the concurrency–memory dilemma that has long constrained
synchronous RL rollout. Compared with live migration sys-
tems for online serving (e.g., Llumnix [36]), it offers greater
scheduling flexibility due to the absence of strict per-request
latency constraints.
3.3 Context-Aware Scheduling
Long-CoT rollout inevitably runs into memory limits, which
means some requests must wait. When the scheduler cannot
anticipate which requests will run long, these delays occur
arbitrarily, creating unnecessary congestion in the tail. As
shown in §2.3, requests within the same GRPO group tend
to exhibit similar output lengths. SEERleverages this struc-
ture to predict which requests are likely to be long-running
and schedules them more intelligently. The goal is to approxi-
mate a longest-first scheduling (LFS) policy (known to reduce
tail latency) without needing to know true output lengths in
advance.
SEER’s key idea is to designate one request per group as a
speculative request. This request serves as an online probe:
by observing how quickly it completes during generation,
SEERcan infer the approximate length of the other requests in
the group. To surface these signals early, speculative requests
are placed in a high-priority path and scheduled according to a
shortest-first strategy (SFS). Because short requests complete
quickly while long ones linger, this “length filtering” step
exposes potential long-tail groups early in the rollout, giving
SEERtime to react before those long generations dominate
system resources.
Specifically, the scheduling process unfolds in three concep-
tual phases. First, speculative requests are executed promptly
so that length information can be gathered with minimal delay.
Second, the Context Manager maintains and continually up-
dates an estimated output length for each group. This estimate
is simply the maximum generation length observed among
completed requests in the group, an approach that naturally
converges toward the true length as more information be-
comes available. For groups in which no request has finished
yet, SEERconservatively assumes they may be long-tail cases
and initializes their estimate to the upper bound on generation
length. Third, with group-level length estimates in place, the
scheduler shifts to an approximate LFS policy: groups pre-
dicted to be long are prioritized so that their requests begin
making progress early, rather than being deferred until system
load is already high. The detailed scheduling algorithm is
provided in the appendix.
6

---

<!-- page:7 -->

Instance & Draft Client
……AsyncAppend
Local SpeculationInferenceEngine4 Embedded DraftLibrary
InstanceInstance
1 PeriodicFetch3
Global Aggregation2
Global Grouped Draft Server
Figure 6: The distributed grouped draft server.
SEERalso incorporates safeguards to mitigate inaccuracies
in early predictions. It occasionally schedules requests from
underserved groups to avoid starvation and updates group
length estimates conservatively to reflect the longest sample
observed so far. These mechanisms ensure stability even when
speculative signals are noisy. As shown later in §4.4.1, this
context-aware scheduling policy approaches the performance
of an oracle LFS scheduler that has perfect knowledge of
request lengths.
3.4 Adaptive Grouped Speculative Decoding
To further improve resource utilization during the rollout
phase, especially in the long-tail stage, SEERimplements
Adaptive Grouped Speculative Decoding, whichadaptively
adjusts draft lengths based on computational intensity and
exploits thegroupedpattern context (as analyzed in Table 2)
to enhance acceptance rates.
3.4.1 Challenges of Speculative Decoding in Rollout
Speculative decoding (SD) benefits from underutilized com-
putational resources when batch sizes are small, enabling
faster parallel verification compared to serial generation. How-
ever, rollout scenarios feature dynamically varying batch sizes
that start large and rapidly decrease as long requests domi-
nate. Fixed-length SD methods often incur excessive draft
or verification overhead, degrading overall throughput. We
now model SD throughput [17] in the rollout scenario to
identify the key factors affecting performance. Let α denote
the acceptance rate, where α=E(β) and β is the acceptance
probability at each position. Let γ represent the number of
draft tokens predicted by the draft model per step, B denote
the current batch size, D(B,γ) represent the forward time of
the draft model, and T(B,γ) represent the forward time of the
target model.
The expected number of tokens generated per request in
each forward is 1−αγ+1
1−α . The expected time for SD to generate
one token per request is:
TSD = (1−α)(D(B,γ) +T(B,γ))
1−α γ+1
SD provides a benefit when TSD is less than the forward time
of the target modelT(B,1).
When B is small, D(B,γ) +T(B,γ) is approximately equal
to T(B,1) , making SD clearly beneficial. However, whenB is
large, D(B,γ) becomes non-negligible and T(B,γ) increases
rapidly with γ as the target model becomes compute-bound,
potentially resulting in negative gains from SD. In rollout sce-
narios, B dynamically varies and can range from 1 to several
hundred. Conventional static SD strategies with fixed draft
lengths suffer from either underutilization of computational
resources with small γ or excessive verification overhead with
large γ, resulting in marginal or even negative performance
gains. Furthermore, to optimize SD performance, the draft
system should minimize D(B,γ) while maximizing α. In roll-
out scenarios dominated by large batches, conventional SD
methods suffer from either high D(B,γ) (e.g., using a sep-
arate draft model) or low α (e.g., naive n-gram methods).
Therefore, an efficient draft system specifically tailored to the
characteristics of rollout workloads is required.
3.4.2 Adaptive Speculation with Disaggregated Archi-
tecture
SEERintroduces the Distributed Grouped Draft Server
(DGDS), a disaggregated SD framework specifically designed
for rollout scenarios. As illustrated in Figure 6, DGDS aggre-
gates response pattern context across requests and instances
through an independent draft server, which asynchronously
distributes the context to the embedded draft client within
each inference instance. The core data structure of DGDS is
the Compressed Suffix Tree (CST), which enables sharing
pattern information across multiple sequences and provides
draft tokens with low complexity 1. Unlike previous suffix
tree methods [27, 40] that serialize CST updates with model
execution, which increases the draft time D(B,γ), DGDS em-
ploys a distributed master-worker architecture and adopts
asynchronous CST updates to minimize speculative decoding
latency in the critical path. The detailed workflow is provided
in the appendix.
To ensure that SD consistently provides positive gains
throughout the rollout process, it is necessary to dynami-
cally adjust γ based on the current batch size according to the
throughput model. Furthermore, as described in §3.3, requests
are classified into high-priority and low-priority categories.
1The complexity is O(p+s) , where p denotes the matching pattern length
andsdenotes the number of speculative tokens.
7

---

<!-- page:8 -->

Algorithm 1Marginal-Benefit-Aware Adaptive Speculation
Require: High-priority and low-priority batch sizesBh and Bl; per-
position acceptance probabilities β[1],β[2], . . .; maximum token
budget per request γmax; priority factor λ∈[1,∞) (e.g., λ=2 ).
Ensure:Draft token countsγ h andγ l.
1:B←B h +B l
2:γ ∗ ←argmin γ TSD(B,γ)▷optimal draft length for batch sizeB
3:Γ ∗ ←γ ∗ ·B▷total token budget
4:ifΓ ∗ <B h then
5:return(γ h =0,γ l =0)▷disable speculation
6:▷Allocate budget based on marginal benefit
7:γ h ←1,γ l ←0,remaining←Γ ∗ −B h
8:whileremaining>0do
9:benefit h ←B h ·(β[γ h]−β[γ h +1])
10:benefit l ←B l ·(β[γ l]−β[γ l +1])
11: if benefith >λ·benefit l and γh <γ max and remaining≥B h
then
12:γ h ←γ h +1▷allocate to high-priority
13:remaining←remaining−B h
14:else ifB l >0andγ l <γ max andremaining≥B l then
15:γ l ←γ l +1▷allocate to low-priority
16:remaining←remaining−B l
17:else
18:break▷cannot allocate further
19:return(γ h,γ l)
High-priority speculative requests are used to probe the length
distribution of requests within the same group and should
complete faster, thus requiring higher draft budgets. However,
acceptance rates decrease rapidly with increasing draft length,
making overly disparate budget allocations between high-
priority and low-priority requests wasteful. To address this
challenge, we propose aMarginal-Benefit-Aware (MBA)
Adaptive Speculationpolicy inspired by classical utility max-
imization and marginal-utility scheduling principles widely
used in resource allocation systems [15], which dynamically
balances overall throughput with the latency of high-priority
requests.
Algorithm 1 presents the MBA strategy. Using offline-
profiled TSD models and online-collected acceptance rates
and batch sizes, the algorithm determines draft lengths(γh,γ l)
for high-priority and low-priority requests. The algorithm is
invoked periodically during rollout, as request characteris-
tics remain relatively stable over short time intervals. Given
the draft lengths, the embedded draft client in DGDS gener-
ates the corresponding number of draft tokens based on the
grouped CSTs for each request. The basic single-path spec-
ulation algorithm follows SuffixDecoding [27], where each
candidate path is assigned a confidence score computed from
suffix probabilities. Building on this foundation, SEERuses
these scores to filter low-probability candidates and is capa-
ble of returning multiple candidate paths via a beam-search
mechanism.
For long-tail requests, adaptive grouped speculative decod-
Table 3: Model configurations and RL workload characteris-
tics.
Metric Moonlight Qwen2-VL-72B Kimi-K2
Model Size 32 GB 146 GB 1 TB
Total GPUs 32 128 256
GPUs per Instance 1 8 32
Reqs per Iter 3200 9600 6400
Group Size 8 16 8
Temperature 1.0 0.8 1.0
Max. Gen. Length 65536 40960 98304
Avg. Gen. Length 22386 7615 38959
ing offers two particular advantages. First, during the long-tail
stage, concurrency is minimal, allowing larger draft lengths
to increase the number of accepted tokens per request. SEER
also implements multi-path speculative decoding to further
improve the acceptance length in the long-tail stage. Second,
as more requests in the same group complete over time, the
CST aggregates richer contextual information. As demon-
strated in Table 2, this enables long-tail requests to achieve
substantially longer acceptance lengths.
4 Evaluation
In this section, we evaluate SEER’s performance advantages
during the rollout phase, particularly in improving throughput
and reducing tail latency (§4.2). In the ablation study (§4.3),
we provide a detailed analysis of how each technique in SEER
contributes to improving end-to-end performance. Finally, we
present extended studies (§4.4) on scheduling, speculative
decoding, and non-strictly synchronous RL to demonstrate
the effectiveness of our context-aware techniques.
4.1 Setup
Testbed.Our experimental infrastructure consists of 32
high-performance compute nodes, each equipped with
8×H800 GPUs, 224 CPU cores, 2TB DRAM, and 4TB
NVMe storage, providing sufficient hardware support for the
global KVCache pool and distributed grouped draft server.
For deployment, we adopt a strategy that balances task per-
formance with resource efficiency by configuring different
numbers of GPUs and parallelization strategies to serve mod-
els based on their size and architecture.
Models and Workloads.To validate the generalizability
of SEER’s design, we evaluate three models with diverse
sizes and output characteristics: Moonlight [24], Qwen2-VL-
72B [42], and Kimi-K2 [37]. All models are trained asreason-
ing modelswith chain-of-thought capabilities, where Moon-
light and Kimi-K2 are trained on mathematical datasets, while
8

---

<!-- page:9 -->

Qwen2-VL-72B is trained on language-vision-mixed reason-
ing tasks using the LLM-as-a-Judge [35] reward model. Their
configurations and RL workload characteristics are shown in
Table 3. We employ Qwen2-VL-72B with tensor parallelism
(TP8) and Kimi-K2 with data parallelism (DP32) and expert
parallelism (EP32).
Settings.We use the GRPO [30] algorithm in our experi-
ments. The group size (i.e., the number of requests per prompt)
follows RL training conventions unless otherwise specified:
we use a smaller group size (8) for mathematical problems and
a larger group size (16) for open-ended tasks with LLM-as-a-
Judge evaluation. For SEER’s SD hyperparameters described
in §3.4.2, we setγ max =8 andλ=2.
Baselines.We evaluate SEERagainst a set of strong and rep-
resentative baselines that capture the best available techniques
for synchronous rollout scheduling, skewness mitigation, and
speculative decoding.
(1) veRL[33]: veRL is a state-of-the-art synchronous RL
system that supports efficient colocation of training and
rollout. It provides a well-engineered baseline for mea-
suring SEER’s end-to-end improvements because it al-
ready incorporates optimized model execution pipelines
and a production-grade rollout subsystem.
(2) StreamRL-Oracle: Although StreamRL [50] is designed
as a disaggregated asynchronous RL framework, its pro-
posed Skewness-Aware Scheduling directly targets the
same long-tail latency issues that appear in synchronous
rollout. StreamRL’s scheduling relies on a small auxil-
iary model that predicts prompt lengths and uses those
predictions to perform bucketing and LFS-style schedul-
ing. To ensure afair and stringent comparison, we eval-
uate against StreamRL-Oracle, which uses the ground-
truth prompt lengths obtained from veRL’s rollout traces
rather than model predictions. This removes the con-
founding effects of prediction errors and reflects the best-
case performance achievable by the StreamRL schedul-
ing approach.
(3) Vanilla Speculative Decoding (SD): To benchmark
against modern decoding accelerators, we implement
strong speculative decoding baselines tailored to each
model family. For Moonlight, we use SuffixDecoding
[27] with a maximum draft length of γmax =16 . For
Qwen2-VL-72B, we adopt a dedicated Qwen2-7B-VL
draft model with γmax =3 . For Kimi-K2, we apply Multi-
Token Prediction (MTP) [22] with γmax =1 . Each of
these SD methods is integrated into both veRL and
StreamRL-Oracle pipelines, enabling direct comparisons
across scheduling regimes and decoding strategies.
To isolate the effect of scheduling and speculative mecha-
nisms, all baselines (including SEER) use a unified in-house
implementation of vLLM [16] as the inference engine. This
ensures that differences in performance arise solely from the
scheduling and decoding techniques under evaluation, not
from underlying infrastructure discrepancies.
Metrics.For the end-to-end experiments, we measure the
average rollout throughput across 5 iterations, which is de-
fined as the average number of output tokens generated per
second in each rollout iteration.
4.2 End-to-End Performance
In our end-to-end experiments, we conduct three RL tasks
based on the aforementioned models, with workload config-
urations detailed in Table 3. We provide a comprehensive
comparison of SEER’s end-to-end performance against dif-
ferent baselines in §4.2.1. To validate SEER’s effectiveness
in mitigating tail latency, we analyze the tail latency phe-
nomenon in our experiments and compare the tail latency
between SEERand the baseline system veRL in §4.2.2.
4.2.1 Rollout Throughput
We compare the throughput performance of SEERagainst
state-of-the-art baselines across different workloads and group
sizes, as shown in Figure 7. Despite significant variations in
model size and workload characteristics across different RL
tasks, SEERconsistently achieves substantial speedups across
all tasks, with throughput improvements ranging from 44%
to 104% over veRL.
SEERalso outperforms StreamRL-Oracle and multiple SD
strategies. StreamRL sets buckets with smaller concurrency
for long-request groups to minimize latency. However, it still
treats each request group as an atomic, non-preemptible unit
and cannot dynamically adjust instance loads at runtime. Con-
sequently, its performance is heavily dependent on the accu-
racy of the preset bucketing algorithm. When encountering
out-of-distribution workloads such as Kimi-K2, StreamRL-
Oracle even underperforms veRL, which uses a simple round-
robin scheduling strategy. For vanilla SD methods, although
we have implemented adaptive draft length, these SD methods
still suffer from either excessive overhead or low average ac-
ceptance length, resulting in inferior performance compared
to our adaptive grouped speculative decoding. A detailed
comparison is provided in §4.4.2.
Across different workloads, SEERachieves greater perfor-
mance improvements on memory-constrained tasks (Moon-
light and Qwen2-VL-72B), where SEER’s divided rollout mit-
igates load imbalance and context-aware scheduling reduces
long request delays. Furthermore, as the group size increases,
the load imbalance caused by veRL’s group-level schedul-
ing becomes more severe, leading to throughput degradation.
In contrast, SEERaddresses the monolithic request group
problem through divided rollout and dynamic load balancing,
and leverages group context to optimize both scheduling and
9

---

<!-- page:10 -->

Group Size=8 Group Size=1615
20
25
30
35Throughput (K tokens/s)
1.00x
1.16x
1.21x
1.62x
1.78x
1.00x
1.34x1.30x
1.56x
1.83x
Group Size=8 Group Size=1615
20
25
30
35
40
45
1.00x
1.14x
1.33x
1.46x
1.82x
1.00x
1.08x
1.43x
1.57x
2.04x
Group Size=8 Group Size=1615
20
25
30
35
1.00x
1.26x
0.99x
1.09x
1.48x
1.00x
1.22x
0.94x
1.02x
1.44x
veRL veRL w. SD StreamRL-Oracle StreamRL-Oracle w. SD Seer
(a) Moonlight (b) Qwen2-VL-72B (c) Kimi-K2
Figure 7: End-to-end rollout throughput of RL systems across different tasks and group sizes.
Moonlight Qwen2-VL-72B Kimi-K20
1000
2000
3000
4000
5000Rollout Time (s)
3910s
1817s
2181s
278s
3957s
2301s
1922s
123s
-84%
-94%
veRL (T otal Time)
veRL (T ail Time)
Seer (T otal Time)
Seer (T ail Time)
0
2000
4000
6000
8000
10000
12000
Rollout Time (s)
12202s
3144s
8267s
850s
-72%
Figure 8: Tail time and total time of three RL tasks.
0 5 10 15 20 25 30 35
Time (min)
0
25
50
75
100KVCache Usage (%)
16 Instances (KVCache Usage) Avg Running Requests
0
100
200
Requests Per Instance
Figure 9: KVCache utilization and average running requests
with SEERduring a rollout iteration of the Qwen2-VL-72B
task.
speculative decoding. As a result, when the group size in-
creases from 8 to 16, SEERachieves an average performance
improvement of 5%.
4.2.2 Long-Tail Time
To analyze SEER’s performance advantages, we examine tail
latency in the rollout process. We definetail requestsas the
last 10% of requests to complete in a synchronous system
rollout, andtail timeas the time spentsolelyprocessing these
tail requests. Figure 8 shows the tail time and total rollout
completion time, averaged across all iterations. The results
demonstrate severe tail latency for memory-constrained tasks
Table 4: Performance improvement breakdown across three
RL tasks. Context Sched. denotes context-aware scheduling
(§3.3), and Grouped SD denotes adaptive grouped speculative
decoding (§3.4).
Method Moonlight Qwen2-VL-72B Kimi-K2
Baseline 1.00 1.00 1.00
+ Divided Rollout 1.41×1.42×1.16×
+ Context Sched. 1.47×1.56×1.27×
+ Grouped SD 1.90×2.04×1.53×
like Moonlight and Qwen2-VL-72B, where the last 10% of
requests consume up to 50% of total time.
The tail latency problem stems from two primary causes.
First, request queuing and preemption lead to delayed schedul-
ing of long-output requests. Second, monolithic request
groups result in load imbalance across instances. We pro-
vide quantitative evidence using the Qwen2-VL-72B task on
veRL shown in Figure 3. In this rollout iteration, a total of
13,686 preemption events occurred. The last 5% of completed
requests have an average length in the top 15th percentile, yet
their average execution start time is at 42% of the total time.
The load imbalance across instances is even more pronounced:
the completion time difference between the earliest and lat-
est instances accounts for 70% of the total time, with each
instance idle for an average of 1,580 seconds, representing
37% of the total time. These scheduling and load imbalances
result in severe tail latency and throughput degradation.
By leveraging group-aware context learning and fine-
grained request scheduling, SEERsignificantly reduces tail
latency by 72% to 94%, thereby substantially improving sys-
tem throughput. Figure 9 illustrates the impact of these tech-
niques, showing that SEERsubstantially reduces the tail phase
duration compared to the baseline shown in Figure 3a.
10

---

<!-- page:11 -->

BaselineNo-ContextContext-Aware
Oracle
1.0
1.2
1.4
1.6Normalized Throughput
1.00x
1.42x
1.56x
1.63x
(a) Normalized throughput.
BaselineNo-ContextContext-Aware
Oracle
0.0
0.5
1.0Normalized T ail Time
1.00x
0.79x
0.11x
0.00x (b) Normalized tail latency.
Figure 10: Impact of length context on improving through-
put and reducing tail latency (defined in §4.2.2). No-Context
applies only divided rollout without using length context to
guide scheduling. Oracle obtains all output lengths in advance
and applies the LFS strategy.
4.3 Improvement Breakdown
We present a systematic ablation study to quantify the con-
tribution of each optimization component to SEER’s overall
performance. Table 4 demonstrates the cumulative end-to-end
speedup obtained by incrementally integrating each major
optimization component, where each set of experiments is
executed on the 5th rollout iteration.
First, SEER’s divided rollout mechanism enables dynamic,
fine-grained load balancing across instances, mitigating tail
latency caused by inter-instance load imbalance and reducing
preemption overhead within instances due to varying KV-
Cache memory consumption. This optimization yields sig-
nificant improvements for memory-constrained tasks, achiev-
ing up to 42% throughput improvement. Building upon di-
vided rollout, context-aware scheduling leverages learned
intra-group length distributions to guide scheduling, provid-
ing up to 14% additional throughput improvement. To further
address the computational underutilization inherent in tail
requests, SEERincorporates adaptive grouped speculative de-
coding, which contributes an additional 26–48% performance
improvement over the scheduling optimizations alone.
4.4 Extended Studies
4.4.1 Context-Aware Scheduling
To validate the effectiveness of length context, we conduct
an ablation study by varying the length information available
to SEER’s scheduler, as shown in Figure 10. Specifically, we
compare: (1) No-Context, which applies only divided rollout
without using length context to guide scheduling, and (2)
Oracle, which obtains the actual output lengths of all requests
in advance and replays the rollout iteration using these precise
lengths to guide scheduling.
While divided rollout significantly improves throughput
through dynamic load balancing, the tail latency problem
persists, with tail latency reduced by only 21% compared
to the baseline. In contrast, context-aware scheduling lever-
Moonlight Qwen2-VL-72B Kimi-K20.9
1.0
1.1
1.2
1.3
1.4
1.5Normalized Throughput
1.00x
1.18x
=1.67
1.26x
=1.89
1.00x
1.06x
=2.19
1.38x
=2.15
1.00x
1.22x
=1.74
 1.25x
=1.77
No SD
Vanilla SD
Seer's SD
Figure 11: Normalized throughput and mean acceptance
length (τ) of SD strategies across three tasks.
1 2 3 4 5
Iteration
20
25
30
35
40Throughput (k tokens/s)
+43%
Seer
Partial-Rollout
(a) Rollout throughput.
0-10 10-20 20-30 30-40 40-50
Output T oken Length (k tokens)
0.7
0.8
0.9
1.0
1.1Relative Ratio
Synchronous
Partial-Rollout (b) Output length distribution.
Figure 12: Rollout experiments comparing SEERand Partial
Rollout on the Qwen2-VL-72B workload.
ages length prediction information from speculative requests
to implement approximate longest-first scheduling, substan-
tially reducing tail latency by 89% compared to the baseline.
Compared to Oracle, context-aware scheduling achieves 96%
of its throughput performance. This demonstrates that de-
spite some performance degradation from prediction errors,
context-aware scheduling still provides substantial benefits.
4.4.2 Adaptive Grouped Speculative Decoding
As discussed in §3.4, the benefits of speculative decoding
during rollout depend on multiple factors including batch
size, draft model overhead, and draft accuracy. We conduct
ablation experiments on different SD strategies based on a
single rollout iteration executed on veRL. As shown in Fig-
ure 11, SEER’s adaptive grouped SD consistently outperforms
vanilla SD in throughput across all tasks, achieving up to
1.3× speedup. Furthermore, by leveraging group context,
our approach improves the mean acceptance length by 0.22
compared to the CST-based SD method [27], and exceeds
MTP’s performance. While vanilla SD with a small draft
model achieves slightly higher mean acceptance length than
our adaptive grouped SD, its excessive draft model overhead
results in the lowest overall throughput gain.
4.4.3 Non-Strictly Synchronous RL
Some RL training systems [8,38,52] employ non-strictly syn-
chronous RL, which defers or reschedules long-tail requests
11

---

<!-- page:12 -->

to subsequent iterations. While non-strictly synchronous RL
can improve resource utilization, it introduces some degree of
inconsistency with synchronous training algorithms. Partial
Rollout [38, 52] is a popular non-strictly synchronous method
that over-issues requests during rollout and terminates the
rollout phase once a predetermined number of requests are
completed, with remaining requests executed with high prior-
ity in the next rollout iteration. In this section, we compare
SEERagainst Partial Rollout using the Qwen2-VL-72B work-
load. Following the experimental setup from APRIL [52],
which applies Partial Rollout, we over-issue twice the number
of requests.
As shown in Figure 12a, SEERachieves 43% higher av-
erage throughput than Partial Rollout. This advantage stems
from two factors. First, in memory-constrained long-CoT
scenarios, Partial Rollout’s doubling of request volume exac-
erbates preemption issues, preventing it from fully leveraging
its higher degree of parallelism. Second, SEER’s adaptive
grouped speculative decoding further enhances throughput
for long-generation requests. Additionally, Figure 12b shows
the output length distributions of synchronous RL methods
and Partial Rollout. Compared to the synchronous method,
Partial Rollout generates a significantly lower proportion of
requests with longer output lengths. This bias may negatively
impact RL training and degrade model performance.
5 Related Work
RL Frameworks for LLM Post-Training.Numerous open-
source RL frameworks [13, 31, 33, 41, 43, 46, 53] have been
proposed to achieve both efficiency and usability, providing
robust infrastructure support for both research and produc-
tion RL workloads. OpenRLHF [13] and veRL [33] seam-
lessly integrate training and inference engines, supporting
various parallelism strategies to improve RL training effi-
ciency. ROLL [43] targets large-scale RL training optimiza-
tion, enabling flexible resource allocation and heterogeneous
task scheduling. Slime [53] balances high performance with
flexibility, supporting highly customizable data generation
pipelines. These works primarily focus on overall RL training
workflow orchestration, while the long-tail problem in the
rollout phase remains unaddressed.
On-Policy RL Optimization.Some efforts have been made
to optimize the RL workflow at both the system and algo-
rithmic levels. RealHF [26] dynamically reallocates model
parameters and memory budgets to optimize utilization. RL-
HFuse [51] proposes stage fusion to overlap reward and expe-
rience computation with the rollout long tail, thereby improv-
ing resource utilization. RLBoost [45] utilizes preemptible
fragmented GPU resources to accelerate rollout at low cost.
Nevertheless, these approaches do not adequately address the
long-tail latency problem. For the long-tail problem, spec-
ulative decoding [2, 6, 14, 17–20, 22, 27] is regarded as an
effective optimization technique. However, these methods
suffer from either high draft overhead or low accuracy in RL
rollout scenarios due to dynamically changing batch sizes and
continuously evolving target LLM weights. RhymeRL [11]
and SPEC-RL [23] propose utilizing historical sequences gen-
erated from previous RL iterations as references to accelerate
decoding. However, in most state-of-the-art model rollout sce-
narios, different iterations sample different data to improve
generalization, leaving no historical sequences to reuse. In
contrast, SEERexploits the similarity among requests within
the same group, eliminating the need to rely on repeated
sequences across iterations. Meanwhile, SEERemploys an
adaptive speculation policy based on a disaggregated archi-
tecture, achieving enhanced accuracy while maintaining low
draft overhead.
Off-Policy RL Optimization.Recently, many works [5, 8,
10, 11, 25, 28, 32, 50, 52] have proposed RL workflows that
do not maintain complete logical synchronization, trading
a degree of off-policy behavior for improved RL efficiency.
StreamRL [50], Areal [5], RhymeRL [11], and Laminar [32]
propose asynchronous training approaches to improve re-
source utilization. In asynchronous RL, training and rollout
are completely decoupled: rollout workers continuously gen-
erate new outputs without waiting, while training workers
update the model whenever a batch of samples is collected.
Other works relax the synchronization constraints of syn-
chronous RL to achieve speedup, referred to as non-strictly
synchronous RL. Partial Rollout [38, 52] defers long-tail re-
quests to continue execution in the next rollout iteration. Roll-
Packer [8] proposes tail batching, which schedules long-tail
requests into a few designated long iterations, effectively re-
ducing GPU idle time. While these techniques improve RL
efficiency, the off-policyness inherent in these algorithms in-
troduces risks of instability and accuracy degradation, limiting
their applicability in research and production environments
where algorithmic fidelity is critical. SEERsignificantly re-
duces rollout tail latency through fine-grained scheduling of
group-aware context learning within the rollout phase, while
maintaining consistency with on-policy algorithms.
6 Conclusion
In this paper, we present SEER, a synchronous RL system that
accelerates the rollout process through group-aware context
learning. With divided rollout, a fine-grained and dynamically
load-balanced scheduling approach, SEERleverages the simi-
larity among requests within the same group in GRPO-like
algorithms to achieve efficient scheduling and speculative de-
coding, while strictly maintaining consistency with on-policy
RL algorithms. Experimental results demonstrate that SEER
achieves up to 2.04× throughput improvement and up to 94%
reduction in long-tail latency compared to the current state-
of-the-art RL framework.
12

---

<!-- page:13 -->

References
[1] Moontshot AI. Checkpoint engine. https://github.
com/MoonshotAI/checkpoint-engine, 2025.
[2] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng,
Jason D Lee, Deming Chen, and Tri Dao. Medusa: Sim-
ple llm inference acceleration framework with multi-
ple decoding heads.arXiv preprint arXiv:2401.10774,
2024.
[3] Ke Cheng, Wen Hu, Zhi Wang, Peng Du, Jianguo Li,
and Sheng Zhang. Enabling efficient batch serving for
lmaas via generation length prediction. In2024 IEEE In-
ternational Conference on Web Services (ICWS), pages
853–864. IEEE, 2024.
[4] LMSYS Corp. Sglang. https://github.com/
sgl-project/sglang, 2025.
[5] Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu
Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Ji-
ashu Wang, et al. Areal: A large-scale asynchronous
reinforcement learning system for language reasoning.
arXiv preprint arXiv:2505.24298, 2025.
[6] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang.
Break the sequential dependency of llm inference using
lookahead decoding.arXiv preprint arXiv:2402.02057,
2024.
[7] Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Sto-
ica, and Hao Zhang. Efficient llm scheduling by learning
to rank.Advances in Neural Information Processing
Systems, 37:59006–59029, 2024.
[8] Wei Gao, Yuheng Zhao, Dakai An, Tianyuan Wu, Lunxi
Cao, Shaopan Xiong, Ju Huang, Weixun Wang, Siran
Yang, Wenbo Su, et al. Rollpacker: Mitigating long-tail
rollouts for fast, synchronous rl post-training.arXiv
preprint arXiv:2509.21009, 2025.
[9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shi-
rong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes rea-
soning in llms through reinforcement learning.Nature,
645(8081):633–638, 2025.
[10] Zhenyu Han, Ansheng You, Haibo Wang, Kui Luo,
Guang Yang, Wenqi Shi, Menglong Chen, Sicheng
Zhang, Zeshun Lan, Chunshi Deng, et al. Asyncflow:
An asynchronous streaming rl framework for efficient
llm post-training.arXiv preprint arXiv:2507.01663,
2025.
[11] Jingkai He, Tianjian Li, Erhu Feng, Dong Du, Qian Liu,
Tao Liu, Yubin Xia, and Haibo Chen. History rhymes:
Accelerating llm reinforcement learning with rhymerl.
arXiv preprint arXiv:2508.18588, 2025.
[12] Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Har-
chaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun
Yang, Jan Kautz, et al. Brorl: Scaling reinforcement
learning via broadened exploration.arXiv preprint
arXiv:2510.01180, 2025.
[13] Jian Hu, Xibin Wu, Zilin Zhu, Weixun Wang, Dehao
Zhang, Yu Cao, et al. Openrlhf: An easy-to-use, scalable
and high-performance rlhf framework.arXiv preprint
arXiv:2405.11143, 2024.
[14] Yuxuan Hu, Ke Wang, Xiaokang Zhang, Fanjin Zhang,
Cuiping Li, Hong Chen, and Jing Zhang. Sam decoding:
Speculative decoding via suffix automaton. InProceed-
ings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 12187–12204, 2025.
[15] Frank P Kelly, Aman K Maulloo, and David K Tan. Rate
control for communication networks: shadow prices,
proportional fairness, and stability.Journal of the Oper-
ational Research Society, 49(3):237–252, 1998.
[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonza-
lez, Hao Zhang, and Ion Stoica. Efficient memory man-
agement for large language model serving with page-
dattention. InProceedings of the ACM SIGOPS 29th
Symposium on Operating Systems Principles, 2023.
[17] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast
inference from transformers via speculative decoding. In
International Conference on Machine Learning, pages
19274–19286. PMLR, 2023.
[18] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang
Zhang. Eagle-2: Faster inference of language
models with dynamic draft trees.arXiv preprint
arXiv:2406.16858, 2024.
[19] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang
Zhang. Eagle: Speculative sampling requires rethinking
feature uncertainty.arXiv preprint arXiv:2401.15077,
2024.
[20] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang
Zhang. Eagle-3: Scaling up inference acceleration of
large language models via training-time test.arXiv
preprint arXiv:2503.01840, 2025.
[21] Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding,
Ruoyu Sun, Ge Zhang, Wenhao Huang, and Zhi-Quan
Luo. Knapsack rl: Unlocking exploration of llms
via optimizing budget allocation.arXiv preprint
arXiv:2509.25849, 2025.
13

---

<!-- page:14 -->

[22] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao
Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng,
Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 techni-
cal report.arXiv preprint arXiv:2412.19437, 2024.
[23] Bingshuai Liu, Ante Wang, Zijun Min, Liang Yao, Haibo
Zhang, Yang Liu, Anxiang Zeng, and Jinsong Su. Spec-
rl: Accelerating on-policy reinforcement learning via
speculative rollouts.arXiv preprint arXiv:2509.23232,
2025.
[24] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang,
Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe
Lu, Junjie Yan, et al. Muon is scalable for llm training.
arXiv preprint arXiv:2502.16982, 2025.
[25] Han Lu, Zichen Liu, Shaopan Xiong, Yancheng He, Wei
Gao, Yanan Wu, Weixun Wang, Jiashun Liu, Yang Li,
Haizhou Zhao, et al. Part ii: Roll flash–accelerating rlvr
and agentic training with asynchrony.arXiv preprint
arXiv:2510.11345, 2025.
[26] Zhiyu Mei, Wei Fu, Kaiwei Li, Guangju Wang,
Huanchen Zhang, and Yi Wu. Real: Efficient rlhf train-
ing of large language models with parameter realloca-
tion.arXiv preprint arXiv:2406.14088, 2024.
[27] Gabriele Oliaro, Zhihao Jia, Daniel F Campos, and Au-
rick Qiao. Suffixdecoding: Extreme speculative decod-
ing for emerging ai applications. InThe Thirty-ninth
Annual Conference on Neural Information Processing
Systems, 2025.
[28] Alexandre Piché, Ehsan Kamalloo, Rafael Pardinas, Xi-
aoyin Chen, and Dzmitry Bahdanau. Pipelinerl: Faster
on-policy reinforcement learning for long sequence gen-
eration.arXiv preprint arXiv:2509.19128, 2025.
[29] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng
Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng,
and Xinran Xu. Mooncake: Trading more storage for
less computation—a KVCache-centric architecture for
serving LLM chatbot. In23rd USENIX Conference on
File and Storage Technologies (FAST 25), pages 155–
170, 2025.
[30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing
the limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, 2024.
[31] Gerald Shen, Zhilin Wang, Olivier Delalleau, Jiaqi Zeng,
Yi Dong, Daniel Egert, Shengyang Sun, Jimmy Zhang,
Sahil Jain, Ali Taghibakhshi, et al. Nemo-aligner: Scal-
able toolkit for efficient model alignment.arXiv preprint
arXiv:2405.01481, 2024.
[32] Guangming Sheng, Yuxuan Tong, Borui Wan, Wang
Zhang, Chaobo Jia, Xibin Wu, Yuqi Wu, Xiang Li, Chi
Zhang, Yanghua Peng, et al. Laminar: A scalable asyn-
chronous rl post-training framework.arXiv preprint
arXiv:2510.12633, 2025.
[33] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu,
Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and
Chuan Wu. Hybridflow: A flexible and efficient rlhf
framework. InProceedings of the Twentieth European
Conference on Computer Systems, pages 1279–1297,
2025.
[34] Mohammad Shoeybi, Mostofa Patwary, Raul Puri,
Patrick LeGresley, Jared Casper, and Bryan Catanzaro.
Megatron-lm: Training multi-billion parameter lan-
guage models using model parallelism.arXiv preprint
arXiv:1909.08053, 2019.
[35] Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim,
and Seunghyeok Hong. Llm-as-a-judge & reward
model: What they can and cannot do.arXiv preprint
arXiv:2409.11239, 2024.
[36] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao,
Xinyi Zhang, Yong Li, and Wei Lin. Llumnix: Dynamic
scheduling for large language model serving. In18th
USENIX symposium on operating systems design and
implementation (OSDI 24), pages 173–191, 2024.
[37] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Ji-
ahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen,
Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agen-
tic intelligence.arXiv preprint arXiv:2507.20534, 2025.
[38] Kimi Team, Angang Du, Bofei Gao, Bowei Xing,
Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao,
Chenzhuang Du, Chonghua Liao, et al. Kimi k1.5: Scal-
ing reinforcement learning with llms.arXiv preprint
arXiv:2501.12599, 2025.
[39] Meituan LongCat Team. Introducing longcat-
flash-thinking: A technical report.arXiv preprint
arXiv:2509.18883, 2025.
[40] vLLM Team. vllm. https://github.com/
vllm-project/vllm, 2025.
[41] Leandro von Werra, Younes Belkada, Lewis Tunstall,
Edward Beeching, Tristan Thrush, Nathan Lambert,
Shengyi Huang, Kashif Rasul, and Quentin Gallouédec.
Trl: Transformer reinforcement learning. https://
github.com/huggingface/trl, 2020.
[42] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao
Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang,
Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language
model’s perception of the world at any resolution.arXiv
preprint arXiv:2409.12191, 2024.
14

---

<!-- page:15 -->

[43] Weixun Wang, Shaopan Xiong, Gengru Chen, Wei
Gao, Sheng Guo, Yancheng He, Ju Huang, Jiaheng
Liu, Zhendong Li, Xiaoyang Li, et al. Reinforcement
learning optimization for large-scale learning: An effi-
cient and user-friendly scaling library.arXiv preprint
arXiv:2506.06122, 2025.
[44] Peter Weiner. Linear pattern matching algorithms. In
14th Annual Symposium on Switching and Automata
Theory (swat 1973), pages 1–11. IEEE, 1973.
[45] Yongji Wu, Xueshen Liu, Haizhong Zheng, Juncheng
Gu, Beidi Chen, Z Morley Mao, Arvind Krishnamurthy,
and Ion Stoica. Rlboost: Harvesting preemptible re-
sources for cost-efficient reinforcement learning on llms.
arXiv preprint arXiv:2510.19225, 2025.
[46] Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase,
Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad
Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor
Holmes, et al. Deepspeed-chat: Easy, fast and affordable
rlhf training of chatgpt-like models at all scales.arXiv
preprint arXiv:2308.01320, 2023.
[47] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gao-
hong Liu, Lingjun Liu, et al. Dapo: An open-source llm
reinforcement learning system at scale.arXiv preprint
arXiv:2503.14476, 2025.
[48] Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li,
Huiqiang Jiang, Junrong Lin, Yuqiong Liu, An Yang, Jin-
gren Zhou, and Junyang Lin. Stabilizing reinforcement
learning with llms: Formulation and practices.arXiv
preprint arXiv:2512.01374, 2025.
[49] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui
Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu,
Rui Men, An Yang, et al. Group sequence policy opti-
mization.arXiv preprint arXiv:2507.18071, 2025.
[50] Yinmin Zhong, Zili Zhang, Xiaoniu Song, Hanpeng
Hu, Chao Jin, Bingyang Wu, Nuo Chen, Yukun Chen,
Yu Zhou, Changyi Wan, et al. Streamrl: Scalable, het-
erogeneous, and elastic rl for llms with disaggregated
stream generation.arXiv preprint arXiv:2504.15930,
2025.
[51] Yinmin Zhong, Zili Zhang, Bingyang Wu, Shengyu
Liu, Yukun Chen, Changyi Wan, Hanpeng Hu, Lei Xia,
Ranchen Ming, Yibo Zhu, et al. Optimizing RLHF train-
ing for large language models with stage fusion. In22nd
USENIX Symposium on Networked Systems Design and
Implementation (NSDI 25), pages 489–503, 2025.
[52] Yuzhen Zhou, Jiajun Li, Yusheng Su, Gowtham Ramesh,
Zilin Zhu, Xiang Long, Chenyang Zhao, Jin Pan, Xi-
aodong Yu, Ze Wang, et al. April: Active partial rollouts
in reinforcement learning to tame long-tail generation.
arXiv preprint arXiv:2509.18521, 2025.
[53] Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contrib-
utors. slime: An llm post-training framework for rl
scaling. https://github.com/THUDM/slime, 2025.
GitHub repository. Corresponding author: Xin Lv.
15

---

<!-- page:16 -->

A Implementation Details of SEER
A.1 Context-Aware Scheduling Algorithm
Algorithm 2 presents the context-aware scheduling workflow
built on top of divided rollout. The algorithm is invoked con-
tinuously by SEER’s global scheduler, each time returning a
scheduling decision (r⋆,i ⋆) that assigns the selected request
r⋆ to an inference instance i⋆, until all requests are completed.
Algorithm 2Context-Aware Scheduling based on Divided
Rollout
Require: Active requests R={r g,i} grouped by prompt g; group-
level length estimates {bLg}; inference instances I with KV-
usage telemetry.
Ensure:A scheduling decision(r ⋆,i ⋆)withr ⋆ ∈Randi ⋆ ∈I.
1:for allr g,i ∈Rdo
2:ifr g,i is finishedthen
3: bLg ←UPDATEESTIMATE( bLg,L g,i)
4:remover g,i fromR
5:else ifr g,i is the group’s speculative requestthen
6:keep in high-priority queueQ spec
7:else
8:add to low-priority candidate setC rest
9:r ⋆ ←None
10:if¬ISEMPTY(Q spec)then
11:r ⋆ ←PICKSFS(Q spec)▷smallest generated length first
12:else if¬ISEMPTY(C rest)then
13:r ⋆ ←PICKLFS(C rest)▷largest bLg first
14:else
15:returnall requests are finished
16:
r⋆.max_tokens←min
 
chunk_size,
r⋆.ori_max_tokens−r ⋆.generated_tokens

17:i ⋆ ←SELECTINSTANCE(I,r ⋆.max_chunk_tokens,KV-usage)
18:ifi ⋆ ̸=None then
19:return(r ⋆,i ⋆)
20:returnno available instance for this cycle
A.2 Workflow of Distributed Grouped Draft
Server
To minimize speculative decoding latency in the critical
path, Distributed Grouped Draft Server (DGDS) adopts asyn-
chronous updates and employs a distributed master-worker
architecture for global sharing of grouped context, as illus-
trated in Figure 13. The system operates through four key
steps, with core APIs listed in Table 5 and Table 6:
1) Asynchronous Append: Each inference instance runs an
independent process to handle output tokens. When newly
generated tokens are produced, the instance invokes the
update_cst API to send them to DGDS, identified by
group_id. To reduce communication overhead, each request
batches a fixed number of tokens before sending updates,
which has negligible impact on draft token quality.
Instance & Draft Client
……AsyncAppend
Local SpeculationInferenceEngine4 Embedded DraftLibrary
InstanceInstance
1 PeriodicFetch3
Global Aggregation2
Global Grouped Draft Server
Figure 13: The distributed grouped draft server.
2) Global Aggregation: DGDS aggregates token updates
from requests belonging to the same group. To prevent cross-
request interference, DGDS isolates updates by request_id,
mapping each new token only to the corresponding local path
in the CST.
3) Periodic Fetch: Each inference instance embeds a draft
client as a library component that periodically synchronizes
the latest CST from DGDS. The client registers active request
groups via register_group and then periodically invokes
fetch_cst to retrieve the latest CSTs for these groups. To
reduce communication overhead, the client supports incre-
mental synchronization based on local cache states.
4) Local Speculation: Inference instances perform spec-
ulation based on their local CSTs by invoking the
batch_speculate API. The local CSTs aggregate paths
from all requests within the same group, enabling instances
to share contextual statistics and obtain higher-quality draft
tokens.
16

---

<!-- page:17 -->

Table 5: Grouped draft server API.
Operation Description Parameters
update_cst Append generated tokens from a specific re-
quest to the compressed suffix tree
const string& group_id, int request_id,
int prev_token_count, const vector<int>&
new_tokens
fetch_cst Fetch incremental draft contexts of request
groups based on their current cache states (if
have)
const vector<string>& group_ids, const
vector<DraftCacheInfo>& draft_cache_infos
Table 6: Draft client API.
Operation Description Parameters
register_group Register a new request group for draft fetch-
ing with TTL
const string& group_id, int ttl_seconds
batch_speculate Generate speculative tokens for multiple re-
quests via zero-copy memory access: reads
input token patterns and writes predicted to-
kens directly to inference engine memory
buffers
const vector<string>& group_ids, const
vector<size_t>& buffer_offsets, void*
pattern_buffer_ptr, void* output_buffer_ptr,
const vector<SpeculationArgs>&
speculation_args
SpeculationArgs:max_spec_tokens(int),pattern_lookup_max(int),pattern_lookup_min(int),top_k(int)
17