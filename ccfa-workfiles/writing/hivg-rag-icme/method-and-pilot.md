# HiVG-RAG for Layout-Rich Document Question Answering

Technical draft for IEEE ICME 2027. Experiments below use development cohorts.

## Abstract

Multimodal document question answering requires evidence at different scales, from a labeled component in a diagram to a relationship spanning several sections. HiVG-RAG is a training-free retrieval framework that organizes section summaries, figures, and tables above a layer of localized regions and entity mentions. It uses the visual similarity of crops to constrain entity merges and link differently named occurrences. At query time, a combination of lexical cues and similarity concentration selects either traversal over the macro graph or localized retrieval within selected parents. In a controlled entity-pair study, visual cues prevented two of four same-name false merges. Automatic indexing correctly named and localized two of 101 annotated endpoints. On 88 development questions from MMLongBench-Doc and DocBench, the system reduced the average input to Qwen2.5-VL-7B by approximately 90% and 78%, respectively, relative to Direct. Fixed micro traversal obtained higher ANLS on both cohorts. The experiments trace these outcomes to grounded mention coverage and the distribution of queries across retrieval layers.

## Introduction

Layout-rich documents place information in several complementary forms. A manual may use the same number for different components in front and side views. A results section may refer to a curve by model name while its legend uses an abbreviation. Both cases require the index to preserve the location and appearance of each mention long enough to resolve its identity. Linking a reference to the wrong occurrence can redirect retrieval to a plausible but unrelated neighborhood. [MMLongBench-Doc](https://arxiv.org/abs/2407.01523) · [DocBench](https://arxiv.org/abs/2407.10701)

The amount of evidence needed also depends on the question. Reading a chart point requires a spatially precise image region. Comparing trends in separate sections requires evidence from multiple pages. Retrieving images of complete pages preserves their layout, and a graph adds links between related evidence. The retrieval procedure still determines whether the generator sees the relevant regions together and how much context it must process. [VisRAG](https://arxiv.org/abs/2410.10594) · [MG²-RAG](https://arxiv.org/abs/2604.04969)

HiVG-RAG addresses these decisions at indexing and retrieval time. The index stores page-level provenance for visually grounded mentions and uses crop similarity to constrain merges and construct weighted alias links. The retriever selects a graph layer based on the query and can pass native-resolution regions to the answer model. We evaluate each operation with annotated mentions or evidence pages and report the resulting trade-off between answer quality and multimodal input tokens.

## Related work

VisRAG and ColPali retrieve page images through visual embeddings. Page representations preserve layout and supply the generator with both text and graphics, which makes them natural controls for a system that retrieves smaller regions. [VisRAG](https://arxiv.org/abs/2410.10594) · [ColPali](https://arxiv.org/abs/2407.01449)

Graph-based multimodal retrieval links evidence at finer granularity. RAG-Anything converts modality-specific content into descriptions and uses hybrid graph and vector retrieval. MMGraphRAG selects candidate entities from text and image graphs and uses a language model to align them. MG²-RAG combines entity-driven visual grounding with relevance propagation across a graph of texts and images at several granularities. HiVG-RAG studies whether visual compatibility can constrain reference resolution at indexing time and whether question-dependent restrictions on graph traversal preserve the evidence needed by the generator. [RAG-Anything](https://arxiv.org/abs/2510.12323) · [MMGraphRAG](https://github.com/wanxueyao/MMGraphRAG) · [MG²-RAG](https://arxiv.org/abs/2604.04969)

## Method

Document question answering requires retaining the local evidence that distinguishes similarly named entities and reaching the pages needed for cross-page questions. In a multimodal graph, both operations depend on how the index represents a mention. If the index contracts two different visual occurrences into one node, traversal can propagate relevance between their neighborhoods. If it discards the coordinates of an occurrence, the answer model receives the surrounding layout at the expense of the small region that supports the answer. HiVG-RAG retains occurrence coordinates during entity resolution and uses the query to determine which level of the graph can supply evidence.

### Two-tier graph

For each document, MinerU extracts figures, tables, layout blocks, and captions. We construct a macro graph of figures, tables, and section summaries. Caption references create edges between blocks, and nearest-neighbor links connect macros with similar textual descriptions. Micro nodes represent layout regions and textual mentions. For grounded mentions, the index stores a physical page number and a bounding box in the pixel coordinates of the rendered page. Each occurrence retains its own mention ID and `PART_OF` relation to its macro parent after entity resolution. This provenance permits traversal to restrict the occurrences of an entity to specific parents.

### Visually grounded entity resolution

A frozen SigLIP encoder embeds descriptions and visual crops. For a candidate pair of entity mentions $u,v$, let $s_t(u,v)$ be the cosine between description embeddings and $s_v(u,v)$ the cosine between crop embeddings when both crops are available. A pair with $s_t\geq\tau_t$ can merge if $s_v\geq\tau_s$ or if visual evidence is missing at one or both endpoints. During clustering, we check this condition on every pair across the two clusters before joining them. Complete-link compatibility prevents a sequence of text-only merges from reconnecting a pair rejected on visual grounds.

When $s_t<\tau_t$ but $s_v\geq\tau_a$, we retain distinct entity nodes and add a `VISUAL_ALIAS` link of weight $s_v$ if their contexts are related. Related contexts include the same page, an explicit reference between macro parents, and nearby pages whose parents share a section or have sufficiently similar descriptions. These links preserve the local descriptions and coordinates of both occurrences for retrieval. We use $\tau_t=0.85$, $\tau_s=0.60$, and $\tau_a=0.85$ in the pilot.

### Query-adaptive hierarchical traversal

We form a distribution $p_q$ by applying a temperature-scaled softmax to the query similarities over the $N$ macro nodes. With a lexical detail prior $\phi(q)$ and mixing weight $\lambda$, the route uses

$$
\gamma(q)=\lambda\phi(q)+(1-\lambda)
\left[1-\frac{H(p_q)}{\log N}\right].
$$

The lexical prior increases for detail cues such as a numbered row or a named chart element. A high-entropy distribution spreads initial relevance across the macro graph. A low-entropy distribution indicates a narrower concentration of scores. When $N=1$, we set the entropy contribution to zero. With the pilot settings $\lambda=0.65$ and $\tau_{\mathrm{zoom}}=0.55$, a zero lexical prior bounds $\gamma$ by $0.35$, keeping the route at the macro level.

Below the threshold, personalized PageRank propagates relevance only over the macro graph. Otherwise, the retriever selects the two most similar macro parents and induces a micro graph over their child occurrences. Relevance propagates only between eligible child nodes, including retained alias links between those children. Within each route, the answer model receives the ranked evidence. In the micro route it receives the native-pixel crops for grounded evidence and the corresponding local text. Macro and micro routes allow six and eight evidence items respectively in the default configuration.

## Pilot experiments

We ask whether visual evidence improves mention resolution when a crop is available, how often automatic indexing finds that evidence, and how the resulting graph trades answer quality for context length. A pairwise probe uses 101 manually reviewed pairs from 12 PDFs. Documents form disjoint development and held-out partitions with 51 and 50 pairs. We compare text-only resolution, full-figure similarity, and similarity between verified entity crops on the same pairs, holding contextual eligibility fixed. The default thresholds merge all four same-name distinct pairs under text-only resolution and two under either visual condition. Entity crops preserve all seven same-name matching pairs. Full figures preserve six. On nine matching aliases defined over different regions, crop and full-figure similarity recover three and two respectively. Crop similarity also creates three false links among its six such predictions. [Pairwise results](../../../runs/vger-pilot/summary.json) · [Held-out cases](../../experiments/vger-pilot/entity-cases.pdf)

Using unmodified automatic indexing on 59 pages from the annotated documents, 20 of 101 labeled endpoints have an overlapping region at IoU at least 0.5. Two endpoints have both a matching name and an overlapping region. This limits how often either visual rule can act in production, even on the annotated pairs for which the correct crop separates the entities. [Automatic coverage](../../../runs/vger-pilot/coverage.log)

With annotated names and boxes provided to the graph builder, entity-pair precision over labeled predictions in the held-out graphs is 12/39 for text, 29/59 for full figures, and 30/48 for crops. Full-figure and crop similarity each link one of the nine matching aliases located in distinct regions. This graph-level probe applies the production context filter, which limits cross-page bridging. [Controlled graph probe](../../../runs/vger-pilot/oracle-graph.json)

A geometry-based indexing probe pairs PDF text positions with nearby MinerU blocks and embedded-image bounds. It correctly names and overlaps 11 of 101 annotated endpoints at rank one and 15 within the first five positions. Eleven of the 15 hits come from subfigures in a single paper. Two of 23 critical cross-region pairs have both endpoints in the first five predictions, and both pairs are repeated labels for different entities. In a separate controlled graph experiment, expanding the alias context window from one page to eight recovers two additional aliases on development documents. Across all tested windows, the held-out graph recovers one of nine aliases spanning different regions. [Geometry probe](../../../runs/vger-pilot/layout-grounding.json) · [Alias context experiment](../../../runs/vger-pilot/alias-context/graphs.json)

The QA pilot compares methods on all 68 questions from seven MMLongBench-Doc PDFs and all 20 questions from four DocBench PDFs. We keep the question IDs, references, and final answer prompt fixed per model. With Qwen2.5-VL-7B, HiVG-RAG averages 450 prompt tokens on MMLongBench-Doc and 300 on DocBench. The corresponding Direct runs use 4,328 and 1,345 prompt tokens. HiVG-RAG obtains ANLS 0.243 and 0.069, compared with Direct at 0.226 and 0.159. Fixed Micro-only obtains 0.318 and 0.151 with 650 and 432 tokens respectively. With Qwen2.5-VL-32B AWQ, HiVG-RAG and Direct reach 0.274 and 0.196 on MMLongBench-Doc, and 0.031 and 0.291 on DocBench. The two MMLongBench-Doc document-bootstrap intervals for the HiVG-RAG minus Direct difference include zero. [QA results](../../../runs/vger-pilot/full-document-results.json) · [ANLS and prompt tokens](../../experiments/vger-pilot/pilot-pareto.pdf)

On the 50 MMLongBench-Doc development questions with evidence pages, the default router sends six of 30 micro questions and eight of 20 macro questions to the micro graph. Using the same six-item limit for both graph layers, the default route covers all annotated pages for 14 questions. Four-parent traversal covers 15 and fixed Micro-only covers 19. For identifying micro questions, AUROC is 0.350 with the lexical score, 0.630 with entropy concentration, and 0.392 with the configured combination. Numerical questions seeking document-level aggregates repeatedly trigger the lexical prior. [Full-document retrieval probe](../../../runs/vger-pilot/full-routing-matched-k6.json) · [Score decomposition](../../../runs/vger-pilot/routing-score-components.json)
