Here is the research paper content formatted for audio narration:

**SIMPLE TM: A Simple Baseline for Multivariate Time Series Forecasting**

By Hui Chen, Viet Luong, Lopamudra Mukherjee, and Vikas Singh

[pause]

**1. Introduction**

Multivariate time series data are prevalent across many fields, from finance and economics to climate science, healthcare, geophysics, and industrial monitoring.  Consequently, techniques for processing and analyzing this type of data have been extensively researched for a long time.  This includes methods like vector autoregressive models, dynamic factor models, and state-space models. These established models have a strong theoretical foundation and offer various solutions tailored to different data assumptions, such as whether the data's variance is constant or changing, the degree of autocorrelation, and whether the data is stationary or not.  These models have been refined over decades, especially in economics, computational finance, and statistics.

While deep learning architectures have brought significant performance improvements to image and natural language processing tasks in the last decade, there's growing interest in using these models for multivariate time series data as well.  Various deep learning architectures, including convolutional neural networks, graph neural networks, and transformers, have been adapted for time series. For example, attention mechanisms are used to model long-term interactions in time, and graph neural networks are employed to adaptively extract interactions between different time series.

However, all these methods encounter difficulties in handling temporal fluctuations and variations between different variables within the same dataset.  More importantly, there's immense variability across different multivariate time series datasets. Data from wearable sensors, for instance, are very different from electroencephalogram (EEG) recordings of brain activity.  Large language and vision models benefit from massive text and image datasets, which helps them handle heterogeneity.  But, while the sheer volume of multivariate time series data generated daily is huge, only a small fraction is publicly available due to privacy regulations and proprietary concerns. This scarcity of data means that deploying pre-trained models in specific time series settings often requires careful fine-tuning, which can be expensive and difficult with limited resources.

Interestingly, recent findings indicate that simple models, like a one-layer linear model, can sometimes outperform more complex Transformer-based approaches on public datasets. This suggests that directly applying complex architectures to diverse multivariate time series data is challenging.  However, recent research has explored making specific adjustments to these architectures to better handle the complexities of multivariate time series data, showing promising and reliable results.  These models serve as our baselines in this work.

[pause]

Another related area of research focuses on reinterpreting time series data as natural language.  This allows leveraging the power of large language models (LLMs). This approach benefits from the vast amounts of text data these models have been trained on.  The idea is to map time series data into word embeddings, which can then be processed by a large language model.  This mapping is often achieved using specialized adapters placed before and after the large language model in the processing pipeline.  While this approach is rapidly developing and showing promise, deploying these models on specific datasets with unique characteristics remains computationally intensive.

[pause]

Our work aligns with the non-generalist approaches, focusing specifically on multivariate time series data forecasting. Instead of heavily modifying large Transformer backbones, we incrementally add modules in a conservative manner.  Similar to large language models, we use tokenization. However, given our focus on time series data, we use tokenization methods based on classical signal processing techniques. We then incorporate a self-attention module with a slight modification to capture richer dependencies between tokens, both across time and across different dimensions.

The key contributions of this paper are:

First, we propose SimpleTM, a simple yet effective architecture that uniquely combines classical signal processing techniques with a slightly modified attention mechanism.

Second, we demonstrate that even a single-layer configuration of SimpleTM can effectively capture complex dependencies in multivariate time series data, while keeping model complexity and parameter requirements minimal.  This simple design achieves performance that is competitive with, or even better than, much larger models, including large Transformer-based architectures, on most commonly used benchmark datasets.

[pause]

**2. Preliminaries: Problem Setup and Notations**

Let's define the problem of time series forecasting formally.

First, consider a univariate time series. Let (x1, ..., xL) be a historical time series of length L, where xt represents the measurement at time step t.  Let (y1, ..., yH) be a future time series of length H.  H is the forecast horizon, and L is the lookback window. Univariate time series forecasting aims to predict (y1, ..., yH) given (x1, ..., xL).

Now, consider a multivariate time series. Let X be a matrix in R^(C x L) and Y be a matrix in R^(C x H), jointly drawn from a distribution P. Here, C is the number of channels or variables, and L and H are the lookback and horizon window lengths, respectively.  We can think of xt and yt as the t-th columns of X and Y.  So, we have L measurements for each of the C channels in X. Our goal is to forecast Y, the time series of H time steps, where each time step in the forecast window is a vector of length C.

We can also consider having multiple multivariate time series. Let N be the number of multivariate time series samples we observe. We can use an index i to refer to a specific sample, where i ranges from 1 to N.

This multivariate setting is valuable when measuring time series data for C different channels simultaneously, especially when there are correlations or dependencies between these variables.

[pause]

We define the forecasting error. For a multivariate time series (X, Y) from distribution P, and a forecasting function f that maps R^(C x L) to R^(C x H), the forecasting error is defined as the expectation over (X,Y) of the squared Frobenius norm of the difference between Y and f(X).  This is written as:

L(f) = E_(X,Y)~P [||Y - f(X)||_F^2]

where ||.||_F denotes the Frobenius norm, and E is the expectation.

Given a set of N independent and identically distributed samples {(X^(i), Y^(i))}^N_(i=1), we define the empirical risk as:

L_hat(f) = (1/N) * Sum from i=1 to N of [||Y^(i) - f(X^(i))||_F^2]

Our goal is to find a function f that minimizes this empirical forecasting error.  We will now describe the specific modules in our SimpleTM model.

[pause]

**3. Module 1: Tokenization via Stationary Wavelet Transform**

For our first module, we want a tokenization scheme for multivariate time series data that reduces the burden on subsequent modules.  Ideally, our tokens should capture temporal information across different scales, from rapid short-term variations to slow long-term trends, and also capture both local and global patterns within each of the C variables.  This would simplify the task of forecasting Y.  If we can process information at different scales separately, then synthesizing this information for forecasting can be more effective.

[pause]

The Wavelet transform is a natural choice for this. It provides a multi-scale decomposition of each signal while preserving time localization. We process each wavelet scale independently, allowing for learning scale-specific interactions within each variable.  If our forecast window is dominated by dependencies at a particular scale, this approach provides relevant information to the downstream modules.

[pause]

Let's detail our tokenization scheme.  Given a multivariate time series X in R^(C x L):

**A) Linear Projection:**  We first apply a linear projection, denoted as g, parameterized by theta, that maps R^L to R^(L'). This embeds each channel into a latent space, resulting in ˜X = g(X).  This is illustrated in Figure 1, where for each channel, time series measurements are passed through a stationary wavelet transform followed by a linear projection to obtain L' tokens.

**B) Stationary Wavelet Transform (SWT):** To get a multi-scale representation, we use a learnable stationary wavelet transform.  SWT maps R^(C x L') to R^(C x L' x (S+1)), where S is the decomposition level and we use learnable filters h0 and g0 of kernel size k. This transformation produces a set of time-frequency tokens, denoted as u^(s) for scales s=0 to S.  These tokens capture information at different temporal scales for each channel independently, as shown in Figures 1 and 2. Figure 2 illustrates tokenization via SWT. The input series is padded and processed through learnable filters. At each scale, SWT inserts zeros between filter coefficients, producing a non-decimated output. This approach, demonstrated for a scale-2 approximation, allows shift-invariant feature extraction while preserving temporal resolution.

SWT is suitable because it provides a time-invariant decomposition while preserving the original temporal structure. This is achieved by avoiding downsampling at each decomposition level, maintaining the length L'. SWT is also shift-invariant, making it effective in capturing localized events across multiple scales.  At the core of SWT are the mother wavelet ψ(t) and scaling function ϕ(t). Discrete wavelets can be expressed as:

ψ_(s,k)(t) = 2^(-s/2) * ψ(2^(-s) * t - k)

and

ϕ_(s,k)(t) = 2^(-s/2) * ϕ(2^(-s) * t - k)

where s controls the scale and k the translation.

**C) Obtaining Wavelet Coefficients:** The embedded time series {˜xt} undergoes decomposition via SWT, yielding approximation coefficients a^(s)_t and detail coefficients u^(s)_t at each level s.  For a univariate series, SWT uses a low-pass filter h and a high-pass filter g, derived from the scaling function ϕ(t) and wavelet function ψ(t):

h(k) = <ϕ(t), ϕ(2t - k)>

and

g(k) = <ψ(t), ϕ(2t - k)>

Starting with a^(0)_t = ˜xt, the decomposition at level s+1 is computed as:

a^(s+1)_t = Sum over k of [h^(s)(k) * a^(s)_(t+k)]

and

u^(s+1)_t = Sum over k of [g^(s)(k) * a^(s)_(t+k)]

Here, h^(s) and g^(s) are upsampled versions of h and g, obtained by inserting 2^s - 1 zeros between each original filter coefficient. This upsampling preserves the signal length, ensuring time invariance. Instead of fixed filter coefficients, we allow h and g to adapt to the data, making them learnable.  Our experiments show that learned filters exhibit correlation patterns similar to the respective variables, but turning off this adaptivity doesn't significantly impact results.

[pause]

In summary, the decomposition yields a final approximation u^(0)_t = a^(S)_t and wavelet coefficients {u^(s)_t}^S_(s=1).  This decomposition allows reconstructing the up-projected time series if needed:

˜xt = Sum over k of [u^(0)_k * ϕ_(S,k)(t)] + Sum from s=1 to S of [Sum over k of [u^(s)_k * ψ_(s,k)(t)]]

In our tokenization scheme, each time-frequency point u^(s)_t serves as a token, encapsulating information at scale s and time t. This multi-resolution representation provides a structured view of the data, where each token inherently retains temporal context and frequency information.

[pause]

**4. Module 2: A Small Modification of Self-Attention**

Each token represents multiple channels at a specific pseudo-time point for a specific wavelet scale. SWT already captures some temporal and frequency information.  We also want to characterize the full range of inter-channel dynamics efficiently.  For example, in finance, some asset prices move together, while others move inversely, and these relationships can change over time.  Tokens from a fine resolution might show high linear independence for a rapidly changing variable, capturing short-term dynamics. Tokens from coarser scales can reveal long-term correlations between different channel subsets, reflecting slower patterns.

The degree of inter-channel complementarity or linear independence is not fully captured by a scalar value alone.  For example, in a five-channel system, tokens (1, 1, 0, 0, 0) and (0, 0, 1, 1, 0) have a zero dot product, but span a 4D subspace, indicating high complementarity. This could indicate that the first two channels and the next two channels behave as coupled pairs.  While large Transformer models might implicitly learn these complex dependencies, explicitly endowing a smaller model with this capability seems beneficial.

[pause]

Geometric algebra product provides these abilities. It extends linear algebra to offer a unified framework for geometric constructions. We use a generalization of self-attention that retains standard dot-product attention capabilities.  While Clifford Algebra based Transformers have been proposed, they are computationally heavy. Our design is lightweight, involves minimal changes to self-attention, and is well-suited for our problem.

[pause]

Let's detail the construction.

**A) Brief Geometric Algebra Review:** Geometric Algebra (GA) provides a framework for representing and manipulating geometric objects. We focus on G2, the GA over a 2-dimensional vector space, as we consider pairs of tokens in our attention mechanism, regardless of token dimensionality.  The fundamental object in G2 is the multivector, expressed as:

M = <M>_0 + <M>_1 + <M>_2

where <M>_k is the k-vector part of M for k in {0, 1, 2}.  The key operation is the geometric product:

αβ = α · β + α ∧ β

where · is the dot product and ∧ is the wedge (or outer) product. The wedge product represents the oriented area of the parallelogram spanned by two vectors. For vectors α and β, α ∧ β results in a bivector.  Figure 3 illustrates geometric product objects. (a) shows the oriented parallelogram of the wedge product α ∧ β and (b) shows the progression from scalars to vectors, bivectors, and trivectors.  In G2, for α = ae1 + be2 and β = ce1 + de2, the wedge product is:

α ∧ β = (ad - bc)(e1 ∧ e2)

Here, (ad - bc) is the area magnitude, and e1 ∧ e2 is the orientation in the plane.

[pause]

**B) Instantiating Geometric Product in our case:** We reformulate attention using the geometric product. For tokens t and t', instead of just their dot product, we use the geometric product, combining the dot product (scalar part) with the wedge product (bivector part).  This encodes both magnitude-based similarity and geometric relationships.  We capture not only scalar similarity but also linear independence and the orientation of the space spanned by the tokens.  This allows detecting complementary information across channels and changing inter-channel dynamics, as illustrated in Figure 4, which shows a simplified illustration of geometric product attention, where the entries of the attention matrix are multivectors. For two tokens α and β, α · β is the scalar similarity, and α ∧ β tells us how they complement each other across channels.

**C) Linear Projection:** Given time-frequency tokens U^(s) in R^(C x L') for scale s, and shared weights W_Q, W_K, W_V in R^(L' x L'), the query, key, and value matrices are:

Q^(s) = U^(s) * W_Q,  K^(s) = U^(s) * W_K,  V^(s) = U^(s) * W_V, for scale s in {0, 1, ..., S}.

Linear projection is applied along L' to keep the number of channels unchanged.

**D) Geometric attention calculation:** Consider α · β + α ∧ β. We modify self-attention to mimic this. We use two different V^(s) matrices, V^(s)_1 and V^(s)_2. Dot-product attention acts on V^(s)_1 for the term Q^(s)^T * K^(s). Separately, the matrix of wedge-product objects, B = {B_(t,t')}, with B_(t,t') = q^(s)_t ∧ k^(s)_(t'), acts on V^(s)_2.

The first part is standard vanilla attention. The wedge product results in a matrix of bivector objects. We compute it element-wise for each pair of tokens.  Operating BV^(s)_2, where V^(s)_2 is column by column, is well-defined within the algebra.  B is an L' x L' matrix of bivectors, and V^(s)_2 is L' x C. We combine this with vanilla self-attention using a reduction function ζ(·) to match dimensions. ζ(·) can be the bivector's magnitude or a trainable MLP taking magnitude and orientation as input.

[pause]

In summary, the geometric attention mechanism is:

GeoProdAttn(Q, K, V) = softmax( (dot-prod(Q, K)) / √C ) * V + ζ( (wedge-prod(Q, K)) / √C ) * V

where C is a scaling factor, and V is used instead of two separate value matrices. The matrix of bivectors acts on V column by column.

[pause]

**5. Module 3: Reconstruction of Multivariate Time Series**

After processing tokens through geometric product attention, we need to reconstruct the signal in the time domain. We use a learnable inverse stationary wavelet transform (ISWT), parameterized by h1 and g1, the synthesis filters for low-pass and high-pass components. These are counterparts to the analysis filters h0 and g0 used in SWT.

Reconstruction is iterative, from coarsest to finest scale. Given initial approximation coefficients ˆa^(S) = ˆu^(0) and processed tokens {ˆu^(s)}^S_(s=0), reconstruction is:

ˆa^(s-1)_t = Sum over k of [h^(s)_1(k) * ˆa^(s)_(t+k)] + Sum over k of [g^(s)_1(k) * ˆu^(s)_(t+k)]

where h^(s)_1 and g^(s)_1 are upsampled versions of h1 and g1 at level s. We iteratively compute ˆa^(s-1) using ˆa^(s) and ˆu^(s) until we reach ˆa^(0).

[pause]

The final reconstructed time series ˆX = {ˆx1, ˆx2, ..., ˆxL'} is given by ˆa^(0). This preserves the temporal structure of the input while incorporating multi-scale information processed by geometric product attention.  This reconstructed time-domain representation ˆX is then passed through a feed-forward network and layer normalization for final refinement to produce the forecast output and calculate the loss. We perform end-to-end training.  Figure 5 provides an overview of the proposed SimpleTM model framework.

[pause]

**6. Experiment**

In this section, we discuss our experimental findings. We evaluated forecasting quality for both long-term and short-term horizons and conducted ablation studies to assess our SimpleTM model's effectiveness.

**6.1 Setup and Baselines**

We compared our model against 15 well-known forecasting models for multivariate time series data, including:

**(a) MLP-based methods:** TimeMixer, TiDE, RLinear, DLinear.
**(b) Transformer-based methods:** iTransformer, PatchTST, Crossformer, FEDformer, Autoformer, FiLM, Stationary Transformer.
**(c) CNN-based methods:** TimesNet, SCINet, MICN.
**(d) GNN-based method:** CrossGNN.

[pause]

**Datasets:** We used the following datasets:

**(a) Long-term forecasting:**  Eight benchmarks: ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2) with electricity transformer data; and Weather, Solar-Energy, Electricity, and Traffic datasets with diverse meteorological, power production, consumption, and road occupancy data.

**(b) Short-term forecasting:** PEMS dataset with four public traffic subsets (PEMS03, PEMS04, PEMS07, PEMS08) recorded every 5 minutes.

We also assessed the forecastability of all datasets and noted that ETT and Solar-Energy are particularly challenging. More dataset details are in Appendix A of the paper.

**6.2 Evaluation Results**

**Long-term forecasting results:** Table 1 in the paper presents the long-term forecasting results. Lower MSE and MAE values indicate better performance. SimpleTM shows robust performance, achieving optimal MSE/MAE in 7 out of 8 datasets. Complete results are in Appendix B.

[pause]

Comparing with TimeMixer, our method reduces MSE by 8.1% for ETTh2 and 8.8% for Electricity dataset. On the Solar-Energy dataset, our model achieves the best MSE, outperforming TimeMixer by 14.8%.  While TimeMixer uses a multi-scale approach, it underperforms in high-dimensional datasets due to its mixing mechanism being limited to linear or lower-order interactions. Its average pooling and ensemble predictions lead to information loss during scale transitions. SimpleTM overcomes these limitations with geometric attention in G2 space and classical tokenization, capturing complex, higher-order relationships by considering both magnitude and orientation of token pairs.

[pause]

Compared to iTransformer, our method achieves MSE reductions of 6.4%, 7.0%, and 4.7% for ETTm1, ETTh1, and Electricity dataset, respectively. iTransformer performs well on high-dimensional time series like Traffic, but struggles with rapidly fluctuating ETT datasets. This is because its variate-wise tokenization fails to capture fine-grained local patterns and lacks inter-channel context in lower-dimensional scenarios. SimpleTM uses wavelet-based tokens that prioritize intra-variable local interactions and effectively capture oscillatory patterns across multiple resolutions.

[pause]

**Short-term forecasting results:** Table 2 shows short-term forecasting results for PEMS datasets. SimpleTM demonstrates superior or comparable performance across all four PEMS datasets, consistently achieving the best results.  This validates our model's effectiveness for high-dimensional, short-term forecasting, complementing its strong long-term forecasting performance. Additional results for larger prediction horizons are in Table 10 in Appendix B.

**6.3 Miscellaneous Additional Analysis: Ablations, Wavelets**

**Ablation Study:** We evaluated geometric attention and stationary wavelet transform through ablation studies. Table 3 summarizes results across four datasets, averaged over four prediction horizons.  Both geometric attention and SWT contribute to our model's performance. Detailed results are in Appendix C.

[pause]

**Filters in the wavelet decomposition:** We analyzed the learned filters.  Figure 6a compares learned filters with theoretical wavelet bases.  Learned filters from random initialization sometimes resembled wavelet structures. Filters initialized with standard wavelets retained their core structure while adapting subtly.  Figure 6b compares correlation heatmaps of learned filters and original channels, showing meaningful patterns. The filter correlation matrix shows a block-like structure, while the original channel correlation matrix shows weaker overall correlations.  The model retains aspects of the original data structure while enhancing relationships through learned filters.

**Multi-scale visualization:** Figure 7 presents a multiscale forecasting example. It shows the model's accuracy in predicting overall patterns, including peaks and troughs, while capturing data cyclicity.  Our model decomposes and reconstructs time series across multiple scales. Scale 1 and 2 panels show high-frequency fluctuations, while Scale 0 reveals the low-frequency trend. This multi-resolution analysis enables extracting relevant features from various timescales for a coherent forecast.

[pause]

**Compute and Memory Footprint:** A key advantage of our single- or two-layer design is capturing complex dependencies without large backbones.  We prioritized memory and computation efficiency by using the bivector's magnitude as the reduction function.  On the Weather dataset, our model uses only 0.3% of iTransformer's parameters and 13% of TimeMixer's parameters, requires 38% and 66% less memory, and runs 1.7x and 3.4x faster, respectively. Further details are in Appendix D.

[pause]

**7. Related Work**

Time series forecasting has evolved from statistical models like ARIMA and ARMA to deep learning approaches.  These approaches can be categorized as:

**1. CNN models:**  Effective at capturing local temporal patterns. Examples include Temporal Convolutional Networks (TCN) and Sample Convolution and Interaction Networks (SCINet). TimesNet uses 2D variation modeling. CNNs can struggle with long-range forecasting due to limited receptive fields.

**2. Graph Neural Network (GNN) methods:** Capture inter-variable relationships.  MTGNN and CrossGNN are examples. GNNs can be computationally intensive.

**3. MLP models:** Offer a balance of simplicity and efficiency. TimeMixer introduces multi-scale mixing, but average pooling can lose information. RLinear shows linear models can be effective.

**4. Transformer models:** Effective at capturing long-range dependencies.  Crossformer introduces cross-dimension self-attention. iTransformer applies attention to channel-tokens but may lack fine-grained local patterns. PatchTST uses patch-based representations.

[pause]

**Multi-scale Modeling:** Capturing patterns at different resolutions is common in vision and has been adapted to time series forecasting. N-HiTS uses hierarchical forecasting. Scaleformer refines forecasts through upsampling and downsampling. Pathformer uses dual attention over patches. Pyraformer uses pyramidal attention. TimeMixer uses decomposable mixing across scales.  Challenges include inflexibility of manually designed scales and information loss from average pooling. Aggregation and reconstruction mechanisms can be fragmented in some models.

[pause]

**8. Conclusions**

Our work introduces SimpleTM, a novel approach for multivariate time series analysis. It combines wavelet-based tokenization and generalized self-attention to capture multi-scale temporal dynamics and inter-channel relationships.  Our results show competitive performance against baselines across various tasks.  Exploiting inter-channel dependency doesn't always improve performance and varies across datasets.  SimpleTM is a simple, lightweight baseline for more complex methods, although it cannot easily be extended to token-by-token generation in its current form. As public multivariate time series datasets grow, larger models may become more suitable for maximizing performance.  However, the individual components in our formulation can still inform the design of specialized adapters and tokenization schemes.

[pause]

We acknowledge discussions with experts and funding support from DARPA Strengthen program.