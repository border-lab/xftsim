To Dr. Margot Brandt and the editorial staff *Nature Genetics*,

Thank you for considering our manuscript "Simple models of non-random
mating and environmental transmission bias standard human genetics
statistical methods\" for publication in *Nature Genetics*. We thank you
and the reviewers for your thorough and insightful comments. We have
addressed the reviewers' concerns, resulting in an improved manuscript.

Below, we reproduce the reviewers' comments and provide detailed
responses, with our replies highlighted in blue.

Sincerely,

![](media/image1.png){width="0.9323501749781278in"
height="0.41437773403324585in"}\
**Richard Border, PhD\
**Assistant Professor\
Ray and Stephanie Lane Computational Biology Department\
Carnegie Mellon University School of Computer Science\
rborder@cs.cmu.edu

# Reviewer 1

I took some time looking into the technical details and mathematical
properties of the cross­trait assortative mating model that the authors
rely on. However, that led to a number of things that I find puzzling.
Most, although not all, of my questions are related to the main
hypothetical example that is used to support the authors\' points. To
make sure my comments do not result from misunderstanding, I would start
by describing the model here based on my interpretation.

The xAM model specified in lines 154 to 157 has cross-mate correlations
all equal to 0.2. Thus, the cross-correlation matrix for the bivariate
xAM model and 5-variate xAM model has respectively 4 (= 2x2) 0.2\'s and
25 (=5x5) 0.2\'s. For both cases, each of the variables has panmictic
heritability 0.5, and there is no overlap of effect variants between
traits, which means the genetic components between traits are
uncorrelated in the 0th generation (before assortative mating is
initiated). I might have missed it, but I see no mentioning of
correlation between the non-genetic, or \'noise\', components of the
traits, which I find puzzling as I believe that to be very important
(see Additional Comments 1-4). But since it is not mentioned, I presume
this means the noise components are assumed to be uncorrelated between
traits by default. This would mean the 2 or 5 traits are uncorrelated
within person in the 0th generation.

Firstly, while I agree that the constant cross-trait correlation model
is simple, as the authors suggested, it is simple in a way that may not
be appreciated by many casual readers. In particular, it is incongruent
with the high-dimensionality of cross-trait correlations emphasized by
the authors elsewhere in the manuscript. Secondly, because the
cross-mate correlations are \'only\' set as 0.2, the models are
described as \'modest xAM\' on line 110. This I completely disagree
with.

If my mathematics is correct, these constant entry models mean that what
is driving the assortative mating is the sum of the variables, or if
standardized, the sum of variables divided by square-root of the number
of variables (the standardized sum). In particular, for the bivariate
case, it is the correlation between (X~1~ + X~2~)/√2 of the mates that
is fully responsible for the assortative mating, while for the 5-variate
case, it is the correlation between (X~1~ + X~2~ + X~3~ + X~4~ +
X~5~)/√5 of the mates that is fully responsible for the assortative
mating. All the 0.2 correlations result from an individual X being
correlated with the sum. The cross-mate correlation of the standardized
sum, in the 0th generation, can be shown to be 0.2 × number of traits,
which is 0.4 for the bivariate case and 1.0 for the 5-variate case.
Thus, the 5-variate case is not modest, but a case of extreme
assortative mating. Actually, this model is so extreme that it cannot be
extended to 6 traits, as the cross-mate correlation of the standardized
sum \'would be\' 1.2, not mathematically possible. It would also be
mathematical inviable for the 5-variate case if the correlations are
increased to 0.21. Moreover, this model is simple in substance because
the sum of the 5 traits is fully responsible for the assortative mating,
so multivariate but single dimensional, using the language of the paper.
What is most puzzling to me is that the authors spend much time/effort
emphasizing the high-dimensionality of the cross-mate correlations
(lines 105 and 106), but in the same paragraph immediately follows by
talking about these constant entry models as illuminating. As a result,
I constantly doubt that I must have made some mistake somewhere. And I
might as well have, but I have not been able to identify my mistake in
interpretation or mathematics yet. So, I remain genuinely puzzled.

We thank the reviewer for their careful and insightful analysis. The
constant-entry model with cross-mate correlations of 0.2 across 5 traits
does imply complete sorting on the standardized sum (latent correlation
= 1.0). We acknowledge that the manuscript as written did not adequately
explain our reasoning, which we address through several additional
experiments and substantial revision.

One source of confusion was a lack of clarity about the importance of
dimensionality of xAM versus the number of traits involved. We use
\"multivariate xAM\" to mean that multiple traits are involved in
cross-mate similarity (as opposed to single-trait AM), but this term is
agnostic to the structure of that similarity. Separately, multivariate
xAM can be either \"unidimensional\"---meaning cross-mate similarity can
be fully captured by assortment on a single linear index---or
\"high-dimensional\"---meaning multiple independent linear combinations
are required (see *Figure S1. Multivariate versus multidimensional
linear mating regimes* and *Figure S2. Canonical correlation analysis of
univariate nonlinear mating regimes* for examples). These distinctions
are orthogonal: the 5-trait constant-entry model is multivariate but
unidimensional (all correlations arise from sorting on the standardized
sum), while empirical xAM in the UKB and for the psychiatric traits is
both multivariate and high-dimensional (e.g., 8 canonical variates were
required to account for 90% of cross-mate covariation in the UKB). In
addition to the explanatory figures, we revise language throughout to
make this distinction clear.

The constant-entry model was chosen for its simplicity, and as the
reviewer correctly notes, the r=0.2 parameterization with 5 traits sits
at the boundary of what is mathematically possible for unidimensional
models (it cannot extend to 6 traits). While pairwise correlations of
0.2 are empirically realistic---many trait pairs show cross-mate
correlations in this range---the constant-entry model with r=0.2 across
5 traits implies perfect assortment on the latent sum. We acknowledge
that this represents an extreme case and agree with the reviewer that
the term "modest" in this context was a mistake. We have omitted this
language from the abstract and the primary text. Further, we make the
implications of perfect assortment on a latent phenotype when r=0.2
explicit throughout. For example, in the Introduction, we write

> ... multivariate xAM with exchangeable cross-mate correlations of
> 0.2---which, despite the low per-trait correlations in the model,
> implies complete assortment on the standardized trait sum---combined
> with VT accounting for 5% of phenotypic variance can result in genetic
> correlation estimates greater than 0.5 ...

and in the Online Methods we write:

> This exchangeable correlation structure is unidimensional: all
> cross-mate correlations arise from sorting on the standardized sum of
> traits, with latent correlation equal to r⋅K (where K is the number of
> traits). For 5 traits at r=0.2, this implies a latent correlation of
> 1.0. We note that this represents a boundary case for unidimensional
> models. However, existing methods proposed for measuring or accounting
> for xAM assume precisely this structure---that multivariate cross-mate
> similarity can be reduced to a single sorting dimension ...

To demonstrate that our findings are not artifacts of the boundary case
at r=0.2, we conducted additional experiments. We examined 5-trait
constant-entry models with cross-mate correlations of 0.1 (latent r =
0.5) versus 0.2 (latent r = 1.0), with and without vertical
transmission. Halving pairwise correlations approximately halves
inflation under xAM alone: genetic correlation estimates at generation 5
are 0.12 at r=0.1 versus 0.30 at r=0.2. However, with vertical
transmission accounting for 5% of phenotypic variance---the more
realistic scenario for many traits---even r=0.1 produces substantial
bias: genetic correlation estimates reach 0.37 when the true value is
zero. This demonstrates that our core findings hold across a range of
cross-mate correlation strengths, and that vertical transmission
amplifies xAM-induced biases regardless of the specific pairwise
correlation values. We present these results in new Supplementary Figure
S5, which we reproduce here:

![](media/image2.png){width="6.4375in" height="3.7552088801399823in"}

**Figure** **S5**. Genetic correlation estimates under xAM with
cross-mate correlations of 0.1 (versus 0.2 in the primary text).
Estimated genetic correlations across generations under 5-trait
constant-entry xAM with pairwise cross-mate correlations of 0.1 (latent
r = 0.5) or 0.2 (latent r = 1.0), with and without vertical transmission
(VT) accounting for 5% of phenotypic variance. Under xAM alone, halving
pairwise correlations approximately halves bias at generation 5 (rg =
0.12 at r=0.1 versus rg = 0.30 at r=0.2). However, with VT, even r=0.1
produces substantial bias: genetic correlation estimates reach 0.37 when
the true value is zero, compared to 0.56 at r=0.2. These results
demonstrate that (1) our core findings are not artifacts of the boundary
case at r=0.2, and (2) vertical transmission amplifies xAM-induced
biases across a range of cross-mate correlation strengths. Error bars
represent standard deviations across simulation replicates. All
simulations assume panmictic heritability of 0.5 and orthogonal genetic
effects (zero pleiotropy).

Additionally, although originally omitted from the main text, we now
report the raw canonical correlations from the UKB analysis. The first
canonical correlation was 0.86---itself very high and approaching
complete assortment on that dimension. Subsequent canonical correlations
were 0.68, 0.59, 0.52, and so on (Supplementary Table S2). Thus,
empirical cross-mate similarity reflects not one dimension but multiple
dimensions of substantial assortment operating simultaneously. The
\"extreme\" nature of our constant-entry simulation (latent r = 1.0) is
therefore not unrealistic as a representation of a single assortment
dimension; rather, it understates complexity by assuming only one such
dimension exists when the empirical data reveal at least eight.

We have also clarified that our reason for emphasizing
high-dimensionality (which most of the simulations presented do not
reflect) were designed to capture it: existing methods proposed for
measuring or accounting for xAM assume a single latent sorting dimension
(e.g., Bilghese et al 2023 \[10.1101/2023.09.01.555983\], Zheng et al.
2025 \[10.1101/2025.02.03.636375\]). Our empirical CCA results
demonstrate that this assumption is violated: real xAM is
high-dimensional and cannot be reduced to a single index. To address
high-dimensional xAM directly, we also present simulations using
empirical correlation structures from psychiatric diagnoses (Figures 2
and 4), which preserve the full complexity of observed cross-mate
similarity (though we note that we draw the same conclusions in this
scenario). We have revised our discussion throughout the manuscript to
clarify this.

The reviewer's comments further raised interesting questions about the
relationship between the raw cross-mate correlations for specific traits
or trait pairs versus the induced correlation between the latent score
individuals sort on. To further understand this, we examined
unidimensional models with fixed latent correlation of approximately
0.5, distributed across varying numbers of traits: 2 traits at r=0.25, 3
traits at r=0.167, 4 traits at r=0.125, and 5 traits at r=0.1. At
generation 5, genetic correlation estimates were 0.27, 0.19, 0.15, and
0.12, respectively. Interestingly, even though the latent correlation is
held constant, the ratio of the genetic correlation estimate to the
per-trait cross-mate correlation increases as assortment is distributed
across more traits. We present these results in new Supplementary Figure
S6, which we reproduce here:

![](media/image3.png){width="4.869792213473316in"
height="2.9759853455818024in"}

**Figure** **S6**. Genetic correlation estimates under unidimensional
xAM with fixed latent correlation. Estimated genetic correlations across
generations under constant-entry xAM models with approximately equal
latent correlations (\~0.5) distributed across varying numbers of
traits: 2 traits at r=0.25 (2xAM.25; latent r = 0.50), 3 traits at
r=0.167 (3xAM.167; latent r = 0.50), 4 traits at r=0.125 (4xAM.125;
latent r = 0.50), and 5 traits at r=0.1 (5xAM.1; latent r = 0.50). All
scenarios are unidimensional, with cross-mate similarity arising from
sorting on a single standardized trait sum. At generation 5, genetic
correlation estimates were 0.27, 0.19, 0.15, and 0.12, respectively.
Interestingly, even with fixed latent correlation, the ratio of the
genetic correlation estimate to the per-trait cross-mate correlation
increases as assortment is distributed across more traits. Error bars
represent standard deviations across simulation replicates. All
simulations assume panmictic heritability of 0.5, orthogonal genetic
effects (zero pleiotropy), and no vertical transmission.

Finally, we include a subsection in the Online Methods explicitly
defining the dimensionality of multivariate xAM, distinguishing
unidimensional from high-dimensional regimes, and explaining why the
constant-entry model falls into the former category despite involving
multiple traits. We present accompanying visual examples in
Supplementary Figures S1-S2.

[Additional Comments]{.underline}

1\. Within person between traits correlations of the non-genetic (noise)
components. As noted above, I did not find in the article discussing
this, which I presume that means they are assumed to be zero, for both
the hypothetical example and the analyses of real data. If these are
zero, and when most of the examples and analyses are done assuming there
is no pleiotropy, this means there are no within person correlations
between traits at the 0th generation, and the within person trait
correlations in successive generations are completely driven be
cross-trait correlations. I find this unrealistic when substantial
cross-trait associations are assumed. For the hypothetical example, if
the noise components of the traits are assumed to be substantially
positive, the models would be less extreme, i.e. the cross-mate
correlation of the sum of the traits would be less than 1 for the
5-variate case (due to variance of the sum being larger). But, I think
that would also lead to the biases investigated being smaller.

The reviewer correctly identifies that our simulations assumed
uncorrelated noise components at generation 0, and that this assumption
was not explicitly stated in the manuscript. We now include extensive
simulations incorporating correlated noise.

The reviewer\'s intuition is correct that introducing positive
within-person noise correlations would reduce the effective latent
correlation under the constant-entry model, because the variance of the
trait sum increases while the cross-mate covariance remains fixed. We
conducted additional simulations to quantify this effect, examining
5-trait constant-entry models with cross-mate correlations of 0.1 and
exchangeable within-person noise correlations (CN) of 0.1 or 0.2.

Under xAM alone (without vertical transmission), correlated noise
substantially dampens bias. At generation 5, genetic correlation
estimates decrease from 0.12 without correlated noise to 0.08 with CN =
0.1 (33% reduction) and 0.06 with CN = 0.2 (54% reduction). This is
consistent with the reviewer\'s expectation.

However, under the more realistic scenario of xAM combined with vertical
transmission---which we argue is relevant for many traits of
interest---correlated noise has a much smaller effect. With 5xAM + VT
(5%), genetic correlation estimates at generation 5 decrease from 0.37
without correlated noise to 0.34 with CN = 0.1 (7% reduction) and 0.32
with CN = 0.2 (12% reduction). The bias remains substantial: a genetic
correlation estimate of 0.32 when the true value is zero still
represents severe inflation. This pattern holds across scenarios with
and without gene-environment interactions.

The differential effect of correlated noise under xAM alone versus xAM +
VT likely reflects the distinct mechanisms generating bias in each case.
Under xAM alone, bias arises primarily from the buildup of genetic
covariance between traits; correlated noise dilutes the phenotypic
signal used in mate selection, slowing this buildup. Under xAM + VT,
bias additionally arises from confounding between genetic and
environmental transmission pathways, which is less affected by the
initial noise correlation structure.

We present these results in new Supplementary Figure S7, which we
reproduce below, and add language to the Methods section describing the
noise correlation assumption and its consequences.

![](media/image4.png){width="6.386719160104987in"
height="5.109375546806649in"}

**Figure** **S7**. Sensitivity of genetic correlation estimates to
within-person noise correlations. Estimated genetic correlations across
generations under 5-trait xAM (r=0.1) with and without vertical
transmission (VT, 5%), comparing uncorrelated noise at generation 0 (CN
= 0) to within-person noise correlations of 0.1 (CN = 0.1) and 0.2 (CN =
0.2). Under xAM alone, correlated noise substantially attenuates bias:
at generation 5, estimates decrease from 0.12 (CN = 0) to 0.08 (CN =
0.1; 33% reduction) and 0.06 (CN = 0.2; 54% reduction). However, under
xAM combined with VT, correlated noise has more limited effects:
estimates decrease from 0.37 (CN = 0) to 0.34 (CN = 0.1; 7% reduction)
and 0.32 (CN = 0.2; 12% reduction). The bias remains substantial under
the more realistic xAM + VT scenario, with genetic correlation estimates
of 0.32 when the true value is zero. This differential effect likely
reflects the distinct bias mechanisms: under xAM alone, correlated noise
dilutes the phenotypic signal for mate selection; under xAM + VT, bias
additionally arises from confounding between genetic and environmental
transmission pathways. Error bars represent standard deviations across
simulation replicates.

2\. The Taiwan data. First, I think the Supplementary Table noted in
line 214 should be Supplementary Table S4 instead of S3, which confused
me quite a bit. One can see from Table S4 that there are substantial
within-person correlations between the traits. Is it realistic to
believe that those correlations are all a consequence of xAM?

We thank the reviewer for flagging this. The reference to Taiwan NHIRD
cross-mate correlations should indeed cite Supplementary Table S4. We
have verified this is correct in the current manuscript.

Regarding whether the within-person correlations observed in Table S4
are entirely a consequence of xAM: no, though we do not claim this.
Within-person correlations among traits can arise from multiple sources,
including true pleiotropy (shared genetic effects), shared environmental
factors, population stratification, diagnostic practices, and---as the
reviewer correctly anticipates---the cumulative effects of xAM over
generations. xAM does induce within-person correlations over time by
creating genetic correlations among traits, which manifest
phenotypically. However, disentangling these sources empirically is
challenging, and our paper does not attempt to attribute observed
within-person correlations to any single cause.

Importantly, the CCA procedure explicitly accounts for within-person
correlations---they enter as normalizing factors when computing
canonical correlations (see our response to Additional Comment 4). The
finding that 7 canonical variates are required to capture 90% of
cross-mate covariation in the Taiwan data holds regardless of the source
of the within-person correlations. That is, even after accounting for
within-person correlation structure, cross-mate similarity remains
high-dimensional and cannot be reduced to a single sorting index.

We address the question of alternative sources of within-person
correlation more fully in our response to comment 3 below.

3\. The correlations of the noise components of the traits within person
can be caused by common environmental factors and/or population
stratification. With psychiatric traits, it could presumably also be a
result of how diagnosis/measurement is made. I am not an expert, but I
guess diagnosis of multiple psychiatric traits for a person is in
general not separate/independent. This add further complications to the
proper understanding of the data. Focusing entirely on xAM and ignoring
these other factors can be very misleading.

The reviewer raises an important point about the multiple potential
sources of within-person trait correlations, including common
environmental factors, population stratification, and diagnostic
practices. We agree that all of these factors may contribute to observed
patterns, and we do not claim that xAM is the sole---or even
primary---source of within-person correlations or bias in genetic
correlation estimates. The goal of our paper is not causal attribution.
We demonstrate that relaxing commonly-made assumptions (random mating,
absence of vertical transmission) can produce biases of the magnitude
observed in empirical data. This does not mean xAM and VT are the only
factors at play; rather, it means these factors cannot be ignored.
Current methods assume random mating, and our results show this
assumption matters.

To clarify this framing, we have added text to the Discussion explicitly
acknowledging alternative sources of within-person correlation and bias.
These include: (1) true pleiotropy, where shared genetic variants affect
multiple traits; (2) shared environmental exposures that influence
multiple phenotypes; (3) population stratification, where
ancestry-associated environmental factors correlate with genetic
background; and (4) diagnostic practices, particularly relevant for
psychiatric phenotypes where assessment of one condition may influence
evaluation of others (or rule them out via exclusion criteria). We note
that our simulations do incorporate some of these factors---for
instance, we examine scenarios with and without pleiotropy, and we model
vertical transmission which captures certain shared environmental
effects. However, we do not attempt to model all potential sources of
confounding, and our simulations should be interpreted as demonstrating
the sensitivity of existing methods to specific assumption violations
rather than as comprehensive models of all factors affecting empirical
estimates. In the Discussion, we now write:\
\
*More generally, within-person correlations among traits---which are
substantial for psychiatric phenotypes (Supplementary Table S4)---can
arise from multiple sources including true pleiotropy, shared
environmental exposures, population stratification, and diagnostic
practices. Our simulations are intended as sensitivity analyses
demonstrating the consequences of relaxing specific assumptions (random
mating, absence of vertical transmission), not as comprehensive models
of all factors contributing to observed patterns. The finding that xAM
and VT can produce biases of the magnitude seen in empirical data does
not imply these are the sole sources of such biases, but rather that
they cannot be safely ignored.*

4\. Canonical Correlations. Canonical correlation analysis (CCA) is used
by the authors to support the high-dimensionality of the cross-trait
correlations in UKBB and the Taiwan data. However, canonical
correlations involve both A) the cross-trait correlation matrix AND B)
the within person correlations of the traits. Thus, I find it puzzling
and unsatisfactory that in the article, the focus is only on A), and B)
seems to be completely ignored.

The reviewer raises a valid technical point. Canonical correlation
analysis does indeed involve both the cross-mate correlation matrices
and the within-person correlation matrices. As discussed in the Online
Methods, CCA solves a generalized eigenvalue problem in which the
within-person matrices appear as normalizing factors.

However, this is precisely what makes CCA the appropriate tool for our
purpose. The canonical correlations quantify cross-mate similarity after
accounting for the within-person correlation structure. If within-person
correlations were ignored, one might mistakenly conclude that high
cross-mate similarity on two traits reflects two dimensions of
assortment, when in fact both cross-mate correlations could arise from
assortment on a single latent factor that happens to load on both traits
within individuals. By normalizing for within-person correlations, CCA
extracts the independent dimensions of cross-mate similarity. We now
articulate this in the Online Methods, writing

> CCA quantifies the dimensionality of cross-mate similarity after
> accounting for within-person trait correlations. The within-person
> correlation matrices enter as normalizing factors in the generalized
> eigenvalue problem (see Taiwan NHIRD CCA procedure below for explicit
> formulation), ensuring that the canonical correlations reflect
> independent dimensions of cross-mate similarity rather than artifacts
> of within-person trait structure. This is the appropriate
> characterization for our purposes: we seek to determine how many
> independent linear combinations are required to capture cross-mate
> similarity, which is precisely what the canonical variates represent.

The finding that 8 canonical variates are required for 90% of cross-mate
covariation in UKB (and 7 in Taiwan) therefore means that even after
accounting for within-person trait correlations, cross-mate similarity
is high-dimensional and cannot be reduced to a single sorting index.
This conclusion holds regardless of the source of within-person
correlations---whether from pleiotropy, shared environments, diagnostic
practices, or the cumulative effects of xAM itself.

We note that xAM does induce within-person correlations over
generations: as genetic correlations build up between traits due to
assortative mating, these manifest as phenotypic within-person
correlations. Thus, in an equilibrium population subject to xAM, the
within-person and cross-mate correlation structures are intertwined
consequences of the same process and cannot be cleanly separated. This
does not undermine our analysis; rather, it reinforces the point that
the CCA appropriately characterizes the dimensionality of cross-mate
similarity in the population as it exists.

5\. Mathematics and Simulations. I appreciate that some analyses are
difficult to deal with by mathematical calculations alone. That could
include the behaviour of many different methods of parameter
estimations. However, some of the parameters such as the true
heritability (lines 162-163) and the true polygenic index correlation
(lines 167-168) are probably amenable to direct mathematical
calculations (as in the one­trait case). Have the authors derived
formulas for that, or are those results also based entirely on
simulations?\
\
The reviewer asks whether we have derived formulas for quantities such
as heritability dynamics and polygenic index correlations, or whether
results are based entirely on simulations. We appreciate the opportunity
to clarify.

A complete theoretical characterization of the co-evolution of genetic
covariance matrices under arbitrary multivariate xAM and vertical
transmission is the subject of a separate manuscript in preparation,
building on foundational work for the univariate case. This full
theoretical treatment is beyond the scope of the current paper, which
focuses on demonstrating biases through simulation and providing tools
for sensitivity analysis. We have added language to the Online Methods
noting this:

A complete theoretical characterization of the dynamics of these
covariance matrices under multivariate xAM and vertical transmission is
beyond the scope of the current manuscript and is the subject of ongoing
work. Here, we obtain stabilized covariance matrices from simulation and
derive analytical expressions for downstream quantities (power, type-I
error) conditional on these matrices.

We'd also like to highlight that the current paper contains substantial
analytical results. The Online Methods present closed-form expressions
for GWAS test statistics under multivariate xAM, including the
non-centrality parameter as a function of genetic covariance matrices,
and analytical expressions for power and type-I error rates. These
derivations show that test statistics follow non-central chi-squared
distributions with non-centrality parameters that depend on the excess
genetic covariance induced by xAM (the matrix *A* in our notation).
Supplementary Figure S8 validates these analytical expressions against
simulation, showing near-perfect agreement:

![](media/image5.png){width="5.922916666666667in"
height="4.935764435695538in"}

**Figure** **S8**. Theoretical predictions versus observed type-I error
rates at off-target loci (top row) and power at on-target loci (bottom
row) for α=0.05 (left column) and α=0.5 (right column) validate
mathematical derivations with high accuracy (all Pearson correlations \>
0.99) for all simulations presented in Figure 1e. "On target" hits refer
to genome-wide significant associations at causal variants for the focal
GWAS trait. "Off target" hits refer to genome-wide significant
associations at causal variants for any of the other traits, and hence
reflect false positives.

Our approach is hybrid: the genetic covariance matrices (*Σ*~AM~ and
*Σ~g~*~,0~) are obtained from simulation because they depend on the
specific mating implementation and equilibrate over a small number of
generations. However, these matrices are invariant to sample size N and
stabilize quickly, so they need only be estimated once per parameter
regime. Given these matrices, power and type-I error are computed
analytically using the derived expressions.

# Reviewer 2

The authors addressed my previous concerns.

The code is publicly available and linked to a detailed documentation.

We are pleased to have addressed the reviewer's concerns and thank them
again for feedback on previous drafts.
