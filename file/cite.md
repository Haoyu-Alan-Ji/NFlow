# Citation checklist for the Introduction

## Formatting / NeurIPS template

- NeurIPS 2026 official Overleaf template confirms that `natbib` is loaded by default and that author-year or numeric citation styles are acceptable if used consistently.

## Variable selection and penalized methods

1. Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso".
   - Used for classical sparse penalized regression.
2. Fan and Li (2001), "Variable Selection via Nonconcave Penalized Likelihood and Its Oracle Properties".
   - Used for nonconvex penalized likelihood and oracle-property variable selection.
3. Zou (2006), "The Adaptive Lasso and Its Oracle Properties".
   - Used for adaptive LASSO baseline/context.
4. Buhlmann and van de Geer (2011), "Statistics for High-Dimensional Data".
   - Used for high-dimensional sparse estimation background.

## Bayesian variable selection and spike-and-slab priors

5. Mitchell and Beauchamp (1988), "Bayesian Variable Selection in Linear Regression".
   - Used for early Bayesian variable selection.
6. George and McCulloch (1993), "Variable Selection via Gibbs Sampling".
   - Used for spike-and-slab Gibbs sampling.
7. George and McCulloch (1997), "Approaches for Bayesian Variable Selection".
   - Used for general Bayesian variable selection overview.
8. Ishwaran and Rao (2005), "Spike and Slab Variable Selection: Frequentist and Bayesian Strategies".
   - Used for spike-and-slab variable selection.
9. O'Hara and Sillanpaa (2009), "A Review of Bayesian Variable Selection Methods".
   - Used for review-level background.
10. Castillo, Schmidt-Hieber, and van der Vaart (2015), "Bayesian Linear Regression with Sparse Priors".
    - Used for high-dimensional sparse Bayesian regression theory.

## ARD / sparse Bayesian learning

11. MacKay (1992), "Bayesian Interpolation".
    - Used for early ARD-style Bayesian evidence/shrinkage ideas.
12. Neal (1996), "Bayesian Learning for Neural Networks".
    - Used for Bayesian neural network / ARD background.
13. Tipping (2001), "Sparse Bayesian Learning and the Relevance Vector Machine".
    - Used for ARD and sparse Bayesian learning.

## Variational inference

14. Jordan et al. (1999), "An Introduction to Variational Methods for Graphical Models".
    - Used for foundational VI.
15. Wainwright and Jordan (2008), "Graphical Models, Exponential Families, and Variational Inference".
    - Used for variational methods and graphical models.
16. Blei, Kucukelbir, and McAuliffe (2017), "Variational Inference: A Review for Statisticians".
    - Used for modern VI overview and limitations.
17. Hoffman et al. (2013), "Stochastic Variational Inference".
    - Used for scalable stochastic VI.
18. Kingma and Welling (2014), "Auto-Encoding Variational Bayes".
    - Used for reparameterized variational inference.
19. Rezende, Mohamed, and Wierstra (2014), "Stochastic Backpropagation and Approximate Inference in Deep Generative Models".
    - Used for stochastic backpropagation and reparameterized inference.

## Variational Bayes for spike-and-slab / variable selection

20. Carbonetto and Stephens (2012), "Scalable Variational Inference for Bayesian Variable Selection in Regression, and Its Accuracy in Genetic Association Studies".
    - Used for mean-field VB in Bayesian variable selection.
21. Ormerod, You, and Muller (2017), "A Variational Bayes Approach to Variable Selection".
    - Used for mean-field spike-and-slab VB theory/methodology.
22. Ray, Szabo, and Clara (2020), "Spike and Slab Variational Bayes for High Dimensional Logistic Regression".
    - Used for high-dimensional spike-and-slab VB.

## Mean-field limitations and posterior uncertainty

23. Turner and Sahani (2011), "Two Problems with Variational Expectation Maximisation for Time-Series Models".
    - Used for underestimation and approximation issues in VI.
24. Giordano, Broderick, and Jordan (2018), "Covariances, Robustness, and Variational Bayes".
    - Used for posterior covariance/uncertainty limitations in VB.

## Normalizing flows

25. Tabak and Vanden-Eijnden (2010), "Density Estimation by Dual Ascent of the Log-Likelihood".
    - Used for early normalizing-flow-style density transformation.
26. Tabak and Turner (2013), "A Family of Nonparametric Density Estimation Algorithms".
    - Used for normalizing flow background.
27. Rezende and Mohamed (2015), "Variational Inference with Normalizing Flows".
    - Used for flow-based variational inference.
28. Dinh, Krueger, and Bengio (2014), "NICE: Non-linear Independent Components Estimation".
    - Used for invertible coupling flows.
29. Dinh, Sohl-Dickstein, and Bengio (2017), "Density Estimation Using Real NVP".
    - Used for real-valued non-volume-preserving coupling flows.
30. Kingma et al. (2016), "Improved Variational Inference with Inverse Autoregressive Flow".
    - Used for autoregressive flows in VI.
31. Papamakarios et al. (2021), "Normalizing Flows for Probabilistic Modeling and Inference".
    - Used for normalizing flow review.

## Continuous relaxations and sparse gates

32. Maddison, Mnih, and Teh (2017), "The Concrete Distribution".
    - Used for continuous relaxations of discrete random variables.
33. Jang, Gu, and Poole (2017), "Categorical Reparameterization with Gumbel-Softmax".
    - Used for differentiable categorical/discrete relaxations.
34. Louizos, Welling, and Kingma (2018), "Learning Sparse Neural Networks through L0 Regularization".
    - Used for hard-concrete/L0-style sparse gates.

## Annealing and early stopping

35. Rose (1998), "Deterministic Annealing for Clustering, Compression, Classification, Regression, and Related Optimization Problems".
    - Used for deterministic annealing / continuation-style optimization.
36. Prechelt (1998), "Early Stopping -- But When?".
    - Used for early stopping as an optimization/regularization principle.