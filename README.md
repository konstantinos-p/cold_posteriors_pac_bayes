<h2 align="center">Cold Posteriors through PAC-Bayes.</h2>

We investigate the cold posterior effect through the lens of PAC-Bayes generalization bounds. We argue that in the non-asymptotic setting, when the number of training samples is (relatively) small, discussions of the cold posterior effect should take into account that approximate Bayesian inference does not readily provide guarantees of performance on out-of-sample data. Instead, out-of-sample error is better described through a generalization bound. In this context, we explore the connections of the ELBO objective from variational inference and the PAC-Bayes objectives. We note that, while the ELBO and PAC-Bayes objectives are similar, the latter objectives naturally contain a temperature parameter $\lambda$ which is not restricted to be $\lambda=1$. For both regression and classification tasks, in the case of isotropic Laplace approximations to the posterior,  we show how this PAC-Bayesian interpretation of the temperature parameter captures the cold posterior effect.

<p align="center">
    <img src="/plots_for_paper/theory/abalone/abalone_original.png" height="270"/>
    <img src="/plots_for_paper/theory/kc_house/kc_house_original.png" height="270"/>
    <img src="/plots_for_paper/theory/diamonds/diamonds_original.png" height="270"/>
</p>
<p align = "center">
Fig.1 - $\mathcal{B}_{\mathrm{original}}$ as a function of the $\lambda$ parameter for different datasets. We plot the Empirical Risk, Moment and KL terms, as well as the $\mathcal{B}_{\mathrm{original}}$ bound values for different $\lambda$. We see that for increasing values of $\lambda$, the empirical risk decreases, while the Moment term increases. Interestingly the KL term (which is the KL divergence + a constant and divided by $n\lambda$) also decreases as we increase $\lambda$. As the KL term dominates the bound, the overall effect is that bound decreases as $\lambda$ increases and is tightest for the largest value of $\lambda$.
</p>

<h2> :memo: Citation </h2>

When citing this repository on your scientific publications please use the following **BibTeX** citation:

```bibtex
@article{pitas2022cold,
  title={Cold Posteriors through PAC-Bayes},
  author={Pitas, Konstantinos and Arbel, Julyan},
  journal={arXiv preprint arXiv:2206.11173},
  year={2022}
}
```

<h2> :envelope: Contact Information </h2>
You can contact me at any of my social network profiles:

- :briefcase: Linkedin: https://www.linkedin.com/in/konstantinos-pitas-lts2-epfl/
- :octocat: Github: https://github.com/konstantinos-p

Or via email at pitas.konstantinos@inria.fr

<h2> :warning: Disclaimer </h2>
This Python package has been made for research purposes.

