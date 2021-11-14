================================================
Estimators
================================================


Direct Method (DM)
--------------------------------------
A widely-used method, DM, first learns a supervised machine learning model, such as random forest, ridge regression, and gradient boosting, to estimate the mean reward function.
DM then uses it to estimate the policy value as

.. math::
    \hat{V}_{\mathrm{DM}} (\pi_e; \calD, \hat{q}) := \E_{\calD} [ \hat{q} (x_t, \pi_e) ],

where :math:`\hat{q}(a \mid x)` is the estimated reward function.
If :math:`\hat{q}(a \mid x)` is a good approximation to the mean reward function, this estimator accurately estimates the policy value of the evaluation policy :math:`V^{\pi}`.
If :math:`\hat{q}(a \mid x)` fails to approximate the mean reward function well, however, the final estimator is no longer consistent.
The model misspecification issue is problematic because the extent of misspecification cannot be easily quantified from data :cite:`Farajtabar2018`.


Inverse Probability Weighting (IPW)
--------------------------------------
To alleviate the issue with DM, researchers often use another estimator called IPW :cite:`Precup2000` :cite:`Strehl2010`.
IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy as

.. math::
    \hat{V}_{\mathrm{IPW}} (\pi_e; \calD) := \E_{\calD} [w(x_t,a_t) r_t ],

where :math:`w(x,a) := \pi_e(a \mid x) / \pi_b(a \mid x)` is the importance weight given :math:`x` and :math:`a`.
When the behavior policy is known, the IPW estimator is unbiased and consistent for the policy value.
However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.


Doubly Robust (DR)
--------------------------------------
The final approach is DR :cite:`Dudik2014`, which combines the above two estimators as

.. math::
    \hat{V}_{\mathrm{DR}} := \E_{\calD} [ \hat{q} (x_t, \pi_e) + w(x_t,a_t)  (r_t-\hat{q}(x_t, a_t) ) ].

DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward function as a control variate to decrease the variance.
It preserves the consistency of IPW if either the importance weight or the mean reward estimator is accurate (a property called *double robustness*).
Moreover, DR is *semiparametric efficient* :cite:`Narita2019` when the mean reward estimator is correctly specified.
On the other hand, when it is wrong, this estimator can have larger asymptotic mean-squared-error than IPW :cite:`Kallus2019` and perform poorly in practice :cite:`Kang2007`.



Self-Normalized Estimators
--------------------------------------
Self-Normalized Inverse Probability Weighting (SNIPW) is an approach to address the variance issue with the original IPW.
It estimates the policy value by dividing the sum of weighted rewards by the sum of importance weights as:

.. math::
    \hat{V}_{\mathrm{SNIPW}} (\pi_e; \calD) :=\frac{\E_{\calD} [ w(x_t,a_t) r_t ]}{\E_{\calD} [ w(x_t,a_t) ]}.

SNIPW is more stable than IPW, because estimated policy value by SNIPW is bounded in the support of rewards and its conditional variance given action and context is bounded by the conditional variance of the rewards:cite:`kallus2019`.
IPW does not have these properties.
We can define Self-Normalized Doubly Robust (SNDR) in a similar manner as follows.

.. math::
    \hat{V}_{\mathrm{SNDR}} (\pi_e; \calD) := \E_{\calD} \left[\hat{q}(x_t, \pi_e) + \frac{w(x_t,a_t)  (r_t-\hat{q}(x_t, a_t) )}{\E_{\calD} [ w(x_t,a_t) ]} \right].


Switch Estimators
--------------------------------------
The DR estimator can still be subject to the variance issue, particularly when the importance weights are large due to low overlap.
Switch-DR aims to reduce the effect of the variance issue by using DM where importance weights are large as:

.. math::
    \hat{V}_{\mathrm{SwitchDR}} (\pi_e; \calD, \hat{q}, \tau) := \E_{\calD} \left[ \hat{q}(x_t, \pi_e) + w(x_t,a_t) (r_t-\hat{q}(x_t, a_t) ) \mathbb{I}\{ w(x_t,a_t) \le \tau \} \right],

where :math:`\mathbb{I} \{\cdot\}` is the indicator function and :math:`\tau \ge 0` is a hyperparameter.
Switch-DR interpolates between DM and DR.
When :math:`\tau=0`, it coincides with DM, while :math:`\tau \to \infty` yields DR.
This estimator is minimax optimal when :math:`\tau` is appropriately chosen :cite:`Wang2016`.


More Robust Doubly Robust (MRDR)
--------------------------------------
MRDR uses a specialized reward estimator (:math:`\hat{q}_{\mathrm{MRDR}}`) that minimizes the variance of the resulting policy value estimator:cite:`Farajtabar2018`.
This estimator estimates the policy value as:

.. math::
    \hat{V}_{\mathrm{MRDR}} (\pi_e; \calD, \hat{q}_{\mathrm{MRDR}}) := \hat{V}_{\mathrm{DR}} (\pi_e; \calD, \hat{q}_{\mathrm{MRDR}}),

where :math:`\mathcal{Q}` is a function class for the reward estimator.
When :math:`\mathcal{Q}` is well-specified, then :math:`\hat{q}_{\mathrm{MRDR}} = q`.
Here, even if :math:`\mathcal{Q}` is misspecified, the derived reward estimator is expected to behave well since the target function is the resulting variance.


Doubly Robust with Optimistic Shrinkage (DRos)
----------------------------------------------------------------------------
:cite:`Su2019` proposes DRs based on a new weight function :math:`w_o: \calX \times \calA \rightarrow \mathbb{R}_{+}` that directly minimizes sharp bounds on the MSE of the resulting estimator.
DRs is defined as

.. math::
    \hat{V}_{\mathrm{DRs}} (\pi_e; \calD, \hat{q}, \lambda) := \E_{\calD} [ \hat{q} (x_t, \pi_e) + w_o (x_t, a_t; \lambda)  (r_t-\hat{q}(x_t, a_t) ) ],

where :math:`\lambda \ge 0` is a hyperparameter and the new weight is

.. math::
    w_o (x, a; \lambda) := \frac{\lambda}{w^{2}(x, a)+\lambda} w(x, a).

When :math:`\lambda = 0`, :math:`w_o (x, a; \lambda) = 0` leading to the standard DM.
On the other hand, as :math:`\lambda \rightarrow \infty`, :math:`w_o (x, a; \lambda) = w(x,a)` leading to the original DR.

----------------------------------------------------------------------------
DR-IC ESTIMATOR:


\bm{$\widehat{V}_\mathrm{DR-IC}(\tau)$}  \textbf{for discrete actions.} The resultant thresholding estimator \eqref{eq:v-dr-ic} based on an information borrowing (IB) reward model, as a function of the threshold $\tau$, can be re-written as follows:
\begin{align*}
\widehat{V}_\mathrm{DR\mbox{-}IC}(\tau)&= \frac{1}{n}\sum_{i=1}^n (r_i - \widehat{r}_\mathrm{IB}(x_i,a_i))  \frac{\pi (a_i \mid x_i)}{\widehat{p} (a_i \mid x_i)})\mathbbm{1}(D_\mathrm{KL}(x_i) <\tau) \\
&\hspace{1cm}  + \frac{1}{n}\sum_{i=1}^n \sum_{a \in \mathcal{A}}  \widehat{r}_\mathrm{IB}(x_i, a)\pi( a \mid x_i)\mathbb{I}(D_\mathrm{KL}(x_i)
< \tau)
+\frac{1}{n}\sum_{i=1}^n \sum_{a \in \mathcal{A}} \widehat{r}_\mathrm{IB}(x_i, a)\pi( a \mid x_i)\mathbb{I}(D_\mathrm{KL}(x_i)
\geq \tau) \\
& = \frac{1}{n}\sum_{i=1}^n (r_i - \widehat{r}_\mathrm{IB}(x_i,a_i))  \frac{\pi (a_i \mid x_i)}{\widehat{p} (a_i \mid x_i)})\mathbb{I}(D_\mathrm{KL}(x_i) <\tau) \\
& + \frac{1}{n}\sum_{i=1}^n \bigg( \widehat{r}_\mathrm{IB}(x_i, a_i)\pi( a_i \mid x_i) + \sum_{a \in \mathcal{A} \setminus \{ a_i \}}  \widehat{r}_\mathrm{IB}(x_i, a)\pi( a \mid x_i) \bigg) \mathbb{I}(D_\mathrm{KL}(x_i)
< \tau) \\
& +\frac{1}{n}\sum_{i=1}^n \bigg( {r}(x_i, a_i)\pi( a_i \mid x_i) + \sum_{a \in \mathcal{A} \setminus \{ a_i \}} \widehat{r}_\mathrm{IB}(x_i, a)\pi( a \mid x_i) \bigg) \mathbb{I}(D_\mathrm{KL}(x_i)
\geq \tau) \\
& = \frac{1}{n}\sum_{i=1}^n (r_i - \widehat{r}_\mathrm{IB}(x_i,a_i))  \frac{\pi (a_i \mid x_i)}{\widehat{p} (a_i \mid x_i)})\mathbb{I}(D_\mathrm{KL}(x_i) <\tau) \\
& + \frac{1}{n}\sum_{i=1}^n \bigg( \big[ \widehat{r}_\mathrm{IB}(x_i, a_i) \mathbb{I}(D_\mathrm{KL}(x_i)
< \tau) + {r}(x_i, a_i) \mathbb{I}(D_\mathrm{KL}(x_i)
\geq \tau) \big] \pi( a_i \mid x_i) + \sum_{a \in \mathcal{A}  \setminus \{a_i \}}  \widehat{r}_\mathrm{IB}(x_i, a)\pi( a \mid x_i) \bigg) \\
& = \frac{1}{n}\sum_{i=1}^n (r_i - \widehat{r}_\mathrm{IB}(x_i,a_i))  \frac{\pi (a_i \mid x_i)}{\widehat{p} (a_i \mid x_i)})\mathbb{I}(D_\mathrm{KL}(x_i) <\tau) + \frac{1}{n}\sum_{i=1}^n \sum_{a \in \mathcal{A} }  \widetilde{r}_\mathrm{comb}(x_i, a)\pi( a \mid x_i),
\end{align*} 
where $\widetilde{r}_\mathrm{comb}(x_i, a) = \widehat{r}_\mathrm{IB}(x_i, a)$ for all $a\in \mathcal{A}  \setminus \{a_i \}$ and, for $a=a_i$, $ \widetilde{r}_\mathrm{comb}(x_i, a_i) = \widehat{r}_\mathrm{IB}(x_i, a_i) \mathbb{I}(D_\mathrm{KL}(x_i)< \tau) + {r}(x_i, a_i) \mathbb{I}(D_\mathrm{KL}(x_i)\geq \tau)$. We implement the final expression above for $\widehat{V}_\mathrm{DR-IC}(\tau)$ in our code. \\
(2) \bm{$\widehat{r}_\mathrm{IB}$}  \textbf{for discrete actions.} Recall from \eqref{eq:r-IB-1} that
\begin{align*} 
 & \widehat{\mathbf{r}}_{\mathrm{IB}}(\widetilde{Z}) =\widetilde{Z}\widehat{\theta}_{\mathrm{ls}} + \Sigma(\widetilde{\mathbf{r}}, \mathbf{r})\Sigma_\mathbf{r}^{-1}\left[\mathrm{diag}\{\Sigma(\widetilde{\mathbf{r}}, \mathbf{r})\Sigma_\mathbf{r}^{-1}\bm{1}_n\}\right]^{-1}(\mathbf{r} - Z\widehat{\theta}_{\mathrm{ls}}).
\end{align*}