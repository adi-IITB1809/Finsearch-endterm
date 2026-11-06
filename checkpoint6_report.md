# Trading Strategy Optimization — Checkpoint 6 Report

**Run date:** 6 November 2026  
**Universe:** 46 tickers listed; 45 downloaded; 43 retained after the minimum-history filter.  
**Historical sample:** adjusted daily Yahoo Finance prices, 4 January 2010–28 June 2019.

## Objective and method

This project compares a Deep Q-Network (DQN) policy with LSTM and ARIMA forecasting baselines and buy-and-hold. It is a historical simulation, not evidence of live-trading performance. The agent interacts with a custom Gymnasium long/flat market environment, with hold, go-long, and go-flat actions. This state/action/reward framing follows standard RL formulations and prior financial portfolio environments [2, 3, 12]. Stable-Baselines3 provides the DQN implementation [4].

The ten state features are one-, five-, and ten-day log returns; ten-day return volatility; log volume change; ten- and fifty-day moving-average gaps; RSI; MACD; and MACD histogram. Features are standardized using training-only means and standard deviations and clipped to five standard deviations. Candidate features are defined from common technical indicators; prior work studies their empirical use and the broader question of predictor selection in portfolio choice [9, 14]. This implementation does not claim to perform a separate feature-selection algorithm.

For the primary split, training runs through 31 December 2016, validation covers 1 January 2017–30 June 2018, and test covers 1 July 2018–28 June 2019. A decision at close *t* earns the close-to-close return on *t* to *t+1*. DQN checkpoints are selected on validation mean Sharpe, and the LSTM uses validation early stopping. Test returns are not used for model selection. Repeated model selection can still overfit validation data [10].

## Models

- **DQN:** 10-day feature window plus current position; 0.1% cost per position change; downside-loss shaping penalty; 400,000 environment steps. Best checkpoint: 79,998 steps, validation mean Sharpe 0.397.
- **LSTM:** one-layer LSTM forecasts the next-day return from a 20-day feature window. Huber loss, validation early stopping, and a long position for positive forecasts. It stopped after six epochs; best validation loss was 0.73658. LSTM architecture follows Hochreiter and Schmidhuber [5].
- **ARIMA:** ARIMA(1,0,1) fitted to training-period log returns for each stock, with fixed parameters for one-step forecasts [6, 11].
- **Buy-and-hold:** long every test day. All strategies use the same 0.1% cost per position change.

Metrics use net daily strategy returns: cumulative return, annualized return, annualized volatility, annualized Sharpe, and maximum drawdown. Sharpe is daily mean divided by sample standard deviation, multiplied by √252, with no risk-free-rate subtraction [7, 8]. Portfolio results are equal-weight daily averages across the stocks available on each date.

## Primary historical test

Test period: 1 July 2018–28 June 2019. Values are equal-weight portfolio results.

| Strategy | Cumulative return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| DQN | 2.55% | 2.66% | 3.76% | 0.718 | -2.33% |
| LSTM | 10.86% | 11.39% | 12.95% | 0.898 | -11.99% |
| ARIMA | 5.38% | 5.63% | 10.72% | 0.564 | -11.77% |
| Buy-and-hold | 12.94% | 13.57% | 14.09% | 0.974 | -13.45% |

Buy-and-hold had the highest cumulative return and Sharpe. LSTM was the strongest active benchmark on both measures. DQN had the lowest volatility and shallowest drawdown, alongside the lowest return. Results are single-period estimates and do not establish generalization.

## Second chronological segmentation

The additional experiment trains from scratch using training through 31 December 2015, validation from 1 January–31 December 2016, and test from 1 January 2017–28 June 2019. Scalers are fit on the alternate training period only; the same 43-stock universe, features, cost assumptions, baseline methods, and metrics are used. DQN validation Sharpe is recorded, and LSTM stopping uses validation loss. The alternate DQN result is from the final 40,000-step model; validation Sharpe is not used to select a DQN checkpoint. The alternate test is evaluated once after training.

| Strategy | Cumulative return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| DQN | 7.07% | 2.85% | 3.51% | 0.817 | -2.80% |
| LSTM | 44.57% | 16.36% | 10.31% | 1.521 | -10.60% |
| ARIMA | 32.91% | 12.41% | 8.89% | 1.361 | -11.82% |
| Buy-and-hold | 60.13% | 21.35% | 12.16% | 1.653 | -13.45% |

In this second split, buy-and-hold again had the highest cumulative return and Sharpe; LSTM led the active models. DQN again had lower volatility and drawdown but lower returns. This split changes the historical regime and the training period, so its values should be compared as a robustness check rather than pooled with the primary test.

## Validation-only DQN tuning

One factor at a time is varied around the original settings. Each candidate uses the primary training split, a fixed 10,000-step training budget, and is ranked only by mean validation Sharpe; no test-period metric is used to choose settings. This is a small exploratory sweep, not an exhaustive hyperparameter search. The original setting and four variants are learning rate (3×10⁻⁴ vs 10⁻⁴), discount factor γ (0.95 vs 0.99), downside-loss penalty (0.25 vs 0.50), and history length (5 vs 10 days).

| Candidate | Learning rate | γ | Loss penalty | History (days) | Steps | Mean validation Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.0001 | 0.99 | 0.50 | 10 | 10,000 | 0.189 |
| Higher learning rate | 0.0003 | 0.99 | 0.50 | 10 | 10,000 | 0.207 |
| Lower discount factor | 0.0001 | 0.95 | 0.50 | 10 | 10,000 | 0.420 |
| Lower loss penalty | 0.0001 | 0.99 | 0.25 | 10 | 10,000 | 0.664 |
| Shorter history | 0.0001 | 0.99 | 0.50 | 5 | 10,000 | -0.047 |

The 0.25 loss-penalty candidate scored highest in this short sweep. The sweep is exploratory because it uses one seed and a limited 10,000-step budget per candidate; the primary 400,000-step model and the alternate-split 40,000-step model retain the original 0.50 penalty. The test results above were not used to choose a sweep candidate.

## Recent six-week check

The recent window is interpreted as a rolling period from the current date minus six weeks through today, as in the original course script. It is not the final six weeks of the 2010–2019 historical sample. Prices were freshly downloaded for the six weeks ending 6 November 2026. Equal-weight portfolio cumulative returns were DQN 0.98%, LSTM 5.72%, ARIMA 2.91%, and buy-and-hold 7.24%. This is about 30 trading days; annualized metrics are not reliable estimates. A rerun downloads a different window and can produce different values.

## Limitations

1. **Survivorship bias:** the supplied ticker list is applied across the full historical period rather than using point-in-time Nifty 50 constituents.
2. **Sample size:** 46 tickers listed; 45 downloaded; 43 retained after the minimum-history filter.
3. **Trading frictions:** the backtest charges a fixed 0.1% per position change and does not model bid–ask spread, market impact, taxes, liquidity limits, or portfolio-level capital constraints.
4. **Model selection:** validation-based selection can overfit, even when a later test period is held out [10].
5. **Portfolio construction:** stocks are simulated independently and then averaged; the experiment does not optimize cross-stock allocations or account for correlated holdings.
6. **Scope:** historical and short recent-period backtests are not live results or investment advice.

## Relevant literature

The DQN method builds on value-based deep reinforcement learning [1], while Sutton and Barto provide the broader RL framework used to define the agent and its interaction with the trading environment [2]. The market simulator makes those interactions concrete, following prior work on financial portfolio environments [12]. Stable-Baselines3 supplies the implementation used here [4]. LSTM is a recurrent model for sequential data [5], and ARIMA is a classical time-series forecasting baseline [6, 11]. Sharpe ratio is used to summarize risk-adjusted returns, with the finite-sample and annualization caveats discussed in the literature [7, 8]. The indicator set is motivated by prior computational analysis of technical trading rules [9]. The Checkpoint 5 RL-environment link is Yang et al.'s stock-trading ensemble paper [12]. Its other link points to Kutzkov's ARIMA/Prophet/LSTM comparison [13]; despite the slide's “Variable Selection” label, the linked article compares forecasting models rather than studying variable selection. Ait-Sahalia and Brandt [14] provide a separate, directly relevant study of variable selection for portfolio choice. That work is related to, but distinct from, the fixed indicator set used in this implementation. Backtest selection risk motivates keeping the test period separate from tuning [10].

## References

1. Mnih, V. et al. (2015). “Human-level control through deep reinforcement learning.” *Nature*, 518, 529–533. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236).
2. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. [Publisher page](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
3. Brockman, G. et al. (2016). “OpenAI Gym.” [arXiv:1606.01540](https://arxiv.org/abs/1606.01540).
4. Raffin, A. et al. (2021). “Stable-Baselines3: Reliable Reinforcement Learning Implementations.” *Journal of Machine Learning Research*, 22(268), 1–8. [JMLR](https://jmlr.org/papers/v22/20-1364.html).
5. Hochreiter, S. & Schmidhuber, J. (1997). “Long Short-Term Memory.” *Neural Computation*, 9(8), 1735–1780. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).
6. Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley. [Publisher page](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021).
7. Sharpe, W. F. (1994). “The Sharpe Ratio.” *The Journal of Portfolio Management*, 21(1), 49–58. [doi:10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501).
8. Lo, A. W. (2002). “The Statistics of Sharpe Ratios.” *Financial Analysts Journal*, 58(4), 36–52. [doi:10.2469/faj.v58.n4.2453](https://doi.org/10.2469/faj.v58.n4.2453).
9. Lo, A. W., Mamaysky, H. & Wang, J. (2000). “Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation.” *The Journal of Finance*, 55(4), 1705–1765. [doi:10.1111/0022-1082.00265](https://doi.org/10.1111/0022-1082.00265).
10. Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). “The Probability of Backtest Overfitting.” *The Journal of Computational Finance*, 20(4). [doi:10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322).
11. Seabold, S. & Perktold, J. (2010). “Statsmodels: Econometric and Statistical Modeling with Python.” *Proceedings of the 9th Python in Science Conference*, 92–96. [doi:10.25080/Majora-92bf1922-011](https://doi.org/10.25080/Majora-92bf1922-011).
12. Yang, H., Liu, X.-Y., Zhong, S. & Walid, A. (2020). “Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy.” In *Proceedings of the First ACM International Conference on AI in Finance (ICAIF ’20)*, 31:1–31:8. [doi:10.1145/3383455.3422540](https://dl.acm.org/doi/abs/10.1145/3383455.3422540).
13. Kutzkov, K. (n.d.). “ARIMA vs Prophet vs LSTM for Time Series Prediction.” *Neptune.ai*. [Article URL](https://neptune.ai/blog/arima-vs-prophet-vs-lstm). The original URL currently redirects.
14. Ait-Sahalia, Y. & Brandt, M. W. (2001). “Variable Selection for Portfolio Choice.” *The Journal of Finance*, 56(4), 1299–1341. [NBER Working Paper 8127](https://www.nber.org/papers/w8127).
