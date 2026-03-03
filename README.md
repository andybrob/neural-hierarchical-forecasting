# Neural Hierarchical Forecasting at Scale

A from-scratch implementation of neural hierarchical time series forecasting, targeting production-grade performance on high-cardinality e-commerce datasets.

## Objective

Build, optimize, and productionalize a "Global" forecasting model that outperforms standard baselines (Auto-ARIMA, Prophet) on the [M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) or [H&M Personalized Fashion](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset.

---

## Architecture

**Model:** Custom N-HiTS or Temporal Fusion Transformer (TFT)

| Input Type | Examples |
|---|---|
| Static Covariates | Store location, product category |
| Known Future Inputs | Promotions, holidays, planned markdowns |
| Past Observed | Historical sales, returns |

**Loss Function:** Quantile loss (P10 / P50 / P90) for calibrated uncertainty intervals.

**Reconciliation:** Forecast reconciliation logic ensuring SKU-level forecasts aggregate correctly to category and store totals (MinT or bottom-up).

---

## Technical Stack

| Layer | Technology |
|---|---|
| Modeling | PyTorch, NeuralForecast |
| Data Pipeline | PySpark (cold-start via embedding similarity) |
| Hardware Optimization | Custom CUDA kernel or TensorRT (attention / multi-rate sampling) |
| Serving | FastAPI + Docker |
| Monitoring | Champion-Challenger re-evaluation (weekly cadence) |
| CI/CD | GitHub Actions (GPU runner on architecture change) |

---

## Deliverables

- [ ] Backtesting report: sMAPE and MASE vs. NHITS baseline
- [ ] Latency profile: Standard PyTorch vs. hardware-optimized inference
- [ ] Containerized microservice with API spec
- [ ] Champion-Challenger monitoring pipeline

---

## Project Phases

1. **Data & Baseline** — Load M5/H&M data, implement ARIMA/Prophet baseline, establish eval harness
2. **Model Architecture** — Implement N-HiTS or TFT with static/dynamic covariates
3. **Quantile Loss & Reconciliation** — Uncertainty intervals + hierarchical consistency
4. **Hardware Optimization** — CUDA kernel or TensorRT integration
5. **Data Engineering** — PySpark cold-start pipeline
6. **MLOps & Productionalization** — Docker, FastAPI, GitHub Actions, monitoring

---

## Status

> Work in progress. Follow along as each phase is implemented.

## References

- [N-HiTS Paper](https://arxiv.org/abs/2201.12886)
- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- [M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [Forecast Reconciliation (MinT)](https://robjhyndman.com/publications/mint/)
