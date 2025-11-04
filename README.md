# MovieRec

This project delivers a fast, modular movie recommendation system you can train and/or use locally or run via Streamlit. It includes data prep for multiple MovieLens rating datasets, two embedding models (BPR-MF and LightGCN), a popularity baseline, and an interactive app with live personalization knobs, genre/year preferences, a watchlist, and 'why this movie' explanations.

**Live app**: [film-recs.streamlit.app](https://film-recs.streamlit.app/)

---
## Features

- **Datasets**
  - MovieLens ML-1M (fast dev)
  - MovieLens ML-32M (large)
- **Models**
  - Bayesian Personalized Ranking - Matrix Factorization (BPR-MF)
  - LightGCN (mini-batch, sampled)
  - Popularity baseline
- **UI (Streamlit)**
  - Build a starter profile (likes/dislikes/hide, watchlist)
  - Personal knobs (novelty, recency, diversity) that re-rank live
  - Genre and year preferences (soft priors)
  - 'Because you liked ...' explanations
- **Retrieval**
  - Approximate nearest-neighbor (ANN) index using inner-product similarity for fast candidate generation
- **Evaluation**
  - Recall@K • NDCG@K • HitRate@K • Coverage@K • Novelty

---

## Personalization and Profile  

- **Likes/Dislikes/Hide** maintained in session; download/load as JSON
- **Watchlist** with optional score boost
- **Genre sliders** as soft weights
- **Year prior** (Gaussian: μ/σ) with optional decade picks
- **Knobs** for novelty, recency (half-life), and diversity (MMR)

---

## Acknowledgements

Thanks to GroupLens for MovieLens.
Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. *Acm transactions on interactive intelligent systems (tiis)*, 5(4), 1-19.
