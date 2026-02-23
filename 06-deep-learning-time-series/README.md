# Deep Learning for Financial Time Series Forecasting

**Author:** Jose Orlando Bobadilla Fuentes | CQF
**Category:** Machine Learning in Finance | Tier 2 - Differentiator

---

## Theoretical Foundation

### LSTM (Long Short-Term Memory)

Hochreiter & Schmidhuber (1997). Addresses vanishing gradient problem via gating:
- Forget gate: `f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)`
- Input gate: `i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)`
- Cell state: `C_t = f_t * C_{t-1} + i_t * tanh(W_C * [h_{t-1}, x_t] + b_C)`
- Output gate: `o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)`
- Hidden state: `h_t = o_t * tanh(C_t)`

### GRU (Gated Recurrent Unit)

Cho et al. (2014). Simplified gating with fewer parameters:
- Reset gate: `r_t = sigmoid(W_r * [h_{t-1}, x_t])`
- Update gate: `z_t = sigmoid(W_z * [h_{t-1}, x_t])`
- `h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W * [r_t * h_{t-1}, x_t])`

### Transformer (Self-Attention)

Vaswani et al. (2017). Attention mechanism:
`Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V`

Advantages: parallelizable, captures long-range dependencies.

### Walk-Forward Validation

Anchored expanding window: train on [0, t], test on [t, t+h].
Prevents look-ahead bias inherent in standard cross-validation.

---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
- Fischer, T., & Krauss, C. (2018). Deep Learning with LSTM for Financial Market Predictions. EJOR.
