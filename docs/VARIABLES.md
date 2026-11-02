# Formula glossary

All candle calculations use information available at the completed candle's close.

| Symbol | Meaning | Units |
| --- | --- | --- |
| t | Current completed candle; t-1 is previous candle | Candle index |
| H_t, L_t, C_t | Candle high, low and close | Dollars/share |
| x_t | Close-weighted price `(H_t + L_t + 2C_t)/4` | Dollars/share |
| TR_t | `max(H_t-L_t, abs(H_t-C_(t-1)), abs(L_t-C_(t-1)))` | Dollars/share |
| A_t | `A_(t-1) + (TR_t-A_(t-1))/ell_t` | Dollars/share |
| ell_t | Adaptive smoothing length; baseline 24 | Candles |
| m_t | Adaptive band-width multiplier; baseline 1.4 | Dimensionless |
| b_t minus / plus | Raw boundaries `x_t - m_t A_t` / `x_t + m_t A_t` | Dollars/share |
| s | Hypothetical direction: +1 long, -1 short | Sign |
| P | Probe entry price | Dollars/share |
| C | Close at the end of the probe's fifth candle | Dollars/share |
| A_entry | Smoothed true range at probe entry | Dollars/share |
| g | Probe result `s(C-P)/A_entry` | Entry-volatility units |
| theta | Adaptive setting, such as multiplier or length | Setting-dependent |
| alpha | Fraction of target change applied | Dimensionless |
| theta_target | Rounded, clipped and blended proposed setting | Same as theta |

RSI is `100 - 100/(1 + G/D)`, where G and D are Wilder-smoothed positive and negative close changes over 14 candles. Zero-gain/loss edge handling follows the implementation. RSI below 30 is the declared oversold threshold; it is not a probability of reversal.

The raw lower boundary trails upward while the previous close is above the previous lower boundary. The raw upper trails downward while the previous close is below the previous upper. Otherwise that boundary resets. The source's legacy variable names `upper` and `lower` refer to subtractive/additive branches and are unintuitive; use the arithmetic, not those names, to interpret them.

## Position sizing in the separate account overlay

`d_i = 2 × ATR_24,i`

`q_i = floor(min(0.10 E/P_i, 0.005 E/d_i))`

E is marked account equity, P_i the entry price of stock i, ATR_24,i its fixed 24-candle smoothed true range, d_i the dollar stop distance and q_i whole shares. Remaining account and sector capacity can reduce q_i further. For a $10,000 value cap and the provider-recorded DELL close of $563.17, the value term permits 17 whole shares. The measured entry ATR can reduce that quantity through the separate loss cap.

Internal probe settings do not override these declared account limits.
