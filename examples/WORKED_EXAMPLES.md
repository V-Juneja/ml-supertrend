# Worked examples of the actual rules

These are arithmetic explanations and one documented account price comparison. They are not a generated OHLCV history or a claim about an observed five-candle trade.

## 1. A probe tests both directions

If the fifth-candle close is one entry ATR above entry, the hypothetical long gains one volatility unit and the matching short loses one. A fall of one entry ATR reverses the signs. Both labels can help the learner even though the account only buys stocks.

The source counts the entry candle as the first of five. An unfinished probe cannot supply its eventual label to an earlier candle.

## 2. A profitable result can receive less weight

Batch sample weight is `confidence / (1 + adverse_excursion_in_ATR_units)`. At identical confidence, adverse excursion of one ATR gives half the weight of zero excursion. Three ATR gives one quarter. The learner distinguishes a smooth profitable path from a path which first moved far against the trade.

## 3. Memory changes the blend

The grid/global blend is `w × context_target + (1-w) × rolling_target`. At the 0.65 context-weight cap, it combines 65% context recommendation and 35% rolling recommendation before the separate batch blend. The 65% is a setting-proposal weight; it does not mean 65% of account equity is invested.

## 4. Adaptation moves settings gradually

An old multiplier of 1.40, bounded target 1.50 and update fraction 0.20 give `1.40 + 0.20 × (1.50 - 1.40) = 1.42`. This illustrates the update arithmetic. The target is derived from completed learning observations; it is not an assumed market forecast.

## 5. Volatility controls band distance

At multiplier 1.4, raw boundaries sit 1.4 ATR from the close-weighted price. If ATR doubles while the multiplier is unchanged, their dollar distance doubles. If the learner increases the multiplier, a larger price move is needed to cross the trend boundary.

## 6. A provider-recorded price and a position-value cap

The DELL provider cache records a 25 September close of $563.17. A $10,000 position-value cap permits `floor(10,000 / 563.17) = 17` whole shares. The actual entry price and measured ATR at entry determine the final order size. This is a sizing reference, not a claim that the strategy traded at that close.

## 7. A gap can exceed a stop budget

A 10% price gap in a position worth 10% of account equity costs approximately 1% of account equity before costs. With 40% gross stock exposure, a uniform 10% stock fall costs about 4%. These are deterministic exposure identities, not assumed future outcomes.
