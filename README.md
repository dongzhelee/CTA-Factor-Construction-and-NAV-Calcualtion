# CTA-Factor-Construction-and-NAV-Calcualtion
This repository defines many functions for constructing various CTA factors and calculating their NAV, using commodity futures data at the market level.

1. Frequency: Daily

2. Explanation of key parameters: (The following "future" means "main contract")
   MarketChange: Return of future (price change rate)
   FactorData: Price of future
   UniverseData: Certain rules of filtering, such as backtest period > 12 months, commodity, certain categories
   Period: Lookback period
   HoldingPeriod: Holding period
   Leverage: Leverage ratio
   DC_Close: Main contract close price
   SDC_Close: Second main contract close price
   DC_DaysUntil: The number of days until the main contract expires
   SDC_DaysUntil: The number of days until the second main contract expires
   Warehouse: Warehouse receipt quantity
   Highp: Highest future price
   Volume: Trading volume
   
3. The factor construction logic is followed by:
   Documentation on Huofuniu Futures Style Factor Explanation
   Research Report of the Financial Engineering Group of CITIC Futures Research Institute
   My internship experience
