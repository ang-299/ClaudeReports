-- Market data for ISINs to add to the basket builder (basket_builder.html).
-- Replace the ISIN list, run in Snowflake, export the result as CSV and load it with
-- "Load market data CSV" in the Custom ISINs layer. Headers match the BAML positions file.
-- Source: DBT_PROD.MART (ICE BofA bond analytics, latest as_of_date). Covers the full ICE universe, not only H0A0.
WITH latest AS (
  SELECT MAX(as_of_date) AS d FROM DBT_PROD.MART.FACT_BAML_BOND_ANALYTICS_TIMESERIES
)
SELECT
  b.isin                                   AS "Isin",
  COALESCE(i.issuer_name, i.ks_issuer_name) AS "Issuer",
  b.bond_ticker || ' ' || TO_CHAR(b.par_weighted_coupon) || ' ' || TO_CHAR(b.bond_maturity_date, 'MM/DD/YY') AS "Security",
  COALESCE(NULLIF(b.ks_gics_sector, ''), i.ks_sector) AS "KS GICS Sector",  -- bond-level value is often blank; falls back to issuer-level
  b.industry_lvl_1_desc                    AS "IndustryLvl1Desc",
  b.industry_lvl_2_desc                    AS "IndustryLvl2Desc",
  b.industry_lvl_3_desc                    AS "IndustryLvl3Desc",
  b.industry_lvl_4_desc                    AS "IndustryLvl4Desc",
  CASE WHEN r.composite_rating_rank <= 10 THEN 'IG'
       WHEN r.composite_rating_rank <= 14 THEN 'BB'
       WHEN r.composite_rating_rank <= 17 THEN 'B'
       WHEN r.composite_rating_rank <= 21 THEN 'CCC and below'
  END                                      AS "Rating Bucket",   -- BB = ranks 11-14, B = 15-17; verify against your bucket convention
  b.bond_country_code                      AS "Iso Country Code",
  'Q' || QUARTER(b.bond_maturity_date)     AS "Quarter of Maturity Date",
  a.price                                  AS "Price",
  a.effective_duration                     AS "Effective Duration",
  a.effective_yield                        AS "Effective Yield",
  a.oas                                    AS "Option Adjusted Spread",
  a.yield_to_maturity                      AS "Yield To Maturity",
  a.yield_to_worst                         AS "Yield To Worst",
  a.as_of_date                             AS "As Of"
FROM DBT_PROD.MART.FACT_BAML_BOND_ANALYTICS_TIMESERIES a
JOIN latest l ON a.as_of_date = l.d
JOIN DBT_PROD.MART.DIM_BAML_BOND b ON b.baml_security_id = a.baml_security_id
LEFT JOIN DBT_PROD.MART.DIM_ISSUER i ON i.id_bb_global_company = a.id_bb_global_company
LEFT JOIN DBT_PROD.MART.FACT_CREDIT_RATING_CURRENT r ON r.baml_security_id = a.baml_security_id
WHERE b.isin IN ('US68245XAJ81', 'US68288AAA51');   -- <-- your ISINs
