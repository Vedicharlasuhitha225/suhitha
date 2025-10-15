1.Find the total, average, minimum, and maximum credit limit of all customers.
--SELECT 
   --  SUM(CUST_CREDIT_LIMIT)   AS TOTAL_CREDIT_LIMIT,
     --AVG(CUST_CREDIT_LIMIT)   AS AVERAGE_CREDIT_LIMIT,
     --MIN(CUST_CREDIT_LIMIT)   AS MIN_CREDIT_LIMIT,
     --MAX(CUST_CREDIT_LIMIT)   AS MAX_CREDIT_LIMIT
 --FROM 
     --SH.CUSTOMERS;

     --2.Count the number of customers in each income level.
 --SELECT
    -- CUST_INCOME_LEVEL, COUNT(*) AS CUSTOMER_COUNT FROM SH.CUSTOMERS GROUP BY CUST_INCOME_LEVEL

    --3.shown total credit limit by state and country.
--SELECT
    -- COUNTRY_ID
     --CUST_STATE_PROVINCE,
     --SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --GROUP BY COUNTRY_ID, CUST_STATE_PROVINCE
 --ORDER BY COUNTRY_ID, CUST_STATE_PROVINCE;

 --4.Display average credit limit for each marital status and gender combination.
 --SELECT
   --  CUST_MARITAL_STATUS
--     CUST_GENDER,
  --   AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
--GROUP BY CUST_MARITAL_STATUS
 --ORDER BY CUST_GENDER

 --5.Find the top 3 states with the highest average credit limit.
--SELECT
  --    CUST_STATE_PROVINCE,
    --  AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_STATE_PROVINCE
 --ORDER BY AVG_CREDIT_LIMIT DESC
 --FETCH NEXT 3 ROWS ONLY;

-- 6.Find the country with the maximum total customer credit limit.
 --SELECT
    -- COUNTRY_ID,
    -- SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --GROUP BY  COUNTRY_ID
 --ORDER BY TOTAL_CREDIT_LIMIT DESC
 --FETCH FIRST 1 ROW ONLY

 --7.show the number of customers whose credit limit exceeds their state average.
 --SELECT
   --   COUNT(*)
 --FROM SH.CUSTOMERS C
 --WHERE C.CUST_CREDIT_LIMIT > (
    -- SELECT AVG(C2.CUST_CREDIT_LIMIT)
    -- FROM SH.CUSTOMERS C2
    -- WHERE C2.CUST_STATE_PROVINCE = C.CUST_STATE_PROVINCE
 --)

 --8.Calculate total and average credit limit for customers born after 1980.
 --SELECT
   --   SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT,
     -- AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --WHERE CUST_YEAR_OF_BIRTH > 1980

 --9.Calculate total and average credit limit for customers born after 1980.
 --SELECT
   --   SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT,
     -- AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --WHERE CUST_YEAR_OF_BIRTH > 1980

 --10.List countries where the average credit limit is higher than the global average.
--SELECT
  --   COUNTRY_ID,
    -- AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --GROUP BY COUNTRY_ID
 --HAVING 
   --  AVG(CUST_CREDIT_LIMIT) > (
     --    SELECT AVG(CUST_CREDIT_LIMIT)
       --  FROM SH.CUSTOMERS
    -- )

  --11.Calculate the variance and standard deviation of customer credit limits by country.
 --SELECT
    -- COUNTRY_ID,
     --VAR_SAMP(CUST_CREDIT_LIMIT) AS VARIANCE,
     --STDDEV_SAMP(CUST_CREDIT_LIMIT) AS STANDARD_DEVIATION
 --FROM SH.CUSTOMERS
 --GROUP BY COUNTRY_ID

 --12.Find the state with the smallest range (max–min) in credit limits.
 --SELECT
    -- CUST_STATE_PROVINCE,
     --MAX(CUST_CREDIT_LIMIT) - MIN(CUST_CREDIT_LIMIT) AS CREDIT_LIMIT_RANGE
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_STATE_PROVINCE
 --ORDER BY CREDIT_LIMIT_RANGE ASC
 --FETCH FIRST 1 ROW ONLY

 --13. showthe total number of customers per income level and the percentage contribution of each.
--SELECT
   -- CUST_INCOME_LEVEL,
    -- COUNT(*) AS TOTAL_CUSTOMERS,
    -- ROUND( 100.0 * COUNT() / SUM(COUNT()) OVER (), 2 ) AS PERCENTAGE_CONTRIBUTION
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_INCOME_LEVEL
 --ORDER BY TOTAL_CUSTOMERS DESC

 --14.For each income level, find how many customers have NULL credit limits.
 --SELECT
   --   CUST_INCOME_LEVEL,
     -- COUNT(*) AS NULL_CREDIT_LIMIT_COUNT
 --FROM SH.CUSTOMERS
-- WHERE CUST_CREDIT_LIMIT IS NULL
 --GROUP BY CUST_INCOME_LEVEL
 --ORDER BY NULL_CREDIT_LIMIT_COUNT DESC

 --15.Display countries where the sum of credit limits exceeds 10 million.
 --SELECT
   --  COUNTRY_ID,
     --SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT
--FROM SH.CUSTOMERS
 --GROUP BY COUNTRY_ID
 --HAVING SUM(CUST_CREDIT_LIMIT) > 10000000
 --ORDER BY TOTAL_CREDIT_LIMIT DESC;

 --16.Find the state that contributes the highest total credit limit to its country.
 --SELECT
   --   COUNTRY_ID,     
     -- CUST_STATE_PROVINCE,
      --TOTAL_CREDIT_LIMIT
 --FROM(
  --   SELECT
    --    COUNTRY_ID,
      --   CUST_STATE_PROVINCE,
        -- SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT,
         --ROW_NUMBER() OVER(
             --PARTITION BY COUNTRY_ID
             --ORDER BY SUM(CUST_CREDIT_LIMIT) DESC
         --) AS RN
     --FROM SH.CUSTOMERS
     --GROUP BY COUNTRY_ID, CUST_STATE_PROVINCE
-- )

--17. Show total credit limit per year of birth, sorted by total descending.
 --SELECT
   --  CUST_YEAR_OF_BIRTH,
     --SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_YEAR_OF_BIRTH
 --ORDER BY TOTAL_CREDIT_LIMIT DESC

 --18.Identify customers who hold the maximum credit limit in their respective country.
--SELECT
  --   CUST_ID,
    -- CUST_FIRST_NAME,
     --COUNTRY_ID,
     --CUST_CREDIT_LIMIT
 --FROM SH.CUSTOMERS
 --WHERE (
   --  COUNTRY_ID, CUST_CREDIT_LIMIT) IN (
     --    SELECT 
       --      COUNTRY_ID, 
          --   MAX(CUST_CREDIT_LIMIT)
        --FROM  SH.CUSTOMERS
        -- GROUP BY COUNTRY_ID
    -- );

    --19. Show the difference between maximum and average credit limit per country.
--SELECT
      --COUNTRY_ID,
     --MAX(CUST_CREDIT_LIMIT) AS MAX_CREDIT_LIMIT,
     -- AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT,
      --MAX(CUST_CREDIT_LIMIT) - AVG(CUST_CREDIT_LIMIT) AS DIFFERENCE
 --FROM SH.CUSTOMERS
-- GROUP BY COUNTRY_ID
 --ORDER BY DIFFERENCE DESC

--20. Display the overall rank of each state based on its total credit limit (using GROUP BY + analytic rank).
 --SELECT 
   --  CUST_STATE_PROVINCE,
     --SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT,
     --RANK() OVER (ORDER BY SUM(CUST_CREDIT_LIMIT) DESC) AS rank
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_STATE_PROVINCE

 B. Analytical / Window Functions (30 Questions)
1.Assign row numbers to customers ordered by credit limit descending.
 -- SELECT
--      CUST_ID,
--      CUST_FIRST_NAME,
--      CUST_CREDIT_LIMIT,
--      ROW_NUMBER() OVER(ORDER BY CUST_CREDIT_LIMIT DESC) AS ROW_NUMBER
-- FROM SH.CUSTOMERS

--2.Rank customers within each state by credit limit.
 --SELECT 
   --  CUST_ID,
     --CUST_FIRST_NAME,
     --CUST_STATE_PROVINCE
     --CUST_CREDIT_LIMIT,
     --RANK() 
     --OVER (PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_CREDIT_LIMIT DESC) AS STATE_RANK
 --FROM SH.CUSTOMERS

 --3. Use DENSE_RANK() to find the top 5 credit holders per country.
 --SELECT 
   --  CUST_ID,
    --CUST_FIRST_NAME,
     --COUNTRY_ID,
     --CUST_CREDIT_LIMIT,
     --CREDIT_RANK
 --FROM (
   -- SELECT 
     --   CUST_ID,
       --  CUST_FIRST_NAME,
         --COUNTRY_ID,
        --CUST_CREDIT_LIMIT,
        --DENSE_RANK() OVER (PARTITION BY COUNTRY_ID ORDER BY CUST_CREDIT_LIMIT DESC) AS CREDIT_RANK
     --FROM SH.CUSTOMERS
--)
--WHERE CREDIT_RANK <= 5;

--4. Divide customers into 4 quartiles based on their credit limit using NTILE(4).
--SELECT 
  --   CUST_ID,
    -- CUST_CREDIT_LIMIT,
     --TILE(4) OVER (ORDER BY CUST_CREDIT_LIMIT) AS quartile
 --FROM SH.CUSTOMERS;

 --5.Calculate a running total of credit limits ordered by customer_id.
-- SELECT
  --    CUST_ID
    --  CUST_FIRST_NAME,
     -- CUST_CREDIT_LIMIT,
      --SUM(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS RUNNING_TOTAL
 --FROM SH.CUSTOMERS;

 --6.show cumulative average credit limit by country.
 --SELECT 
   --  CUST_ID,
     --CUST_FIRST_NAME,
     --COUNTRY_ID,
     --CUST_CREDIT_LIMIT,
     --AVG(CUST_CREDIT_LIMIT) OVER (PARTITION BY COUNTRY_ID ORDER BY CUST_ID) AS CUMULATIVE_AVG
 --FROM 
  --   SH.CUSTOMERS;

  --7. Compare each customer’s credit limit to the previous one using LAG().
 --SELECT
   --  CUST_ID,
     --CUST_CREDIT_LIMIT,
     --LAG(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS PREVIOUS_CREDIT_LIMIT,
     --CUST_CREDIT_LIMIT - LAG(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS DIFFERENCE
 --FROM
   --  SH.CUSTOMERS;

  -- 8. Show next customer’s credit limit using LEAD().
 --SELECT 
   --   CUST_ID,
    --  CUST_CREDIT_LIMIT,
    --  LEAD(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS NEXT_CREDIT_LIMIT
 --FROM SH.CUSTOMERS

 --9.Display the difference between each customer’s credit limit and the previous one.
 --SELECT
   --  CUST_ID,
    -- CUST_CREDIT_LIMIT,
     --LAG(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS PREVIOUS_CREDIT_LIMIT,
     --(CUST_CREDIT_LIMIT - LAG(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID)) AS DIFFERENCE
 --FROM SH.CUSTOMERS;

 --10.For each country, display the first and last credit limit using FIRST_VALUE() and LAST_VALUE()
 --SELECT
   --  CUST_ID,
     --COUNTRY_ID,
     --CUST_CREDIT_LIMIT,
     --FIRST_VALUE(CUST_CREDIT_LIMIT) OVER ( PARTITION BY COUNTRY_ID ORDER BY CUST_CREDIT_LIMIT) AS FIRST_VALUE,
     --LAST_VALUE(CUST_CREDIT_LIMIT) OVER ( PARTITION BY CUST_CREDIT_LIMIT ORDER BY COUNTRY_ID) AS LAST_VALUE
 --FROM SH.CUSTOMERS

 --11.Compute percentage rank (PERCENT_RANK()) of customers based on credit limit.
 --SELECT
   --  CUST_ID,
     --CUST_CREDIT_LIMIT,
     --PERCENT_RANK() OVER (ORDER BY CUST_CREDIT_LIMIT) AS CREDIT_LIMIT_PERCENT_RANK
 --FROM SH.CUSTOMERS
 --ORDER BY CUST_CREDIT_LIMIT;

 --12.Show each customer’s position in percentile (CUME_DIST() function).
 --SELECT
   --   CUST_ID,
     -- CUST_CREDIT_LIMIT,
      --CUME_DIST() OVER (ORDER BY CUST_CREDIT_LIMIT) AS PERCENTILE_CREDIT_LIMIT
 --FROM SH.CUSTOMERS

 --13.Display the difference between the maximum and current credit limit for each customer.
--SELECT
  --   CUST_ID,
    -- CUST_CREDIT_LIMIT,
     --MAX(CUST_CREDIT_LIMIT) OVER () AS MAX_CREDIT_LIMIT,
     --(MAX(CUST_CREDIT_LIMIT) OVER () - CUST_CREDIT_LIMIT)AS DIFFERENCE
 --FROM
   --  SH.CUSTOMERS
 --ORDER BY
  -- CUST_ID;

  --14.Rank income levels by their average credit limit.
 --SELECT
   --   CUST_INCOME_LEVEL,
    -- AVG(CUST_CREDIT_LIMIT) AS AVG_CREDIT_LIMIT,
      --RANK() OVER (ORDER BY AVG(CUST_CREDIT_LIMIT) DESC) AS RANK_BY_AVG_CREDIT
 --FROM SH.CUSTOMERS
 --GROUP BY CUST_INCOME_LEVEL

 --15.Calculate the average credit limit over the last 10 customers (sliding window).
--SELECT 
  --   CUST_ID,
    -- CUST_CREDIT_LIMIT,
    -- AVG(CUST_CREDIT_LIMIT) 
      --   OVER (ORDER BY CUST_ID
        --     ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        -- ) AS AVG_LAST_10_CUSTOMERS
 --FROM SH.CUSTOMERS
 --ORDER BY CUST_ID;

 --16.For each state, calculate the cumulative total of credit limits ordered by city.
 --SELECT
   --  CUST_ID,
     --CUST_STATE_PROVINCE_ID,
     --CUST_CREDIT_LIMIT,
     --CUST_CITY,
     --SUM(CUST_CREDIT_LIMIT) OVER (PARTITION BY CUST_STATE_PROVINCE_ID ORDER BY CUST_CITY) AS CUMULATIVE_CREDIT
 --FROM SH.CUSTOMERS;

 --17. Find customers whose credit limit equals the median credit limit (use PERCENTILE_CONT(0.5)).
--SELECT *
 --FROM ( 
   --  SELECT
     --    CUST_ID,
       --  CUST_FIRST_NAME,
         --CUST_LAST_NAME,
         --CUST_CREDIT_LIMIT,
         --PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CUST_CREDIT_LIMIT) OVER () AS median_credit
     --FROM SH.CUSTOMERS
-- )
 --WHERE CUST_CREDIT_LIMIT = median_credit

 --18.Display the highest 3 credit holders per state using ROW_NUMBER() and PARTITION BY.
 --SELECT *
 --FROM (
   --  SELECT
     --    CUST_ID,
       --  CUST_STATE_PROVINCE,
         --CUST_CREDIT_LIMIT,
         --ROW_NUMBER() OVER (PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_CREDIT_LIMIT) AS rn
     --FROM SH.CUSTOMERS
-- )
--WHERE rn <= 3;

19.Identify customers whose credit limit increased compared to previous row (using LAG).
 --SELECT *
 --FROM (
   --  SELECT
     --    CUST_ID,
       --  CUST_CREDIT_LIMIT,
         --LAG(CUST_CREDIT_LIMIT) OVER (ORDER BY CUST_ID) AS PREVIOUS_CREDIT_LIMIT
     --FROM SH.CUSTOMERS
-- )
 --WHERE CUST_CREDIT_LIMIT > PREVIOUS_CREDIT_LIMIT;

 --20.calculate moving average of credit limits with a window of 3.
--SELECT
  --   cust_id,
    -- cust_credit_limit,
     --AVG(cust_credit_limit) OVER (
       --  ORDER BY cust_id
         --ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
     --) AS moving_avg_3
 --FROM SH.CUSTOMERS

 --21.Show cumulative percentage of total credit limit per country.
--SELECT
  --    COUNTRY_ID,
    --  CUST_CREDIT_LIMIT,
      --SUM(CUST_CREDIT_LIMIT) OVER (PARTITION BY COUNTRY_ID ORDER BY CUST_ID)/
     -- SUM(CUST_CREDIT_LIMIT) OVER (PARTITION BY COUNTRY_ID) * 100 AS CUMULATIVE_PCT
 --FROM SH.CUSTOMERS

 --22.Rank customers by age (derived from CUST_YEAR_OF_BIRTH).
 --SELECT
   --  CUST_ID,
     --CUST_YEAR_OF_BIRTH,
     --(EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) AS age,
     --RANK() OVER (ORDER BY (EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) DESC) AS age_rank
 --FROM SH.CUSTOMERS

 23.Calculate difference in age between current and previous customer in the same state.
 --SELECT 
   --  CUST_ID,
     --CUST_STATE_PROVINCE,
     --EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH AS AGE,
     --LAG(EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) 
       --  OVER (PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_ID) AS PREVIOUS_AGE,
     --(EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) - 
     --LAG(EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) 
       --  OVER (PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_ID) AS AGE_DIFF
 --FROM SH.CUSTOMERS;

 --24.Use RANK() and DENSE_RANK() to show how ties are treated differently.
 --SELECT 
   --  CUST_ID,
     --CUST_STATE_PROVINCE,
     --CUST_INCOME_LEVEL,
     --RANK() OVER(PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_INCOME_LEVEL) AS  RANK_ORDER,
     --DENSE_RANK() OVER(PARTITION BY CUST_STATE_PROVINCE ORDER BY CUST_INCOME_LEVEL) AS DENSE_RANK_ORDER
 --FROM SH.CUSTOMERS

 --25.Compare each state’s average credit limit with country average using window partition.
 --SELECT
    --  COUNTRY_ID,
     -- CUST_ID,
     -- CUST_STATE_PROVINCE,
     --CUST_CREDIT_LIMIT,
      --AVG(CUST_CREDIT_LIMIT) OVER (PARTITION BY CUST_STATE_PROVINCE) AS STATE_AVG,
      --AVG(CUST_CREDIT_LIMIT) OVER (PARTITION BY COUNTRY_ID) AS COUNTRY_AVG
 --FROM SH.CUSTOMERS

 --26. Show total credit per state and also its rank within each country.
 --SELECT
   --   COUNTRY_ID,
     --CUST_STATE_PROVINCE,
      --SUM(CUST_CREDIT_LIMIT) AS TOTAL_CREDIT_LIMIT,
      --RANK() OVER(PARTITION BY COUNTRY_ID ORDER BY SUM(CUST_CREDIT_LIMIT) DESC) AS state_rank_in_country
 --FROM SH.CUSTOMERS
 --GROUP BY COUNTRY_ID, CUST_STATE_PROVINCE;

 --27. Find customers whose credit limit is above the 90th percentile of their income level.
--SELECT *
 --FROM (
   --  SELECT 
     --    CUST_ID,
       --  CUST_FIRST_NAME,
         --CUST_LAST_NAME,
         --CUST_INCOME_LEVEL,
         --CUST_CREDIT_LIMIT,
         --PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY CUST_CREDIT_LIMIT) OVER (PARTITION BY CUST_INCOME_LEVEL) AS pct_90
     --FROM SH.CUSTOMERS
--)
 --WHERE CUST_CREDIT_LIMIT > pct_90;

 28. Display top 3 and bottom 3 customers per country by credit limit.
 --SELECT *
 --FROM (
   --  SELECT
     --    CUST_ID,
      --   COUNTRY_ID,
        -- CUST_CREDIT_LIMIT,
        -- ROW_NUMBER() OVER (PARTITION BY COUNTRY_ID ORDER BY CUST_CREDIT_LIMIT DESC) AS rn_top,
         --ROW_NUMBER() OVER (PARTITION BY COUNTRY_ID ORDER BY CUST_CREDIT_LIMIT ASC) AS rn_bottom
     --FROM SH.CUSTOMERS
-- )
 --WHERE rn_top <= 3 OR rn_bottom <= 3;

 --29.Calculate rolling sum of 5 customers’ credit limit within each country.
-- SELECT
  --    CUST_ID,
    --  COUNTRY_ID,
      --CUST_CREDIT_LIMIT,
      --SUM(CUST_CREDIT_LIMIT) OVER(PARTITION BY COUNTRY_ID ORDER BY CUST_ID ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS rolling_5_customer_sum
 --FROM SH.CUSTOMERS
 --ORDER BY COUNTRY_ID, CUST_ID;


--30.For each marital status, display the most and least wealthy customers using analytical functions.
 --SELECT *
 --FROM (
   --  SELECT
     --     CUST_ID,
       --   CUST_MARITAL_STATUS,
        --  CUST_CREDIT_LIMIT,
        --  RANK() OVER (PARTITION BY CUST_MARITAL_STATUS ORDER BY CUST_CREDIT_LIMIT DESC) AS rank_most_wealthy,
          --RANK() OVER (PARTITION BY CUST_MARITAL_STATUS ORDER BY  CUST_CREDIT_LIMIT ASC) as rank_least_wealth
     --FROM SH.CUSTOMERS
 --)
 --WHERE rank_most_wealthy = 1 OR rank_least_wealth = 1

 C. Conditional, CASE, and DECODE (10 Questions)
--1. Categorize customers into income tiers: Platinum, Gold, Silver, Bronz
 --SELECT
   --  CUST_ID,
    -- CUST_INCOME_LEVEL,
     --CASE
       -- WHEN CUST_INCOME_LEVEL = 'E' THEN 'Platinum'
        --WHEN CUST_INCOME_LEVEL = 'D' THEN 'Gold'
         --WHEN CUST_INCOME_LEVEL = 'C' THEN 'Silver'
         --ELSE 'Bronze'
     --END AS INCOME_TIER
 --FROM
   -- SH.CUSTOMERS;

   2.Display “High”, “Medium”, or “Low” income categories based on credit limit.
-- SELECT
--     CUST_ID,
--     CUST_CREDIT_LIMIT,
--     CASE
--         WHEN CUST_CREDIT_LIMIT >= 10000 THEN 'High'
--         WHEN CUST_CREDIT_LIMIT BETWEEN 5000 AND 9999 THEN 'Medium'
--         WHEN CUST_CREDIT_LIMIT < 5000 THEN 'Low'
--         ELSE 'Unknown'
--     END AS INCOME_CATEGORY
-- FROM
--     SH.CUSTOMERS;

3. Replace NULL income levels with “Unknown” using NVL.
-- SELECT
--     CUST_CREDIT_LIMIT,
--     CUST_ID,
--     CUST_INCOME_LEVEL,
--     CASE
--         WHEN CUST_CREDIT_LIMIT > (SELECT AVG(CUST_CREDIT_LIMIT) FROM SH.CUSTOMERS)
--         THEN 'AVG'
--         ELSE 'BELOW AVG'
--     END AS STATUS
-- FROM SH.CUSTOMERS


        4. Show customer details and mark whether they have above-average credit limit or not.
-- SELECT
--     CUST_ID,
--     CUST_CREDIT_LIMIT,
--     CASE
--         WHEN CUST_CREDIT_LIMIT > (
--             SELECT
--                 AVG(CUST_CREDIT_LIMIT)
--             FROM
--                 SH.COUNTRIES
--         ) THEN
--             'ABOVE AVG'
--         ELSE
--             'BELOW AVG'
--     END AS STATUS
-- FROM
--     SH.CUSTOMERS;

5. Use DECODE to convert marital status codes (S/M/D) into full text.
-- SELECT
--      CUST_ID,
--      CUST_MARITAL_STATUS,
--      DECODE(CUST_MARITAL_STATUS, 'S', 'SINGLE', 'M', 'MARRIED', 'D', 'DIVORCED', 'UNKNOWN') AS  MARITAL_STATUS
-- FROM SH.CUSTOMERS


6.Use CASE to show age group (≤30, 31–50, >50) from CUST_YEAR_OF_BIRTH.
-- SELECT
--      CUST_ID,
--      CUST_YEAR_OF_BIRTH,
--      CASE
--         WHEN(EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) <= 30 THEN '≤30'
--         WHEN(EXTRACT(YEAR FROM SYSDATE)- CUST_YEAR_OF_BIRTH) BETWEEN 31 AND 50 THEN '31-50'
--         ELSE
--            '> 50'
--      END AS AGE_GROUP
-- FROM SH.CUSTOMERS

7. Label customers as “Old Credit Holder” or “New Credit Holder” based on year of birth < 1980.
-- SELECT
--      CUST_ID,
--      CUST_YEAR_OF_BIRTH,
--      CASE
--         WHEN CUST_YEAR_OF_BIRTH < 1980 THEN 'OLD CREDIT HOLDER'
--         ELSE
--            'NEW CREDIT HOLDER'
--         END AS CREDIT_HOLDER_CATEGORY
-- FROM SH.CUSTOMERS

8. Create a loyalty tag — “Premium” if credit limit > 50,000 and income_level = ‘E’.
-- SELECT
--      CUST_CREDIT_LIMIT,
--      CUST_INCOME_LEVEL,
--      CUST_ID,
--      CASE 
--         WHEN CUST_CREDIT_LIMIT > 50000 AND CUST_INCOME_LEVEL = 'E' THEN 'PREMIUM'
--         ELSE 'STANDARD'
--     END AS LOYALTY_TAG
-- FROM SH.CUSTOMERS

9. Assign grades (A–F) based on credit limit range using CASE.
-- SELECT
--      CUST_ID,
--      CUST_CREDIT_LIMIT,
--      CUST_FIRST_NAME,
--      CASE
--         WHEN CUST_CREDIT_LIMIT >= 90000 THEN 'A'
--         WHEN CUST_CREDIT_LIMIT BETWEEN 70000 AND 89999 THEN 'B'
--         WHEN CUST_CREDIT_LIMIT BETWEEN 50000 AND 69999 THEN 'C'
--         WHEN CUST_CREDIT_LIMIT BETWEEN 30000 AND 49999 THEN 'D'
--         WHEN CUST_CREDIT_LIMIT BETWEEN 10000 AND 29999 THEN 'E'
--         ELSE 'F'
--     END AS CREDIT_GRADE
-- FROM SH.CUSTOMERS;



     10. Show country, state, and number of premium customers using conditional aggregation.
-- SELECT
--     COUNTRY_ID,
--     CUST_STATE_PROVINCE,
--     COUNT(
--         CASE
--             WHEN CUST_CREDIT_LIMIT > 50000 THEN 1
--         END
--     ) AS PREMIUM_CUSTOMERS
-- FROM
--     SH.CUSTOMERS
-- GROUP BY
--     COUNTRY_ID,
--     CUST_STATE_PROVINCE;


D. Date & Conversion Functions (10 Questions)
--1. Convert CUST_YEAR_OF_BIRTH to age as of today.
-- SELECT  
--      CUST_ID,
--      CUST_YEAR_OF_BIRTH,
--      EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH AS AGE
-- FROM SH.CUSTOMERS

2.Display all customers born between 1980 and 1990.
-- SELECT
--     CUST_ID,
--     CUST_FIRST_NAME,
--     CUST_YEAR_OF_BIRTH
-- FROM
--     SH.CUSTOMERS
-- WHERE
--     CUST_YEAR_OF_BIRTH BETWEEN 1980 AND 1990;

3.Format date of birth into “Month YYYY” using TO_CHAR.
-- SELECT
--     CUST_ID,
--     TO_CHAR(TO_DATE(CUST_YEAR_OF_BIRTH, 'YYYY'), 'Month YYYY') AS FORMATTED_DOB
-- FROM
--     SH.CUSTOMERS;


 4.Convert income level text (like 'A: Below 30,000') to numeric lower limit.
-- SELECT
--     CUST_ID,
--     CUST_FIRST_NAME,
--     CUST_LAST_NAME,
--     CUST_INCOME_LEVEL,
--     CASE
--         WHEN CUST_INCOME_LEVEL LIKE '%Below 30%' THEN 0
--         WHEN CUST_INCOME_LEVEL LIKE '%30,000%' THEN 30000
--         WHEN CUST_INCOME_LEVEL LIKE '%70,000%' THEN 70000
--         WHEN CUST_INCOME_LEVEL LIKE '%100,000%' THEN 100000
--         WHEN CUST_INCOME_LEVEL LIKE '%Above 100%' THEN 100000
--         ELSE NULL
--     END AS INCOME_LOWER_LIMIT
-- FROM SH.CUSTOMERS
-- ORDER BY INCOME_LOWER_LIMIT;

5.Display customer birth decades (e.g., 1960s, 1970s).
--  SELECT
--       CUST_ID,
--       CUST_FIRST_NAME,
--       CUST_LAST_NAME,
--       TO_CHAR(TRUNC(CUST_YEAR_OF_BIRTH /10)*10) || 's' AS BIRTH_DECADES
-- FROM SH.CUSTOMERS
-- ORDER BY  BIRTH_DECADES

6. Show customers grouped by age bracket (10-year intervals).
-- SELECT
--     TRUNC((EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) / 10) * 10 AS AGE_BRACKET_START,
--     TRUNC((EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) / 10) * 10 + 9 AS AGE_BRACKET_END,
--     COUNT(*) AS NUM_CUSTOMERS
-- FROM SH.CUSTOMERS
-- GROUP BY TRUNC((EXTRACT(YEAR FROM SYSDATE) - CUST_YEAR_OF_BIRTH) / 10) * 10
-- ORDER BY AGE_BRACKET_START;

7.Convert country_id to uppercase and state name to lowercase.
-- SELECT
--      COUNTRY_ID,
--      CUST_STATE_PROVINCE,
--      UPPER(COUNTRY_ID) AS UPPER_COUNTRY_ID,
--      LOWER(CUST_STATE_PROVINCE) AS LOWER_STATE_NAME
-- FROM SH.CUSTOMERS
-- ORDER BY LOWER_STATE_NAME,UPPER_COUNTRY_ID

8.Show customers where credit limit > average of their birth decade.
-- SELECT
--      CUST_ID,
--      CUST_CREDIT_LIMIT,
--      CUST_FIRST_NAME,
--      CUST_LAST_NAME,
--          TRUNC(CUST_YEAR_OF_BIRTH / 10) * 10 || 's' AS BIRTH_DECADE
-- FROM SH.CUSTOMERS c
-- WHERE CUST_CREDIT_LIMIT > (
--     SELECT AVG(CUST_CREDIT_LIMIT)
--     FROM SH.CUSTOMERS
--     WHERE TRUNC(CUST_YEAR_OF_BIRTH / 10) * 10 = TRUNC(c.CUST_YEAR_OF_BIRTH / 10) * 10
-- )
-- ORDER BY BIRTH_DECADE, CUST_CREDIT_LIMIT DESC;

--9.Convert all numeric credit limits to currency format $999,999.00.
-- SELECT
--      CUST_ID,
--      CUST_CREDIT_LIMIT,
--      TO_CHAR(CUST_CREDIT_LIMIT, '$999,999.00') AS NUMERIC_CREDIT_LIMIT
-- FROM SH.CUSTOMERS
-- ORDER BY CUST_CREDIT_LIMIT DESC;

--10.Find customers whose credit limit was NULL and replace with average (using NVL).
-- SELECT
--      CUST_ID,
--      CUST_FIRST_NAME,
--      CUST_LAST_NAME,
--      CUST_CREDIT_LIMIT,
--      NVL(CUST_CREDIT_LIMIT,(SELECT AVG(CUST_CREDIT_LIMIT) FROM SH.CUSTOMERS))AS CREDIT_LIMIT_FILLED
-- FROM SH.CUSTOMERS
-- ORDER BY CREDIT_LIMIT_FILLED DESC;

-- E. String Functions (10 Questions)
--1.Show customers whose first and last name start with the same letter.
-- SELECT
--      CUST_ID,
--      CUST_FIRST_NAME,
--      CUST_LAST_NAME
-- FROM SH.CUSTOMERS
-- WHERE UPPER(SUBSTR(CUST_FIRST_NAME, 1, 1)) = UPPER(SUBSTR(CUST_LAST_NAME, 1, 1));

--2.Display full names in “Last, First” format.
-- SELECT
--     CUST_ID,
--     CUST_LAST_NAME,
--     CUST_FIRST_NAME,
--     CUST_LAST_NAME || ', ' || CUST_FIRST_NAME AS FULL_NAME
-- FROM SH.CUSTOMERS
-- WHERE UPPER(SUBSTR(CUST_FIRST_NAME, 1, 1)) = UPPER(SUBSTR(CUST_LAST_NAME, 1, 1));

-- --3.Find customers whose last name ends with 'SON'.
-- SELECT
--      CUST_ID,
--      CUST_LAST_NAME
-- FROM SH.CUSTOMERS
-- WHERE UPPER(SUBSTR(CUST_LAST_NAME,-3)) = 'SON'

--4.Display length of each customer’s full name.
-- SELECT
--     CUST_ID,
--     CUST_FIRST_NAME,
--     CUST_LAST_NAME,
--     CUST_FIRST_NAME || ' ' || CUST_LAST_NAME AS FULL_NAME
-- FROM SH.CUSTOMERS
-- WHERE LENGTH(CUST_FIRST_NAME || CUST_LAST_NAME) > 10;


--5.Replace vowels in customer names with '*'.
-- SELECT CUST_ID,
--        CUST_FIRST_NAME,
--        REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
--          CUST_FIRST_NAME,
--          'A','*'),
--          'E','*'),
--          'I','*'),
--          'O','*'),
--          'U','*') AS MASKED_FIRST_NAME
-- FROM SH.CUSTOMERS;

--6.Show customers whose income level description contains ‘90’.
SELECT CUST_ID,
       CUST_FIRST_NAME,
       CUST_LAST_NAME,
       CUST_INCOME_LEVEL
FROM SH.CUSTOMERS
WHERE INSTR(CUST_INCOME_LEVEL, '90') > 0;

--7.Display initials of each customer (first letters of first and last name).
-- SELECT CUST_ID,
--        CUST_FIRST_NAME,
--        CUST_LAST_NAME,
--        SUBSTR(CUST_FIRST_NAME, 1, 1) || SUBSTR(CUST_LAST_NAME, 1, 1) AS INITIALS
-- FROM SH.CUSTOMERS;
--8.Concatenate city and state to create full address.
-- SELECT
--      CUST_CITY,
--      CUST_STATE_PROVINCE,
--      CUST_CITY || ',' || CUST_STATE_PROVINCE AS FULL_ADDRESS
-- FROM SH.CUSTOMERS

--9.Extract numeric value from income level using REGEXP_SUBSTR.
-- SELECT
--     CUST_INCOME_LEVEL,
--     REGEXP_SUBSTR(CUST_INCOME_LEVEL, '[0-9]+') AS INCOME_NUMERIC
-- FROM SH.CUSTOMERS;


--10.Count how many customers have a 3-letter first name.
--SELECT 
--     CUST_FIRST_NAME,
--     CUST_LAST_NAME
-- FROM SH.CUSTOMERS
-- WHERE LENGTH(CUST_FIRST_NAME) = 3;
















