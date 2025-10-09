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










