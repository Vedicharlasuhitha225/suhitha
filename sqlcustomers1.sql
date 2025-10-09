-select * from SH.CUSTOMERS

--  Questions on WHERE
--1. Find customers born after the year 1990.
--select  CUST_ID from SH.CUSTOMERS where CUST_YEAR_OF_BIRTH >1990

--2. List all male customers (`CUST_GENDER = 'M'`).
--select CUST_GENDER from  SH.CUSTOMERS where  CUST_GENDER = 'M'

--3. Retrieve all female customers (`CUST_GENDER = 'F'`) living in Sydney.
--Select CUST_GENDER, CUST_CITY  From SH.CUSTOMERS Where CUST_GENDER = 'F' AND CUST_CITY = 'Sydney';

--4. Find customers with income level `"G: 130,000 - 149,999"`.
--select CUST_ID from SH.CUSTOMERS where CUST_INCOME_LEVEL = 'G: 130,000 - 149,999'

--5. Get all customers with a credit limit above 10,000.
--select  CUST_ID from SH.CUSTOMERS where CUST_CREDIT_LIMIT >10000

--6. Retrieve customers from the state "California".
--select  CUST_ID from SH.CUSTOMERS where CUST_CITY = 'California'

--7. Find customers who have provided an email address.
--select CUST_ID,CUST_EMAIL from SH.CUSTOMERS where CUST_EMAIL IS NOT NULL

--8. List customers with missing marital status.
--select CUST_ID,CUST_MARITAL_STATUS from SH.CUSTOMERS where CUST_MARITAL_STATUS IS NOT NULL

--9. Find customers whose postal code starts with "53".
--select CUST_ID,CUST_POSTAL_CODE from SH.CUSTOMERS where CUST_POSTAL_CODE LIKE '53%'

--10. Get customers born before 1980 with a credit limit above 5,000.
--select CUST_ID,CUST_YEAR_OF_BIRTH,CUST_CREDIT_LIMIT from SH.CUSTOMERS where CUST_YEAR_OF_BIRTH < 1980 AND CUST_CREDIT_LIMIT >5000

--11. Retrieve customers from Almere or Amersfoort.
--select CUST_ID,CUST_CITY from SH.CUSTOMERS where CUST_CITY = 'Almere' OR CUST_CITY = 'Amersfoort'

--12. Find customers who do not have a credit limit.
--select CUST_ID,CUST_CREDIT_LIMIT from SH.CUSTOMERS where CUST_CREDIT_LIMIT IS NULL

--13. List customers whose phone number starts with "487".
--select CUST_ID, CUST_MAIN_PHONE_NUMBER from SH.CUSTOMERS where CUST_MAIN_PHONE_NUMBER LIKE '487%'

--14. Find married customers with income level `"Medium".
--select CUST_ID,CUST_INCOME_LEVEL,CUST_MARITAL_STATUS from SH.CUSTOMERS where CUST_MARITAL_STATUS='married' AND CUST_INCOME_LEVEL = 'Medium'

--15. Get customers whose last name starts with "G".
--select CUST_ID,CUST_LAST_NAME from SH.CUSTOMERS where CUST_LAST_NAME LIKE 'G%'

--16. Find customers with city_id = 51057.
--select CUST_ID, CUST_CITY_ID from SH.CUSTOMERS where CUST_CITY_ID = '51057'

--17. Retrieve all customers who are valid (`CUST_VALID = 'A'`).
--select CUST_ID,CUST_VALID from SH.CUSTOMERS where CUST_VALID = 'A'

--18. Find customers whose effective start date (`CUST_EFF_FROM`) is after 2020.
--select CUST_ID, CUST_EFF_FROM from SH.CUSTOMERS where CUST_EFF_FROM > TO_DATE('2020' ,'YYYY')

--19. Retrieve customers whose effective end date (`CUST_EFF_TO`) is before 2021.
--select CUST_ID, CUST_EFF_TO from SH.CUSTOMERS where CUST_EFF_TO < TO_DATE('2021','YYYY')

--20. Find customers with credit limit between 5,000 and 9,000.
--select CUST_ID, CUST_CREDIT_LIMIT from SH.CUSTOMERS where CUST_CREDIT_LIMIT BETWEEN 5000 AND 9000

--21. Get all customers from country_id = 101.
--select CUST_ID ,COUNTRY_ID from SH.CUSTOMERS where COUNTRY_ID = '101'

--22. Find customers whose email ends with `"@company.example.com"`.
--select CUST_ID,CUST_EMAIL from SH.CUSTOMERS where CUST_EMAIL LIKE '%@company.example.com'

--23. List customers with `CUST_TOTAL_ID = 52772`.
--select CUST_ID, CUST_TOTAL_ID from SH.CUSTOMERS where CUST_TOTAL_ID = '52772'

--24. Find customers with `CUST_SRC_ID` in (10, 20, 30).
--select CUST_ID, CUST_SRC_ID from SH.CUSTOMERS where CUST_SRC_ID IN (10, 20, 30)+

--25. Retrieve customers who either do not have email or do not have a credit limit.
--select CUST_ID,CUST_EMAIL,CUST_CREDIT_LIMIT from SH.CUSTOMERS where CUST_EMAIL IS NULL OR CUST_CREDIT_LIMIT IS NULL


 -- Questions on GROUP BY and HAVING

--26. Count the number of customers in each city.
--select CUST_CITY, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_CITY 

--27. Find cities with more than 100 customers.
--select  CUST_CITY, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_CITY HAVING COUNT(*) > 100

--28. Count the number of customers in each state.
--Select CUST_STATE_PROVINCE, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_STATE_PROVINCE

--29. Find states with fewer than 50 customers.
--select CUST_STATE_PROVINCE, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_STATE_PROVINCE HAVING COUNT(*) < 50

--30. Calculate the average credit limit of customers in each city.
--SELECT CUST_CITY, AVG(CUST_CREDIT_LIMIT) AVG_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_CITY;

--31. Find cities with average credit limit greater than 8,000.
--select CUST_CITY  , AVG(CUST_CREDIT_LIMIT) AVG_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_CITY HAVING AVG(CUST_CREDIT_LIMIT) > 8000

--32. Count customers by marital status.
--select CUST_MARITAL_STATUS, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_MARITAL_STATUS

--33. Find marital statuses with more than 200 customers.
--select CUST_MARITAL_STATUS ,COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_MARITAL_STATUS HAVING COUNT(*) > 200

--34. Calculate the average year of birth grouped by gender.
--select CUST_GENDER, AVG(CUST_YEAR_OF_BIRTH) CUST_YEAR_OF_BIRTH from SH.CUSTOMERS GROUP BY CUST_GENDER
 
--35. Find genders with average year of birth after 1990.
--select CUST_GENDER, AVG(CUST_YEAR_OF_BIRTH) CUST_YEAR_OF_BIRTH from SH.CUSTOMERS GROUP BY CUST_GENDER HAVING AVG(CUST_YEAR_OF_BIRTH) > 1990
 
--36. Count the number of customers in each country.
--select COUNTRY_ID, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY COUNTRY_ID

--37. Find countries with more than 1,000 customers.
--select COUNTRY_ID, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY COUNTRY_ID HAVING COUNT(*) > 1000

--38. Calculate the total credit limit per state.
--select CUST_STATE_PROVINCE SUM(CUST_CREDIT_LIMIT) TOTAL_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_STATE_PROVINCE

--39. Find states where the total credit limit exceeds 100,000.
--select CUST_STATE_PROVINCE,SUM(CUST_CREDIT_LIMIT) TOTAL_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_STATE_PROVINCE HAVING SUM(CUST_CREDIT_LIMIT) > 100000

--40. Find the maximum credit limit for each income level.
--select CUST_INCOME_LEVEL, MAX(CUST_CREDIT_LIMIT) MAX_CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_INCOME_LEVEL

--41. Find income levels where the maximum credit limit is greater than 15,000.
--select CUST_INCOME_LEVEL, MAX(CUST_CREDIT_LIMIT) MAX_CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_INCOME_LEVEL HAVING MAX(CUST_CREDIT_LIMIT) >15000

--42. Count customers by year of birth.
--select CUST_YEAR_OF_BIRTH, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_YEAR_OF_BIRTH 

--43. Find years of birth with more than 50 customers.
--select CUST_YEAR_OF_BIRTH, COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_YEAR_OF_BIRTH HAVING COUNT(*) > 50

--44. Calculate the average credit limit per marital status.
--select CUST_MARITAL_STATUS, AVG(CUST_CREDIT_LIMIT) CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_MARITAL_STATUS

--45. Find marital statuses with average credit limit less than 5,000.
--select CUST_MARITAL_STATUS, AVG(CUST_CREDIT_LIMIT) CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_MARITAL_STATUS HAVING AVG(CUST_CREDIT_LIMIT) < 5000

--46. Count the number of customers by email domain (e.g., `company.example.com`).
--select SUBSTR(CUST_EMAIL, INSTR(CUST_EMAIL, '@') + 1)  EMAIL_DOMAIN , COUNT(*) NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY SUBSTR(CUST_EMAIL, INSTR(CUST_EMAIL, '@') + 1) ORDER BY NUM_CUSTOMERS DESC

--47. Find email domains with more than 300 customers.
--select SUBSTR(CUST_EMAIL, INSTR(CUST_EMAIL, '@') + 1) AS EMAIL_DOMAIN, COUNT(*)  NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY SUBSTR(CUST_EMAIL, INSTR(CUST_EMAIL, '@') + 1) HAVING COUNT(*) > 300;

--48. Calculate the average credit limit by validity (`CUST_VALID`).
--select CUST_VALID, AVG(CUST_CREDIT_LIMIT) AVG_CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_VALID;

--49. Find validity groups where the average credit limit is greater than 7,000.
--select CUST_VALID, AVG(CUST_CREDIT_LIMIT) CUST_CREDIT_LIMIT from SH.CUSTOMERS GROUP BY CUST_VALID HAVING AVG(CUST_CREDIT_LIMIT) > 7000

--50. Count the number of customers per state and city combination where there are more than 50 customers.
select CUST_STATE_PROVINCE, CUST_CITY, COUNT(*)  NUM_CUSTOMERS from SH.CUSTOMERS GROUP BY CUST_STATE_PROVINCE, CUST_CITY HAVING COUNT(*) > 50;