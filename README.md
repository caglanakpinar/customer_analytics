# CustomerAnalytics
 
---

### A. Understanding the concept of CustomerAnalytics

This platform helps the business related to e-commerce. It also helps the business which has users (B2C) and has their data related to engagement to the business. 
At this platform, There are parts which are Exploratory analysis and Machine Learning Works. These parts have sub analysis and works. These parts also have individual built-in dashboards in or to visualize the results of the outputs at them.  

CustomerAnalytics platform needs to be supported with data-source connections, data sources of columns matching to its data sources (Sessions, Customers, and Products), accessible ElasticSearch Service, and triggering data to the ElasticSearch by the users. The platform supports the data source connections which are ‘.csv’, ‘PostgreSQL’, ‘GCLoud Bigquery’, and ‘AWS redShift’. Sessions and Customers data sources are required, but the Products data source is optional (without products data source, you can need to see the real charts for A/B Test Product, Product analytics). In addition to that additional actions are optional (you probably see the additional actions in Funnels). If you need to check the data source with a dimension (region, location,  business type, etc), it is possible to assign dimensions columns at the session data source. You may filter each dashboard with the given dimensions after that. Promotions of each order can be added to the data sources (optional). After that, you may see A/B Test Promotions, promotion individual searching (check G. Searching) with real data sets.

At ElasticSearch, there are created 3 indexes. Orders Index (orders) is fetched data from Session and Products Data sources with columns. These Orders Index columns are id (session ID), client (client ID), session_start_date (timestamp), end_date (session end timestamp, optional), payment amount (float numeric amount of purchased orders), discount amount (discount of purchased orders, optional), promotion_id (optional), dimension (optional), has_purchased(if True, session is ended with purchase, if False, session is ended with no o-purchase). Order Index of Product data source columns is a basket which is an object with keys are also the products data source of columns which are order_id (Session ID with basket), price (the price of the product), category (category of the Product), product (product name of ID). Downloads Index (downloads) is also created with customer Data source in order to assign data for customers when they are first engaged to the business or when they downloaded or signup. 3rd Index is Reports Index (reports) which is stored with KPIs data sets, outputs, and results of Exploratory Analysis and Machine Learning Works. Each report is a document of the index with a separate date. report_name represents the main analysis, type represents sub-analysis and report_Date represents the created date of the report or analysis.

    
---

### B. How to use CustomerAnalytics

![image](https://user-images.githubusercontent.com/26736844/128701752-613908b4-1ee1-42c3-8bd9-9cf47e3aa6b1.png)

There are 3 steps for initializing CustomerAnalytics;

 - Configuration of ElasticSearch Connection
 - Sessions & Customers & Products Data source Connections
 - Initialize the Schedule process 

All data sets (Sessions, Customers, Products) are transferred to ElasticSearch Indexes. During the processes, there are temporary reports which are also stored at ElasticSearch. In order to run the platform, an accessible ElasticSearch Connection is required. In addition to that, there must be a folder with write/delete permission which the platform will use and store the reports with given folders there.


![image](https://user-images.githubusercontent.com/26736844/128701903-86baed38-b518-48f7-8127-9d777803ed04.png)

Once you click on ‘Connect’, the platform will check for both connectivities of ElasticSearch host/url/port and temporary folder of accessibility.

It is also crucial that RAM and CPU consumption are checked by the user. While ElasticSerach is running, it might be useful to use with command; 

    ES_JAVA_OPTS="-Xms4g -Xmx4g" ./elasticsearch

    
<img width="833" alt="Screen Shot 2021-08-09 at 14 53 19" src="https://user-images.githubusercontent.com/26736844/128702097-8443ff79-56f4-4b5a-80cd-c4c98180b23d.png">


When indexes are created there are some constant parameters that are stored at ***configs.py***. These constants are used at ***settings*** when indexes are created. These constants are;


-  ***number_of_shard:*** The number of primary shards that an index should have. Defaults to 2.
-  ***number_of_replicas:*** The number of replicas each primary shard has. Defaults to 0.
-  ***total field.limit:*** maximum number of fields at document.
-  ***max_result_window:*** The maximum value of ***from + size*** for searches to this index. Defaults to 10000000.


These are also fields that are mapping with their types;

-  ***date:*** orders index - type; date
-  ***session_start_date:*** orders index - type; date
-  ***discount_amount:*** orders index - type; date
-  ***payment_amount:*** orders index - type; date
-  ***actions.has_sessions:*** orders index - type; date
-  ***actions.has_sessions:*** orders index - type; date
-  ***actions.has_sessions:*** orders index - type; date
-  ***actions.order_screen:*** orders index - type; date
-  ***actions.purchased:*** orders index - type; date
-  ***download_date:*** orders index - type; date
-  ***signup_date:*** orders index - type; date
-  ***report_date:*** orders index - type; date
-  ***report_types.to:*** orders index - type; date
-  ***report_types.from:*** orders index - type; date
-  ***frequency_segments:*** orders index - type; date
-  ***recency_segments:*** orders index - type; date
-  ***monetary_segments:*** orders index - type; date


At ElasticSearch, Session and Products Data are stored in the same Index which is ***orders index***. In orders index, There is a ***basket object*** that refers to the products of the related order ID. This data set is transferred from the Products Data Source. There is an ***actions object*** that refers to actions columns of the given session. When Sessions Data Source is created, actions are able to be included.


![image](https://user-images.githubusercontent.com/26736844/128704295-b53ea2d9-338b-4c9c-ab35-7342d4fd0d54.png)


Another index is stored with customers of unique information that is called ***downloads index***. This index is shaped with unique customers of dates from 1st interaction with the business to further interactions. Each document at the downloads index represents the unique customer. The first interaction of each customer might be the downloaded date or signup date. If there isn`t any downloaded date, it must be a date that indicates the first interaction date before the first sessions of each customer.


![image](https://user-images.githubusercontent.com/26736844/128704378-322026d3-633e-4b70-8b68-57f081d87943.png)


After the platform is connected to the ElasticSearch successfully, the ***Data Sources Connection*** process is eligible to connect to any data access.


![image](https://user-images.githubusercontent.com/26736844/128704493-68a7462d-24aa-4cd8-8de0-3fb1c6f47d5d.png)


***Sessions Connection*** is the main responsibility of the data processed.  See the example of the Session data set below in the ***Example of Session Data Source Columns Matching Process***. There are 2 parts to the connection process. First, Session Data Source of columns and ElasticSearch orders Index of fields must be matched. The main concept of matching is assigning the columns name to the orders index fields on Data Source configuration page. There are required columns which are Order ID, client, start date, amount, purchase. There are also optional columns which are date, discount, promotion, dimension, and actions. Second, Data Source connection type and other information such as user, pw, host, port DB. After connecting the data source, it is able to be changed when it is needed. See the Sessions Connection process below.


***Order ID***, represents unique ID for each session that is the purchased/non-purchased with string type.

***Client***, represents the customer ID who creates the purchased/non-purchased session with the string type. Client values must be stored at Customers Data Source and be transferred to the downloads index.

***Start Date***, is the date when the session is started with date type and yyyy-mm-dd hh:mm format.

***Amount***, is the value of the purchased sessions with float type. If the session is non-purchased, it is directly assigned as 0.

***Discount***, is the value of discount at the purchased sessions with float type. If the session is non-purchased, it is directly assigned as 0.

***Purchase***, represents whether the session is ended with the purchased or non-purchased. It is represented with the boolean format.

***Date***, represents the end date of the purchased/non-purchased session with date type and yyyy-mm-dd hh:mm format.

***Promotion***, represents the promoted the purchased sessions with string type. It is optional. When it is assigned A/B Test for Promotion Analysis is enabled. Each order might have promotion ID which is an indicator of the organic/inorganic order. It would be more efficient if both promotion ID and discount amount are assigned to the same data set.


***Dimension***, helps for grouping data sets when it is needed. While you need to split the data set you need to assign a dimensional column to the session data set. This process will be directly added to the ElasticSearch orders (sessions) index as a field and you may choose dimensions as filters in each dashboard in the filter sections.

***Actions***, represents key points at each session that are good to be added to the data set. It could be add-to-basket, purchase-screen, remove-product-from-basket, etc. This might help us to see how customers engage with the business and which process might give them an idea to leave the session or end it with the purchase.


![image](https://user-images.githubusercontent.com/26736844/128704812-962e7115-0024-4153-85fa-25f18558be98.png)


#### Example of Session Data Source Columns Matching Process


![image](https://user-images.githubusercontent.com/26736844/128704900-086ce8e1-0e8f-443d-8afd-9b406f3f2527.png)


***Customer Connection*** is a data source related to customers informations such as download date, signup date, etc. Each customer (client field at ElasticSearch) at the Session source must be included at Customer Connections. See the example of Customer data set below in ***Example of 
Customer Data Source Columns Matching Process***. There are 2 parts of connection process. First, Customers Data Source of columns and ElasticSearch downloads Index of fields must be matched. Main concept of matching is the assigning the columns name to the downloads index fields in Data Source configuration page. There are required columns which are client, download date. Actions and signup date are optional columns. Second, Data Source connection type and other informations such as user, pw, host, port DB. After connecting the data source, it is able to be changed when it is needed. See the Customer Connection process below.

***Client***,, represents unique identification for each customer with string type.

***Download Date***,, represents the date which is the first interaction date to the business with date type and yyyy-mm-dd hh:mm format.

***Actions***, represents key points at each session that are good to be added to the data set. It could be payment-amount-add, first-session, etc. This might help us to see how customers engage with the business and which process might give them an idea to leave the session or end it with the purchase. Unlike sessions of Actions, Customers Actions must be with date type and yyyy-mm-dd hh:mm format.


![image](https://user-images.githubusercontent.com/26736844/128705089-fc19a816-c61a-4367-8fc2-4b1f6917790b.png)


##### Example of Customers Data Source Columns Matching Process

![image](https://user-images.githubusercontent.com/26736844/128705724-4ab2f259-e13a-4d40-9d23-a1df57d87e14.png)


***Products Connection*** is a data source related to the basket of the sessions with products and it covers information such as product ID/name, price, product of the category. Customers and Sessions Data Sources are required, however, Product Connection is optional. A/B Test Products are related to products data sources. Products data source is transferring to orders index into the baskets object on each sessions document. See the example of Product data set below in ***Example of Products Data Source Columns Matching Process***. There are 2 parts to the connection process. First, Products Data Source of columns and ElasticSearch orders Index of fields must be matched. There are required columns which are order ID, product ID/name, category, price.

***Order ID***, indicates an ID for each session (purchased/non-purchased sessions). Please assign a column name that refers to Order ID at your data source. You need to write the correct column name in the box. It is crucial that all order IDs must be matched with order IDs at the session data source. Otherwise missing order IDs wouldn’t be matched and can not be transferred to the orders index.

***Product ID/Name***, represents ID or product name of related orders. This shows us the products at the basket in related order ID.

***Price***, indicates the price of each product. It is not the multiplication of price and amount. It is a unit of product price.

***Category***, indicates the category of the products.

![image](https://user-images.githubusercontent.com/26736844/128705956-ff991673-6368-40ad-bcea-1406a5b37ed7.png)

##### Example of Products Data Source Columns Matching Process

![image](https://user-images.githubusercontent.com/26736844/128706046-e4cfe210-90f1-45f9-bc09-8e59027bce17.png)

***Schedule Data Process*** is the main console to track the data transferring processes, Exploratory Analysis, and ML Works of scheduling and triggering processes. Once the schedule is created, data transfers from sessions/customers/products data sources to the ElasticSearch Indexes, and Exploratory Analysis and ML Words Processes will be started. If It is aimed at triggering these processes once, choose 'once' for the time period Otherwise process is triggered according to the time period. When you delete the schedule, the process will stop there and never updates the data sets.

There are 3 parts to Schedule Data Process;
-  ***Schedule*** Assign Tag Schedule Name. Choose ***Time Period***. Create Schedule Job.
-  ***Data Conections:*** This is the table where you can find Sessions/Customers/Products Connections information and Schedule Process of logs. This table of each cell is assigned while it is connected to the Sessions/Customers/Products.
-  ***Logs:*** In order to check which process is triggered, logs help us to find which process is running on.

![image](https://user-images.githubusercontent.com/26736844/128707466-c82289db-3525-4d22-8421-2f07f99eb56d.png)

---

### C. Exploratory Analysis

CustomersAnalytics provides us some crucial and popular Analysis with individual dashboards and charts. These are Funnels, Cohorts, Descriptive Stats, Product Analytics, Churn Analysis, Promotion Analytics.

#### 1. Funnels

They allow you to see where your potential customers fall within your desired goal flow and enable you to track and measure conversion rates for each step on that path. Funnel at CustomersAnalytics allows us to see conversion from 2 main parts. First One is the Session Funnels which covers from session start event to purchase event. If there is an action between session start and purchase event, these are also added to the Funnels as Action. These actions must be added to Session data source while it is created, Second one is the Customers Funnels which covers the customers of events and actions. This type of funnel covers all events from download date to session start date. If there are additional actions, they must be added as actions while Customers Data Source is created.


##### a. Sessions of Actions Funnel

Session start and purchase events are by default actions for this type of funnel. This funnel shows the values from session to purchase. At the ElasticSearch orders index, It is the ‘actions’ object that gives us the clue of the Sessions of Actions. Funnels count the True values of each metric at the ‘actions’ objects. If there are additional actions, they are also stored in ‘actions' object. Each action has been calculated related to time periods. This process is an aggregation of related order id according to time periods. There are 4 time periods; monthly, weekly, daily, hourly.


- ***i. Daily Funnel:*** Total order count per action (session and purchase actions included) per day
- ***ii. Weekly Funnel:*** Total order count per action (sess. and purch. actions included) per week
- ***iii. Monthly Funnel:*** Total order count per action (sess. and purch. actions included) per month
- ***iv. Hourly Funnel:*** Average total order count per action (sess. and purch. actions included) per hour

![image](https://user-images.githubusercontent.com/26736844/128707902-c3329f4a-48fb-4127-b676-7663b30c4c30.png)

##### b.   Customers of Actions Funnel

Customers Funnels are aggregated count of events per customer according to time periods. They are started to be counted from the user's first engagement events such as download or signup. Customers Funnels also covers first Sessions start and first purchase event, For instance, daily funnel for Feb. 12, 2021, There are 29.435K  customers who have downloaded, 20.49K customers who have their first sessions, 13.7K customers who have their signups, and 9K customers who have their purchase. Just like Sessions of Actions funnels, there are 4 types of Funnels related to time periods which are daily, weekly, monthly, and hourly.


- ***i. Daily Funnel:*** Total customer count per customer of action (download and purchase actions included) per day
- ***ii. Weekly Funnel:*** Total customer count per customer of  action (download and purchase actions included) per week
- ***iii. Monthly Funnel:*** Total customer count per customer of action (sess. and purch. actions included) per month
- ***iv. Hourly Funnel:*** Average total customer count per customer of action (sess. and purch. actions included) per hour

![image](https://user-images.githubusercontent.com/26736844/128708087-bc6d9c36-644b-4623-95d3-ea6beb0e5beb.png)

#### 2. Cohorts

Cohorts are aggregated counts of customers who are acting in the same way at a given time period. In our study, cohorts cover the number of customers who have the same transaction in the given time period and upcoming days/weeks. At each cohort y-axises represent the date of the first action, x-axises represent the next action after x days/weeks. For instance, let`s check the Daily Download to 1st Order Cohort. Y-axis is the y Downloaded Date and the x-axis is the x day after downloaded. if y; Feb 28, 2021, and x; 10 and z; 10 customers who had their 1st orders. From the customers, who had downloaded on Feb 28, 2021 (y), there are 10 customers (z) who had orders after 10 days (x). There are 2 types of time periods for cohorts which are daily and weekly. According to the time period, the y-axis changes. There is 4 kind of Cohorts which are Download to 1sr Order, From 1st to 2nd Order, From 2nd to 3rd Orders, From 3rd to 4th Orders. Here are the kind of Cohorts;

- ***Download to First Order Cohort:*** Download to First Order CohortIf there isn`t any downloaded date, you may assign any date which is related to customers of the first event with your business. This cohort of date column represents download date. If it is Weekly Cohort it will represent the Mondays of each week. Otherwise, it will represent days. Each Numeric column from 0 to 15 is the day difference after the downloaded date. For instance, if the date column is 2021-05-06, and the numeric column is 10 and the value is 100, this refers that there are 100 customers who have downloads in 2021-05-06 and have first orders 10 days later.

- ***First to Second Order Cohort:*** This cohort of date column represents the first order date. If it is Weekly Cohort it will represent the mondays of each week. Otherwise, it will represent days. Each Numeric column from 0 to 15 is the day difference after the first order date. For instance, if the date column is 2021-05-06, and the numeric column is 10 and the value is 100, this refers that there are 100 customers who have first orders in 2021-05-06 and have second orders 10 days later.

- ***Second to Third Order Cohort:*** This cohort of date column represents second order date. If it is Weekly Cohort it will represent the mondays of each week. Otherwise, it will represent days. Each Numeric column from 0 to 15 is the day difference after the second order date. For instance, if the date column is 2021-05-06, and the numeric column is 10 and the value is 100, this refers that there are 100 customers who have second orders in 2021-05-06 and have third orders 10 days later.

- ***Third to Fourth Order Cohort:*** This cohort of date column represents the third order date. If it is Weekly Cohort it will represent the mondays of each week. Otherwise, it will represent days. Each Numeric column from 0 to 15 is the day difference after the third order date. For instance, if the date column is 2021-05-06, and the numeric column is 10 and the value is 100, this refers that there are 100 customers who have third orders in 2021-05-06 and have fourth orders 10 days later.

##### a. Daily/Weekly Order Cohort

There are 2 types of time periods for cohorts which are daily and weekly. According to the time period, the y-axis changes.

- ***i. Daily Download to 1st Order Cohort***

    This cohort enables us to see the ***daily*** engagement of the customers and the time period between each customer of download and 1st order.


![image](https://user-images.githubusercontent.com/26736844/128708497-b41fac70-597f-4e31-9e17-5f04d8bff484.png)


- ***ii. Daily Cohort From 1st to 2nd Order***

    This cohort enables us to see the ***daily*** engagement of the customers and the time period between each customer of 1st order and 2nd order.
    
 ![image](https://user-images.githubusercontent.com/26736844/128710372-b4f7db7f-23ba-46ea-938e-925ed27a4cfa.png)   

- ***iii. Daily Cohort From 2nd to 3rd Order***

    This cohort enables us to see the ***daily*** engagement of the customers and the time period between each customer of 2nd order and 3rd order.
    
 ![image](https://user-images.githubusercontent.com/26736844/128710413-0d27867f-f6ef-4c74-9f29-01f1c9933ba7.png) 

- ***iv. Daily Cohort From 3rd to 4th Order***

    This cohort enables us to see the ***daily*** engagement of the customers and the time period between each customer of 3rd order and 4th order.
    
![image](https://user-images.githubusercontent.com/26736844/128710463-bdd99e5b-cfee-4aac-87ab-0f9d0e009060.png)  

- ***v. Weekly Download to 1st Order Cohort***

    This cohort enables us to see the ***weekly*** engagement of the customers and the time period between each customer of download and 1st order.
    
![image](https://user-images.githubusercontent.com/26736844/128710533-dbd41131-b3dc-4839-99b7-cb4d51055af9.png)
    
- ***vi. Weekly Cohort From 1st to 2nd Order***

    This cohort enables us to see the ***weekly*** engagement of the customers and the time period between each customer of 1st order and 2nd order.
    
![image](https://user-images.githubusercontent.com/26736844/128710576-0a6c7db7-b4bb-4ae7-ae31-e71cf1ced102.png) 
    
- ***vii. Weekly Cohort From 2nd to 3rd Order***

    This cohort enables us to see the ***weekly*** engagement of the customers and the time period between each customer of 2nd order and 3rd order.
    
![image](https://user-images.githubusercontent.com/26736844/128710640-81d64ba2-e3c9-475f-b984-123f3cfafaed.png)
    
- ***viii.  Weekly Cohort From 3rd to 4th Order***

    This cohort enables us to see the ***weekly*** engagement of the customers and the time period between each customer of 3rd order and 4th order.

![image](https://user-images.githubusercontent.com/26736844/128710690-c270a7d1-891f-48da-9951-aec2a4de1a1a.png)

#### 3. Descriptive Stats

Descriptive Stats are some overall values that need to check each day for businesses. There is two main dashboards for Descriptive Stats which are Stats Purchase and Descriptive Statistics.
I addition to that, customerAnalytics Overall and Customers Dashboards, search dashboards related to promotion, clients have some charts related to Descriptive Stats of Analysis.

##### a. Stats Purchase

The dashboard covers the total orders per day/week/hour/month. It enables us to see the pattern of the order counts such as rush hours week day/weekend, summer/winter time.

- ***i.  Daily Orders***

    The total number of orders per day. Total number of unique session IDs which has has_purchase (has_purchase = True) breakdown with day.
    
    ![image](https://user-images.githubusercontent.com/26736844/128714421-22924183-45bb-4de4-9b2f-9e12f23c7331.png)

- ***ii.  Hourly Orders***

    Average total order count per day. The total number of order counts per day and per hour is aggregated. The next step is calculating the average order count per hour by using aggregated data from the previous step.
    
    ![image](https://user-images.githubusercontent.com/26736844/128714465-afb1669f-cec1-474b-96b0-da136e5be5af.png)

- ***iii.  Weekly Orders***

    The total number of orders per week. Each week is represented by Mondays.

    ![image](https://user-images.githubusercontent.com/26736844/128714516-ab7bd8f7-42cf-43ac-a9bd-79789383e322.png)

- ***iv.  Monthly Orders***

    Total number of orders per month; the total number of purchase unique session IDs are counted per month.

    ![image](https://user-images.githubusercontent.com/26736844/128714549-9bfb9adb-d3e3-41c1-9327-4fba39d42969.png)
    

##### a. Descriptive Statistics

- ***i.  Weekly Average Session Count per Customer***

    Each customer's total unique session count per week is calculated. By using this aggregated date average session count per customer per week is calculated. This might give us the clue of which weeks the customers of engagement to the business are increased/decreased. In addition to that, It gives us information about the customers` session frequency per week.
    
    ![image](https://user-images.githubusercontent.com/26736844/128714694-a5577732-286b-445b-9d6e-de714e6e360f.png)


- ***ii.  Weekly Average Session Count per Customer***
    
    Each customer's total order count per week is calculated. By using this aggregated date average order count per customer per week is calculated. This might give us the clue of which weeks the customers of order possibilities are increased/decreased. In addition to that, It gives us information about the customers` purchase frequency per week.
    
    ![image](https://user-images.githubusercontent.com/26736844/128714752-baacb0d6-f3de-4ae8-a535-0a9d30776147.png)
    
- ***iii.  Payment Amount Distribution***

    This chart is a histogram which shows the bins of total customers. Each bin represents a range of payment amounts related to purchased orders.
    
    ![image](https://user-images.githubusercontent.com/26736844/128714930-e5d2da61-f4ea-4d4a-8f52-ec404be1b297.png)
    
- ***iv.  Weekly Average Payment Amount*** 
    
    Average payment amount of orders user who has the purchased orders.
    
    ![image](https://user-images.githubusercontent.com/26736844/128715075-91a9b5bd-61c1-4176-8816-e704204bea73.png)
    
#### 3. Product Analytics

When Product Data Sources is added and the data is transferred to the ElasticSearch, Product Analytics Charts are able to be updated. These give the basic idea of the products and their categories of purchase processes. These charts help us figure out the relation between products and the customers of order probability.
    
##### a. Top Purchased Products

- ***i.  Most Combined Products***
    
    The most preferred products for the customers. Each bar represents the total number of orders per product. According to the example below, There are 250 orders which have both p_153 and p_166 in their baskets.
    
    ![image](https://user-images.githubusercontent.com/26736844/128715478-5645d840-40cf-4ed3-b40f-8ed75d5d51a8.png)
    
- ***ii.  Most Purchased Products***

    The most preferred product categories for the customers. Each bar represents the total number of orders per product category.ü
    
    ![image](https://user-images.githubusercontent.com/26736844/128715636-bb9ab166-601b-4977-89db-da7254950902.png)

- ***iii.  Most Purchased Categories***

    Each product pairs have counted an order and the Total number of order count per product pairs are calculated.
    
    ![image](https://user-images.githubusercontent.com/26736844/128715731-4c84868b-a88b-4909-9120-f6a066e869bd.png)
    
    
---

### D. Machine Learning Works

Additional to Exploratory Analysis, ML Works enables us the more data investigation with deeper methods in the CustomerAnalytics. These methods are popular on Customer Analytics Data Scientist teams which are A/B Test, Customer Lifetime Value Prediction, Anomaly Detection, and Customer Segmentation. These complex methods will help the business in decision-making. 

At each ML Works part, There is an individual dashboard with charts. Customer Segmentation is presented with RFM values which are Recency, Monetary, Frequency. CLV Prediction predicts the next 6 months of daily total payment amounts and Segmented Customers of total payment amounts. A/B Test compares before and after the time periods with metrics which are order counts or average payment amounts for products and promotions. It gives results about how well the product improves the order count or average payment amount.

There are Machine Learning Algorithms that help us to calculate these ML processes. Customer Segmentation classifies the customers with K-Means, CLV Predictions creates results with LSTM & Conv NN, A/B Tests creates tests by finding the statistical distribution of the data sets (Poisson, Binomial, Gaussian, Gamma, Beta) and applies Hypothesis in order to calculate results. Anomaly Detections finds the abnormal data points with Deep Learning AutoEncoder.

#### 1. A/B Tests

At Customers Analytics, A/B Test compares the before - after the metrics. There are 3 types of A/B Tests which are Products Comparisons, Promotions Comparisons, And Customer Segments Comparisons. These comparisons are measured by 2 metrics individually which are Purchase Order Count and Average Payment Amount.

Products and Promotion Comparisons are applied similar way. They are applied for each product/promotion of usage before and after time periods. Each customer of order with product/promotion is assigned as 0 points. Before 0 points of orders are compared with after 0 points of Orders. Assigning as 0 point process is applied for the customers of each product/promotion usage order.

Unlike Products and Promotion Comparisons at Segment Comparison, Before and After time periods are constants time periods which are day, week, and month. Each segment of the current day, current week, and current month are compared with the last 4 weeks of the same week day of the current week day, the last 4 weeks, and the last 4 months.

##### a. A/B Test Promotion 

A/B Test Promotions are a bunch of tests that compare the effect of Promotions on the business. There are two metrics that help us to measure comparison values. These are the Number of orders counts per promotion and the Average purchase amount per promotion. Promotion of Usage related to these metrics are tested with Before-After Analysis in order to answer the question is "Is there any significant increase/decrease on Order Count / Average Payment Amount per User after the Promotion is used?" While we need to see to Promotion which increases/decreases the number of orders and the average purchase amount, This Section can help us to find the right promotion.

-  Hypothesis are;
    -   ***H0 :*** There is no significant Difference between A - B of order count/purchase amount.
    -   ***H1 :*** There is a significant Difference between A - B of order count/purchase amount.
    
-   Bootstrapping Method;

    Each test process is applied with Bootstrapping Method. So, Iteratively randomly selected A and B sample customers of order count / average purchase amount will be tested acceptable enough time. Each iteration A - B samples are randomly selected. The accepted ratio will show us the confidence of our test results. For instance, there are 100 tests and 75 of them are ended with H0 Accepted. So, tests are % 75 our tests are H0 Accepted.
    
-   How are Promotions Compared?

    Each combination of promotions is tested individually. The average of Purchase Payment Amount per User Sample for Promotion (1) is compared to the Average of Purchase Payment Amount per User Sample for Promotion (2).
    
-   How are Before - After Promo Usage Tests designed?

    It might be very confusing and it might seem a very sophisticated A/B Test process, however, it basically compares promotions related to customers of usage. Each promotion of usage is tested separately. Each promotion of used users and their used timestamp are collected for each promotion. Each customer of one week before promotion usage and one week after the promotion use time periods are compared with to metrics which are order count and average purchase amount. We aim to detect a significant increase after the time period while comparing to before the time period about purchase amount or order count.
    
![image](https://user-images.githubusercontent.com/26736844/128716596-31991c37-bf79-4494-8214-7f35d33f9f63.png)

-   ***i. Promotion Comparison***

    Y-axis represents the total number of Test Accepted Count. The X-axis represents Test Reject Ratio. Each dot on the scatter plot represents each promotion. This is a comparison of each promotion pair. Each promotion pair is tested and Each dot represents 1st promotion of each promotion pair. More valuable Promotions are the dots that have less Reject Ratio and more Accept Count.

    At Promotion Comparison, each promotion of usage per customer is collected. These customers of orders before and after the used promotion of orders are compared. The average payment Amount per customer is compared. For instance, there are 1K customers who used promotion p_1. From the 1K population, the average payment amount per customer is randomly sampled 100 times and tested. In the end, the number of Null Hypotheses of Reject and Accept test count is calculated. If there is a significant increase in the average payment amount after the p_1 usage, the promotion would be selected as Perfect Promotion.

    At this instance below, there is 40 test which says this promotion did not affect the payment amount after the usage of promotion. In addition to that 36 % of the tests are rejected which means that they have resulted in a significant increase in the payment amount.
    
    ![image](https://user-images.githubusercontent.com/26736844/128716761-8e05f18e-548a-4c40-857b-86facbe2180b.png)
    
-   ***ii. Order and Payment Amount Difference for Before and after Promotion Usage***
    
    Y-axis represents the difference in Average Payment Amount per customer. The X-axis represents the difference in the Average Purchase Count per customer. Differences are calculated per user and per promotion. Differences are calculated between customers of purchase before and after using the promotion. Each dots represents the promotion with the difference in Average Payment.
    
    ![image](https://user-images.githubusercontent.com/26736844/128716995-c2838ba5-babc-4242-b1fd-e8c97786ec17.png)
    
    
-   ***iii. Before - After time Periods customers’ Average Purchase Payment Amount Test (Test Accepted!)***
    
    Average Payment Amount per customer is calculated according to before and after the usage of the promotions. The assumption is “There is a significant payment amount INCREASE after the promotion of usage”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128718557-14028494-8353-4fec-8133-ce1c236c5f9c.png)
  
-   ***iv. Before - After time Periods customers’ Average Purchase Payment Amount Test (Test Rejected!)***

    Average Payment Amount per customer is calculated according to before and after the usage of the promotions. The assumption is “There is a significant payment amount DECREASE after the promotion of usage”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128718720-1a2521f6-6ee0-4202-9a0c-7868b5ee03ed.png)
    
-   ***v. Before - After time Periods customers’ Total Purchase Count Test (Test Accepted!)***
    
    The total purchase count per customer is calculated according to before and after the usage of the promotions. The assumption is “There is a significant purchase count INCREASE after the promotion of usage”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128718899-3a41912e-d38d-4414-80e8-61e814ef1f8c.png)
    
-   ***vi. Before - After time Periods customers’ Total Purchase Count Test (Test Rejected!)***

    The total purchase count per customer is calculated according to before and after the usage of the promotions. The assumption is “There is a significant purchase count DECREASE after the promotion of usage”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128719008-b97a9580-4372-47b0-bbe6-944cb3cabb30.png)
    
##### b. A/B Test Product

A/B Test Products are a bunch of tests that compare the effect of Products on the business. There are two metrics that help us to measure comparison values. These are the Number of orders counts per product and the Average purchase amount per product. Products of Purchases related to these metrics are tested with Before-After Analysis in order to answer the question that is "Is there any significant increase/decrease on Order Count / Average Payment Amount per User after the Products is purchased?" While we need to see to the product which increases/decreases the number of orders and the average purchase amount, This Section can help us to find the right product for the business.

- Hypothesis are;
    
    -   ***H0 :*** There is no significant difference between A - B of order count/purchase amount.
    -   ***H1 :*** There is a significant difference between A - B of order count/purchase amount.
    
- Bootstrapping Method;
    
    Each test process is applied with Bootstrapping Method. So, Iteratively randomly selected A and B sample customers of order count / average purchase amount will be tested acceptable enough time. Each iteration A - B samples are randomly selected. The accepted ratio will show us the confidence of our test results. For instance, there are 100 tests and 75 of them are ended with H0 Accepted. So, tests are % 75 our tests are H0 Accepted.

- How are Before - After Product Usage Tests designed?

    It might be very confusing and it might seem a very sophisticated A/B Test process, however, it basically compares products related to customers of product selection at the basket. Each product selection at the basket is tested separately. Each timestamp of the selected product at the basket is collected for each product per customer individually. Each customer of one week before product selection and one week after the product selection time periods are compared with to metrics which are order count and average purchase amount. We aim to detect a significant increase after the time period while comparing to before the time period about purchase amount or order count.

-   ***i. Before - After time Periods customers’ Average Purchase Payment Amount Test (Test Accepted!)***
    
    The average Payment Amount per customer is calculated according to before and after the purchase of products. The assumption is “There is a significant payment amount INCREASE after the purchase of products”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128719717-271a471b-b0ac-45b2-b47f-f018efbaa5c6.png)
    
-   ***ii. Before - After time Periods customers’ Average Purchase Payment Amount Test (Test Rejected!)***

    The average Payment Amount per customer is calculated according to before and after the purchase of products. The assumption is “There is a significant payment amount DECREASE after the purchase of products”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.

    ![image](https://user-images.githubusercontent.com/26736844/128719739-b140f7d1-37d1-49be-94ce-83792579015f.png)
    
-   ***iii. Before - After time Periods customers’ Total Purchase Count Test (Test Accepted!)***   
    
    The total purchase count per customer is calculated according to before and after the purchase of products. The assumption is “There is a significant purchase count INCREASE after the purchase of products”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128720113-ec8b4d6b-8301-44cb-b8e7-0dd82e1c8684.png)
    
-   ***iv. Before - After time Periods customers’ Total Purchase Count Test (Test Rejected!)*** 
    
    The total purchase count per customer is calculated according to before and after the purchase of products. The assumption is “There is a significant purchase count DECREASE after the purchase of products”. So, the accepted Null Hypothesis according to this assumption is shown in the chart below.
    
    ![image](https://user-images.githubusercontent.com/26736844/128720179-81cdd2c8-e573-454e-9b4b-4bc06c5c30c9.png)
    
##### c. A/B Test Customer Segment Change

A/B Test Customer Segment Change is a bunch of tests that compare the effect of Customers Segments of Change within Time Periods on the business. Before and After time periods are constants time periods which are day, week and month. Each segment of the current day, current week and current month are compared with the last 4 weeks of the same weekday of current weekday, last 4 weeks and last 4 months. There are two metrics that help us to measure comparison values. These are the number of orders counts per segment and the Average purchase amount per segment. Difference of Segments between before and after time periods related to these metrics are tested with Before-After Analysis. in order to answer the question is "Is there any significant increase/decrease on Order Count / Average Payment Amount per Customer Segment" While we need to see to Segment which increases/decreases the number of orders and the average purchase amount, This section can help us to find the right Segment of Improvements.

The main concept of the Customer Segment Change Analysis is for detecting the significant increase or decrease at segments. This might help us to see how growth works for each segment.

-   ***i. Daily Customer Total Order Count per Customer Segment*** 

    The current day of total order count is compared with the last 4 weeks of the weekday which is the same weekday as the current-day weekday. For instance, If today is Friday, the last 4 Fridays are compared with today. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 weeks) are not included. 
        
    ![image](https://user-images.githubusercontent.com/26736844/128720389-0a7c5d88-ab78-4336-9c1f-89e0f60e2868.png)

-   ***ii. Weekly Customer Total Order Count per Customer Segment*** 

    The current week of total order count is compared with the last 4 weeks. For instance, If today is at week 40, the last 4 weeks are weeks 36, 37, 38, 39 is compared with week 40. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 weeks) are not included.
    
    ![image](https://user-images.githubusercontent.com/26736844/128720401-4545d1fb-3b46-44f7-aa89-32cca3f9e9a3.png)


-   ***iii. Monthly Customer Total Order Count per Customer Segment*** 

    The current month's total order count is compared with the last 4 months. For instance, If today is in July, the last 4 months are March, April, May, June is compared with July. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 months) are not included.
    
    ![image](https://user-images.githubusercontent.com/26736844/128720410-c62e85d0-9195-438f-9ba4-a081f223e09f.png)
    

-   ***iv. Daily Customers’ Average Payment Amount per Customer Segment*** 

    The current day of total order count is compared with the last 4 weeks. For instance, If today is on Friday, the last 4 weeks of Fridays are compared with the current day. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 weeks) are not included.

    ![image](https://user-images.githubusercontent.com/26736844/128720415-1b2cc863-3924-4bcf-9e59-1fbd90dc68ff.png)

    

-   ***v. Weekly Customers’ Average Payment Amount per Customer Segment*** 

    The current week of total order count is compared with the last 4 weeks. For instance, If today is in week 40, the last 4 weeks (week 39, 38, 37, 36) are compared with the current week. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 weeks) are not included.

    ![image](https://user-images.githubusercontent.com/26736844/128720431-9601ab57-adce-4be2-9183-c1c3b8f27be1.png)



-   ***vi. Monthly Customers’ Average Payment Amount per Customer Segment*** 

    The current month of total order count is compared with the last 4 months. For instance, If today is in July, the last 4 months (March, April, May, June) are compared with the current month. Each segment of customers of orders is collected separately. If there are customers whose segments changed during the time period (last 4 months) are not included.
    
    ![image](https://user-images.githubusercontent.com/26736844/128720452-310adb5c-b11c-4501-9d2d-66f36eaaa2d1.png)
    
#### 2. Customer Segmentation

Customer Segmentation is one of the popular arguments while analyzing the CustomerAnalytics. It uses one of the popular Segmentation perspective which is RFM (recency - monetary - frequency). 

RFM is the most common Marketing measurement for each business. It is easy to apply for any kind of business. It is the combination of 3 metrics which are Recency - Monetary - Frequency.

The customers, who are engaged with the business, are counted as millions. However, Each individual customers have their unique behavior with the business, mostly they can be clustered in the same way according to their engagement and attention to your business. In that case, Customer Segmentation allows us to see this similarity of customers by separating the customers' population to the individual homogeny sub-groups.

-   Recency

    It is a time difference measurement related to how recently the customer engages with the business.
    
-   Monetary

    It is the value of the purchases per customer.
    
-   Frequency

    It is a time difference measurement related to how average hourly difference between 2 orders per customer in the business.
    
    
##### a. RFM

RFM is the most common Marketing measurement for each business. It is easy to apply for any kind of business. It is the combination of 3 metrics which are Recency - Monetary - Frequency.

-   ***i. RFM 3D Scatter***

    Each point represents individual customers’ RFM values The X-axis represents recency; Y-axis represents monetary, Z-Axis represents frequency values.

    ![image](https://user-images.githubusercontent.com/26736844/128722705-ca0d67bc-19e3-4bc5-90ef-46197f262c53.png)

-   ***ii. Frequency - Recency***

    The X-axis represents frequency; Y-axis represents recency values. Colors represent Customer Segments.
    
    ![image](https://user-images.githubusercontent.com/26736844/128722992-0ce7ff32-6f71-4afe-ba9c-bbc1908fb91d.png)
    
-   ***iii. Monetary - Frequency***   

    The X-axis represents monetary; Y-axis represents frequency values. Colors represent Customer Segments.
    
    ![image](https://user-images.githubusercontent.com/26736844/128723257-33ab3237-5207-48d5-b243-f7a427a54ddd.png)
    
-   ***iv. Recency - Monetary***     
    
    The X-axis represents recency; Y-axis represents monetary values. Colors represent Customer Segments.

    ![image](https://user-images.githubusercontent.com/26736844/128723972-4bb8ec90-753e-4498-9961-ad4e03897d2e.png)
    

##### b. Segmentation

The customers, who are engaged with the business, are counted as millions. However, Each individual customers have their unique behavior with the business, mostly they can be clustered in the same way according to their engagement and attention to your business. In that case, Customer Segmentation allows us to see this similarity of customers by separating the customers' population to the individual homogeny sub-groups.

-   ***Segments***

    By using RFM values it is possible to segment each customer. We aim to find how they engage with the business which part of them can be encouraged to engage the business, who are gone? Who are new to the business? Here are the segments;
    -   ***Champions :*** They love your business and are pleased with your services or products. Probably, They won`t let you down very soon. You are the rock start for them.
Loyal Customer: Just like champions, they do love your business and not thinking to leave you soon. They are pleased about your service and products of quality. Just They do love your rock band and never miss your concert, however, they would like to watch it at the back.
    -   ***Potential Loyalist:*** This group of customers would like to engage with your business recently, frequently, but they are still not sure that they would like to totally engage. They probably need a push.
    -   ***Can`t lose them:*** Just like the ‘potential loyalist’ segment of customers, these are engaged customers, but their average basket value is a bit higher when we compare with loyal customers.
    -   ***At risk:*** They probably leave and become churn customers. Make sure to keep them engaged in the business.
    -   ***Need Attention :*** Just like ‘at risk’ segment of customers, these are going to become churn customers. Like the ‘at risk’ segment of customers, you have more chance to make them engaged again.
    -   ***Lost :*** They are churn customers or very close to become the churn customers.
    -   ***Newcommers :*** They have just arrived at your business. And we might not know about their satisfaction because of the lack of information about them. Track their behaviors and see how they become churn or champions in the future.

-   How does Customer Segmentation work?

    The calculated RFM values are segmented individually with 5 segments (K=5). New process will be the assigning of segment name according the top segments for RFM values. For instance, A customer who has recency; 0.2 hr (average recency; 3 days), monetary; 70 (average monetary is 30), frequency; 1 day (average frequency is 4 days). This customer's frequency segment will be 1, recency segment will be 1, monetary segment will be 1. This customer will be segmented as ‘Champions’.
    
-   ***i. Customer Segmentation TreeMap*** 
    
    This a treemap with % of total unique customer count per segment.
    
    ![image](https://user-images.githubusercontent.com/26736844/128728978-ca72e799-188f-4191-b789-3a380807edf6.png)
    
-   ***ii. Frequency Segmentation*** 

    Each customer's average time difference is calculated (Frequency Calculation). Frequency values are clustered with K-Means clustering into the 5 clusters (K=5). According to each frequency value number of customers is counted per frequency segment. The X-axis represents average frequency values. Y-Axis represents the total number of client count per average frequency per frequency segment.
    
    ![image](https://user-images.githubusercontent.com/26736844/128729432-47dd0e7d-2fee-4d87-b5fe-ed1f0f995124.png)

-   ***iii. Monetary Segmentation*** 

    Each customer's average purchased amount is calculated (Monetary calculation). Monetary values are clustered with K-Means clustering into the 5 clusters (K=5). According to each monetary value number of customers is counted per monetary segment. The X-axis represents average monetary values. Y-Axis represents the total number of client count per average monetary per monetary segment.
    
    ![image](https://user-images.githubusercontent.com/26736844/128729272-fd7348b6-d5d1-4fd9-b3d1-73766b94cd35.png)

-   ***iv. Recency Segmentation*** 
    
    Each customer's average time difference between the current date to last purchased date is calculated (recency calculation). Recency values are clustered with K-Means clustering into the 5 clusters (K=5). According to each recency value number of customers is counted per recency segment. The X-axis represents average recency values. Y-Axis represents the total number of client count per average recency per recency segment.
    
    ![image](https://user-images.githubusercontent.com/26736844/128729338-70b4b83d-cf05-4097-92c4-c98ec620a945.png)
    
    
#### 3. CLV Prediction

CLV Prediction enables us to see how the customers engage with the business, what this engagement brings to the businesses. In order to predict the future of customer engagement, each of them must be well understood. At CustomerAnalytics, it is applied for next 6 months. While it is scheduling, the predictions for the future 6 months is applied once in two weeks. 

-   ****How does CLV Prediction work?***
    
    Each customer's next purchase date is predicted for the next 6 months. According to their purchase amount per their order predicted purchase dates of amounts are also predicted. However, we are only involved by using this mythology, newcomers of the total purchase amount are also predicted individually. CLV Prediction processes are not executed per day. It is executed per week. Each week the clv predictions are calculated for the next 6 months. You probably see overlapping days on the line chart. This is because of the weekly prediction process. In addition to that, This overlapping also helps us to track the CLV Performance about comparing Actual and Predicted Total Payment Amount per day.
    
-   ****i. Next Week CLV Prediction***

    In the timeline, you probably see the daily total amount and predicted purchase amount per day. In that case, it is possible to see how well your business works. There are two main categorical data which are 'actual' and 'predicted'. Actual data is the total purchase amount per day. Predicted Data is the total predicted amount per day.
    
    ![image](https://user-images.githubusercontent.com/26736844/128742864-60140065-81cc-4d94-962e-8a179d12ab0c.png)
    
-   ****ii. Customers of Segments of Total Predicted Purchase Amounts***

    This chart shows us the Predicted Customers of Segments. Their segments of % give us the idea of the business of growth.
    
    ![image](https://user-images.githubusercontent.com/26736844/128743015-81a9deae-552b-45b5-a656-201c3260b69a.png)
    
#### 4. Anomaly Detection

In order to catch the significant increase/decrease, even we think that the business doing ok/not well, anomaly detection allows us the alarming days or weeks or time periods. This process also allows us to see where the business did well and where can be improved.

- ***How does Anomaly Detection work?***

    Anomaly Detection mainly concerns with the significant increase/decrease of a data point in the given metric. These metrics are Daily Funnel, Daily Cohort Anomaly, Daily Orders Anomaly, CLV Prediction of RFM Vs Current RFM Anomaly. Each metric of values of abnormal measurements is detected by AutoEncoder. AutoEncoder generates scores for each metric of the data point. Residuals (the difference between the actual value and predicted value) are calculated. The outliers of the residuals are the Abnormal values.
 
- ***i. Daily Funnel Anomaly***
  
    Daily Funnel is the actions of totals per day. With the combination of all actions (from session count to purchase count per day), it is possible to detect abnormal days by using ML technique

    ![image](https://user-images.githubusercontent.com/26736844/128743285-af511f1a-14f9-4c15-9d25-068224c19860.png)
    
- ***ii. Daily Cohort Anomaly***
    
    Date columns represent downloaded day and each purchase count column represents the day that the customers had the first order after they have download. If there is an abnormal date which has a significant low/high first purchase count related to the downloaded date, this chart allows us the see the exact downloaded date as the abnormal date.
    
    ![image](https://user-images.githubusercontent.com/26736844/128743450-01b542a6-d99f-4101-bdaa-e4c7bb2319f2.png)
    
- ***iii. Daily Cohort Anomaly With Scores (Download to First Order)***

    Date columns represent downloaded day and each purchase count column represents the day that the customers had the first order after they have download. If there is an abnormal date that has a significant low/high first purchase count related to the downloaded date, this chart allows us the see the exact downloaded date as the abnormal date.

    ![image](https://user-images.githubusercontent.com/26736844/128743539-93af915c-f7be-42e9-86fc-050a241ad1ad.png)

- ***iv. Daily Order Anomaly***

    This chart allows us to see the % of increase/decrease compared with previous days of purchase counts.
    
    ![image](https://user-images.githubusercontent.com/26736844/128743703-bb993946-fbcf-4c77-8d7b-9eb3b7ae4b63.png)
    
- ***v. CLV RFM Vs Current RFM***
    
    There are engaged customers whose purchases are predicted with CLV Prediction. We also know their Recency - Monetary - Frequency values that are calculated with their historic purchases. If we calculate their future RFM values via the CLV prediction and subtract them in order to detect a significant increase/decrease for each metric, we might clearly see how our customers ' behaviors might change in the future.
    
    ![image](https://user-images.githubusercontent.com/26736844/128743808-e0f61a58-13a6-495d-9863-1c8661d74e97.png)
    
---

### F. Configugraitions

CustomerAnalytics allows to change profile pictures and chat according to individual charts in the system. For instance, you would like to create a comment about anomaly increase on Feb 20, 2021, on the Daily Orders Chart. It is possible with creating the message.


#### 1. Profiles

At profiles, you can create messages or you can change your profile picture.

![image](https://user-images.githubusercontent.com/26736844/128744089-df21bcee-8d52-4117-bf73-93815929bbb5.png)

#### G. CustomerAnalytics DashBoard

It is a dashboard with a combination of dashboards related to Exploratory analysis and Machine Learning Works. When it is logged to the CustomersAnalytics, first you are directed to the Overall Dashboard includes Orders - Revenue - Visitors - Discount KPIS, Daily Orders, Customer Journey, Churn Rate, Churn Rate Weekly, Top 10 Purchased Products, Top 10 Purchased Categories. At the other tab, you can see the Customer Dashboard includes Payment Amount Distribution, Total Number Customer Breakdown with Purchased Order Count, RFM, Download to First Order Cohort, Daily Funnel.

![image](https://user-images.githubusercontent.com/26736844/128744178-8787c901-1907-4f00-88ff-18d2e320c996.png)

#### 1. Overall Dashboard

-   ***Orders - Revenue - Visitors - Discount***

    Orders; Number of purchase count Revenue; Total Purchase Amount Visitors; Total Unique Visitors Count Discount; Total Discount Amount (Optional)
    
![image](https://user-images.githubusercontent.com/26736844/128744254-06a2a49a-72b0-4a5b-9b89-c08f6914cc90.png)


-   ***Daily Orders***
    
    Total Number of Success Purchase Transaction per day.
    
    ![image](https://user-images.githubusercontent.com/26736844/128744944-ca12fee0-0537-4e51-a2b5-1e635d0dfd11.png)


-   ***Customer Journey***

    Customers Journey Calculation; 1. Calculate average Hour difference from Download to 1st orders. 2 . Calculate average order count 3. For each calculated average order count, calculate the average purchase amount, Example; average 2 orders, 1st orders avg 30.3£, 2nd orders avg 33.3£ Calculate average recent hours customers last order to a recent date.
    
    ![image](https://user-images.githubusercontent.com/26736844/128745079-9433a70d-53dc-46f4-9fe0-40e6f7707cdd.png)
    
-   ***Churn Rate***    
    
    First, the Frequency value is calculated for each user. Frequency is the time difference between each sequential order per customer. Each customer's last order date is not expected to be purchased before the average frequency day. In other words, Each customer is expected to order at most average frequency days before. A number of unique customers who have ordered between average frequency date before the current date and current date are engaged users Churn is engaged users divided by total unique ordered users.
    
    ![image](https://user-images.githubusercontent.com/26736844/128745140-95858dee-6a39-429e-81f1-25206dd15908.png)
    
-   ***Churn Rate Weekly***  

    It is calculated the same as the churn rate per week (each Monday of the week).
    
    ![image](https://user-images.githubusercontent.com/26736844/128745308-b6ed2d27-d279-425c-906c-c5263d490725.png)
    
-   ***Top 10 Purchased Products*** 
    
    The most preferred products for the customers. Each bar represents the total number of order per product (for more details check Product Analytics).
    
-   ***Top 10 Purchased Categories***   

    The most preferred product categories for the customers. Each bar represents the total number of order per product category (for more details check Product Analytics).

#### 1. Customers Dashboard

-   ***Payment Amount Distribution*** 

    The X-Axis is the purchase amount values with bins. Y-Axis is the number of Unique Customer count related to their average purchase amounts. This Distribution allows us the see how much customers are willing to pay for a purchase.
    
    ![image](https://user-images.githubusercontent.com/26736844/128745540-914b1a7c-5ca8-4b98-80bb-44fb2f6343b0.png)
    
-   ***Total Number Customer Breakdown with Purchased Order Count***
    
    X Axis is Order Counts staring with 1. Y Axis is number of Unique Customer count related to their order counts. This Distribution allows us the see overall potential of your business.
    
-   ***RFM***

    X Axis represents 'recency'. Y Axis represents 'monetary'. Z Axis represents 'frequency'. Recency metric is a time difference between customers of last purchase transaction date to recent date. Monetary metric is average purchase amount per customer. Frequency metric is average time difference per between purchase date of each order per customer. There is a colored dots. These are represents the segmented customers according to their RFM values.
    
    
![image](https://user-images.githubusercontent.com/26736844/128745680-700dc370-4040-4ec0-9cbf-5cd288fa2cac.png)

-   ***Download to First Order Cohort***

    If there isn`t any downloaded date, you may assign any date which is related to customers of first event with your business. This cohort of date column represent download date. If it is Weekly Cohort it will represent the mondays of each week. Otherwise it will represent days. Each Numeric column from 0 to 15 are the day differences after the downloaded date. For instance, if date columns is 2021-05-06, and numeric column is 10 and value is 100, this refers that there are 100 customers who have downloads in 2021-05-06 and have first orders 10 days later.
    
    ![image](https://user-images.githubusercontent.com/26736844/128749399-e7ad990a-2ef5-48fc-9c4a-f873026d1f3f.png)
    
-   ***Daily Funnel***   
    
    X Axis represents days that are stored in ElasticSearch. Y Axis represents number of transactions such as order count, session count, add product the basket transaction count per day. In order to show actions such as add product the basket transaction count except order count, session count, these actions must be added to actions label while creating Sessions data source (For more information pls. check Create Data Source - Sessions & Customers).
    
    ![image](https://user-images.githubusercontent.com/26736844/128749465-a21461f1-aa9b-4310-8ff2-555c2790e0e6.png)
    
-   ***CLV Prediction - Daily***   
    
    In a timeline, you probably see the daily total amount and predicted purchase amount per day. In that case, it is possible to see how well your business works. There are two main categorical data which are 'actual' and 'predicted'. Actual data is the total purchase amount per day. Predicted Data is the total predicted amount per day.
    
    ![image](https://user-images.githubusercontent.com/26736844/128749609-04cb015b-f849-4b1b-b2ca-b32ab01a3ebd.png)

#### H. Searching

There are 4 types of search;

-   Products
-   Clients
-   Promotions
-   Dimensions
    
Each type of search represents an individual dashboard with results. When the expected search value is typed on the search bar;

-   ***Results are checked individually for each type of search.***
    -   Create ngrams of search value.
    -   Create ngrams for each list of search types (product IDs list, client IDs list, etc).
    -   Calculate the similarity between ngrams of search value and ngrams for each list of each types.
    -   Remove score = 0.
    -   Find the top score and assign is as a detected search results.
    
-   ***Create a temporary .csv file at temporary_folder_path that is assigned by the user.***

    There are 3 charts of .csv files;
    -   chart_2_search.csv is positioned at right top.
    -   chart_3_search.csv is positioned at left bottom.
    -   chart_4_search.csv is positioned at right bottom.
    
    There are 4 KPIs with 1 .csv file;
    -   chart_1_search.csv is positioned at left top. These KPIs will be changed.

-   ***Each chart of data is created at temporary file.***

    (chart_1_search.csv, .. , chart_4_search.csv) will be removed after the dashboard is shown at the user interface.
    
-   ***Each search type has individual charts and KPIs.***

    So, each creation of charts and KPIs .csv files must be applied individually.
    
##### 1. Product

Each product ID or name can be searched from the search bar. Product search is able to check unique product IDs or names at product columns that are created at Products Data Source. Once the scheduling process has been done, It is possible to search for unique products. In order to search for the new product name or ID, it must be rescheduled.

![image](https://user-images.githubusercontent.com/26736844/128750165-fdb95ad3-b86b-4688-a953-fb63b9cfbbdf.png)


##### 2. Client

Each client or client ID can be searched from the search bar. Client search is able to check unique clients at the client column that is created at  Sessions Data Source and Customers Data Source. Once the scheduling process has been done, It is possible to search for unique clients. In order to search for the new client name or ID, it must be rescheduled.

![image](https://user-images.githubusercontent.com/26736844/128750314-e4b49609-02a3-4703-b760-da9bedae59fc.png)

##### 3. Promotion

Each promotion or promotion ID can be searched from the search bar. Promotion search is able to check unique promotions at the promotion column that is created at  Sessions Data Source and it is the optional column. Once the scheduling process has been done, It is possible to search for unique promotions. In order to search for the new promotion name or ID, it must be rescheduled.
    
![image](https://user-images.githubusercontent.com/26736844/128750395-0dc1cdd1-8774-4e67-a59d-aed73eabe9e2.png)
    
##### 4. Dimension

Each dimension value can be searched from the search bar. Dimension search is able to check unique dimensions at the dimension column that is created at  Sessions Data Source and it is the optional column. Once the scheduling process has been done, It is possible to search unique dimensions. In order to search for the new dimension, it must be rescheduled.

![image](https://user-images.githubusercontent.com/26736844/128750487-df620acc-23e4-4f62-af31-9097b70973a9.png)


#### I. Use CustomerAnalytics via Python

Creating ElasticSearch connection, temporary data folder, Sessions - Customers - Products Data Sources, Exploratory Analysis, Machine Learning Works and running/shutting down CustomerAnalytics user interface are able to be operated via Python code.
    
    
    import customeranalytics


##### 1. Creating ElasticSearch connection and Temporary Data Folder

It enables us to configure ElasticSearch with host and port. Another requirement which is a temporary path is for importing files such as CLV Prediction of model files and .csv format files with the build_in_reports folder.

    customeranalytics.create_ElasticSearch_connection(            
						       port="9200, 
                               host="localhost", 
                               temporary_path=“/*****”)    

##### 2. Sessions - Customers - Products Data Sources

This process is for the connecting data sources which are SESSIONS, CUSTOMERS PRODUCTS. Each data source has its own unique form to connect. This process checks the connection failure, before store the connection information to the SQLite DB. There are 3 main updating processes for the store process;

-   ***1. db connection :*** These process included data_source_name, data_source_type, DB_name, DB_user, DB_name, data_source_path/query, etc.
-   ***2. actions :*** This is for actions both for Sessions and Customer Data Source. Example of actions; "has_basket, order_screen". Actions are split with ',' and string format.
-   ***3. column names :*** data source columns must be matched with ElasticSearch fields for each data source (Session, Customers, products) individually. column names must be string format. Example of columns;
    -   ***1. sessions_fields :***
        -   ***order_id :*** unique ID for each session (purchased/non-purchased sessions.
        -   ***client :*** ID for client this column can connect with client ID (client_2) or customer_fields.
        -   ***session_start_date :*** eligible date format (yyyy-mm-dd hh:mm)
        -   ***date :*** eligible date format (yyyy-mm-dd hh:mm)
        -   ***payment_amount :*** value of purchase (float/int). If it not purchased, please assign None, Null ‘-'.
        -   ***discount_amount :*** discount of purchase (float/int). If it not purchased, please assign None, Null ‘-'.
        -   ***has_purchased :*** True/False. If it is True, session ends with purchased. If it is False, session ends with non-purchased.
        -   ***optional columns :*** date, discount_amount.

    -   ***2. Customer Fields :***
        -   ***client_2 :*** unique ID for client this column can connect with client ID (client) or session_fields.
        -   ***download_date :*** eligible date format (yyyy-mm-dd hh:mm). This date can be any date which customers first appear at the business.
        -   ***signup_date :*** eligible date format (yyyy-mm-dd hh:mm). First event of timestamp after the download_Date per customer.
        -   ***optimal columns :*** signup_date
    
    -   ***3. Product Fields :***
        -   ***order_id :*** Order ID for each session which has the created basket. This column is eligible to merge with Order Id column at session fields.
        -   ***product :*** product Id or name. Make sure it is easy to read from charts.
        -   ***price :*** price per each product.
        -   ***category :***  product of category.
    
        
Products and sessions data sets are stored into the orders Index at ElasticSearch. Customer data sets are stored in the customers Index at ElasticSearch.

-   ***Parameters***

    -   ***customers_connection :*** dictionary with data_source, data_query_path, host, port, password, user, db.
    -   ***sessions_connection :*** dictionary with data_source, data_query_path, host, port, password, user, db.
    -   ***products_connection :*** dictionary with data_source, data_query_path, host, port, password, user, db.
    -   ***sessions_fields :*** dictionary with order_id, client, session_start_date, date, payment_amount, discount_amount, has_purchased.
    -   ***customer_fields :*** client_2, download_date, signup_date.
    -   ***product_fields :*** order_id, product, price, category.
    -   ***actions_sessions :*** string with comma separated for sessions.
    -   ***actions_customers :*** string with comma separated for customers. 
    -   ***promotion_id :*** string column name for promotions.
    -   ***dimension_sessions :*** string column name for dimensions.
    
    
        customers_connection = {'data_source_type': "postgresql",     
                                'data_query_path': """                
                                SELECT *        
                                FROM customers  
                                """,                
                                'user': "c****",                      
                                'password': "1******",                
                                'port': "5432",                       
                                'host': "127.0.0.1",                  
                                'db': "c******"                       
                                }
        
        
        
        
        
        sessions_connection = {'data_source_type': "postgresql",      
                               'data_query_path': """                
                               SELECT *        
                               FROM sessions   
                               """,                
                               'user': "c****",                      
                               'password': "1******",                
                               'port': "5432",                       
                               'host': "127.0.0.1",                  
                               'db': "c******"                       
                               }
        
        
        products_connection = {'data_source_type': "postgresql",      
                               'data_query_path': """                
                               SELECT *        
                               FROM products   
                               """,                
                               'user': "c****",                      
                               'password': "1******",                
                               'port': "5432",                       
                               'host': "127.0.0.1",                  
                               'db': "c******"                       
                               }   
                               
                               
        sessions_fields = {'order_id': ‘order_id',                    
                   'client': ‘client',                        
                   ‘session_start_date':'session_start_date', 
                   'date': ‘end_date',                        
                   'payment_amount': ‘payment_amount',        
                   'discount_amount': ‘discount_amount',      
                   'has_purchased': ‘has_purchased’} 
                   
        
        customer_fields = {'client_2': 'client',                      
                           'download_date': 'download_d',             
                           'signup_date': ‘signup_d'} 
        
        product_fields = {'order_id': ‘order_id',                     
                          'product': 'product',                       
                          'price': ‘price',                           
                          'category': ‘category'} 
        
        actions_sessions = "has_basket, order_screen"
        actions_customers = “first_login_date“
        promotion_id = “promotion_id“       
        dimension_sessions = “dimension“     
        
        customeranalytics.create_connections(                          
            dimension_sessions=dimension_sessions,  # optional          
            customers_connection=customers_connection, # required        
            sessions_connection=sessions_connection,  # required         
            sessions_fields=sessions_fields,  # required                 
            customer_fields=customer_fields,  # required                 
            product_fields=product_fields,  # required if there is a.  product data source                                                  
            products_connection=products_connection,  # optional       
            actions_sessions=actions_sessions,  # optional             
            actions_customers=actions_customers,  # optional           
            promotion_id=promotion_id  # optional                          
            )       
            
##### 3. Run CustomerAnalytics user interface

    customeranalytics.create_user_interface() 
    http://127.0.0.1:5000
    

##### 4. Shut Down CustomerAnalytics user interface
    
    customeranalytics.kill_user_interface() 
   
##### 5. Run Schedule Process

    customeranalytics.create_schedule(time_period=‘once’) 
    
##### 6. Delete Schedule Process

    customeranalytics.delete_schedule()
    
##### 7. Collecting Reports
    
If any report needs as a pandas data-frame, this can help us collect the report with the given name. Name list of the reports are available at report_names.

    customeranalytics.collect_report('daily_clv', 
                                     date=None, 
                                     dimension=‘main')   

##### 8. Collecting all report names


    customeranalytics.report_names()   

    ['weekly_funnel',
     'daily_clv',
     'daily_funnel',
     'daily_funnel_downloads',
     'weekly_average_payment_amount',
     'product_usage_before_after_orders_reject',
     'chart_4_search',
     'recency_clusters',
     'weekly_cohort_from_2_to_3',
     'daily_dimension_values',
     'daily_cohort_from_3_to_4',
     'promotion_comparison',
     'promotion_usage_before_after_orders_reject',
     'most_ordered_products',
     'daily_organic_orders',
     'monetary_clusters',
     'frequency_recency',
     'segments_change_monthly_before_after_amount',
     'daily_promotion_revenue',
     'chart_1_search',
     'daily_cohort_from_2_to_3',
     'weekly_cohort_from_3_to_4',
     'product_usage_before_after_orders_accept',
     'client_feature_predicted',
     'frequency_clusters',
     'monthly_orders',
     'daily_products_of_sales',
     'hourly_funnel',
     'weekly_average_order_per_user',
     'daily_inorganic_ratio',
     'daily_cohort_downloads',
     'hourly_inorganic_ratio',
     'promotion_number_of_customer',
     'inorganic_orders_per_promotion_per_day',
     'promotion_usage_before_after_orders_accept',
     'chart_3_search',
     'recency_monetary',
     'segments_change_monthly_before_after_orders',
     'customer_journey',
     'purchase_amount_distribution',
     'dfunnel_anomaly',
     'dcohort_anomaly_2',
     'dimension_kpis',
     'segments_change_weekly_before_after_orders',
     'user_counts_per_order_seq',
     'daily_cohort_from_1_to_2',
     'hourly_organic_orders',
     'weekly_cohort_downloads',
     'hourly_funnel_downloads',
     'product_usage_before_after_amount_accept',
     'client_kpis',
     'dorders_anomaly',
     'churn',
     'monthly_funnel_downloads',
     'most_combined_products',
     'avg_order_count_per_promo_per_cust',
     'dcohort_anomaly',
     'weekly_cohort_from_1_to_2',
     'segments_change_daily_before_after_orders',
     'rfm',
     'promotion_usage_before_after_amount_accept',
     'monthly_funnel',
     'kpis',
     'churn_weekly',
     'clvsegments_amount',
     'order_and_payment_amount_differences',
     'hourly_orders',
     'clvrfm_anomaly',
     'weekly_funnel_downloads',
     'product_usage_before_after_amount_reject',
     'weekly_average_session_per_user',
     'chart_2_search',
     'daily_promotion_discount',
     'product_kpis',
     'segments_change_weekly_before_after_amount',
     'daily_orders',
     'weekly_orders',
     'segmentation',
     'segments_change_daily_before_after_amount',
     'most_ordered_categories',
     'promotion_usage_before_after_amount_reject',
     'monetary_frequency',
     'promotion_kpis']













    
    
    