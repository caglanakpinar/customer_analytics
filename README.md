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
    
#### 3. Customer Segmentation

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
    


    
    
    
    
    
    
    
    
    
    
    
    
    
















    
    
    