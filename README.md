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




