# CustomerAnalytics
 
### **A.** Understanding the concept of CustomerAnalytics
### **B.** How to use CustomerAnalytics




### **A.** Understanding the concept of CustomerAnalytics

This platform helps the business related to e-commerce. It also helps the business which has users (B2C) and has their data related to engagement to the business. 
At this platform, There are parts which are Exploratory analysis and Machine Learning Works. These parts have sub analysis and works. These parts also have individual built-in dashboards in or to visualize the results of the outputs at them.  

CustomerAnalytics platform needs to be supported with data-source connections, data sources of columns matching to its data sources (Sessions, Customers, and Products), accessible ElasticSearch Service, and triggering data to the ElasticSearch by the users. The platform supports the data source connections which are ‘.csv’, ‘PostgreSQL’, ‘GCLoud Bigquery’, and ‘AWS redShift’. Sessions and Customers data sources are required, but the Products data source is optional (without products data source, you can need to see the real charts for A/B Test Product, Product analytics). In addition to that additional actions are optional (you probably see the additional actions in Funnels). If you need to check the data source with a dimension (region, location,  business type, etc), it is possible to assign dimensions columns at the session data source. You may filter each dashboard with the given dimensions after that. Promotions of each order can be added to the data sources (optional). After that, you may see A/B Test Promotions, promotion individual searching (check G. Searching) with real data sets.

At ElasticSearch, there are created 3 indexes. Orders Index (orders) is fetched data from Session and Products Data sources with columns. These Orders Index columns are id (session ID), client (client ID), session_start_date (timestamp), end_date (session end timestamp, optional), payment amount (float numeric amount of purchased orders), discount amount (discount of purchased orders, optional), promotion_id (optional), dimension (optional), has_purchased(if True, session is ended with purchase, if False, session is ended with no o-purchase). Order Index of Product data source columns is a basket which is an object with keys are also the products data source of columns which are order_id (Session ID with basket), price (the price of the product), category (category of the Product), product (product name of ID). Downloads Index (downloads) is also created with customer Data source in order to assign data for customers when they are first engaged to the business or when they downloaded or signup. 3rd Index is Reports Index (reports) which is stored with KPIs data sets, outputs, and results of Exploratory Analysis and Machine Learning Works. Each report is a document of the index with a separate date. report_name represents the main analysis, type represents sub-analysis and report_Date represents the created date of the report or analysis.

    








