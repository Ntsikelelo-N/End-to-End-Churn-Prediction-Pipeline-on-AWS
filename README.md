# End-to-End Customer Churn Prediction Pipeline on AWS
## Project Overview
Customer churn is a major business challenge - retaining existing customers is significantly cheaper than acquiring new ones.
This project builds an automated, end-to-end data engineering and data science pipeline to predict customer churn using AWS Free Tier services, Python and other libraries.

The system:
 - Ingest and stores data in amazon S3
 - Performs automated ETL and data validation
 - Conducts exploratory data analysis (EDA)
 - Trains and evaluates machine learning models

 The data:
 The data is collected from [scottdangelo github](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). The dataset is Telco customer churn from IBM, which simulates a company's customer churn. For more on the dataset [click here](https://www.ibm.com/docs/en/cognos-analytics/11.1.x?topic=samples-telco-customer-churn). This data can be downloaded and stored locally or can use a python function to download it.

## Setting up AWS environment
### Set up an IAM user
Open the AWS management console from your root user and create a user, by navigating to **IAM**. Navigate to **Users**. Choose create user and name the user what you want. Attach administrator access policy to the user. After the user has been created, click on the name of your user, underneath the **summary** click on **Create access key**. Choose **Command Line Interface (CLI)**, click **next** at the bottom of the screen. Add description for the tag, and click **Create access key**. You can download the generated access key on the next page in csv format. Keep this information safe as it allows a user access to your AWS resources via the command line.

For extra security, it is recommended to set up MFA device, which adds an additional layer of safety to keep your account safe.


![Created IAM USER](screenshots/IAM_user.png)

### Download the AWS CLI
Navigate to [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install/update your AWS CLI. This is going to be helpful to access the AWS services via management console.

### Accessing the AWS resources programmatically
Open a terminal of your choice on your device. Run the command `aws configure`. This will prompt you to provide access key, copy and paste the `access key` and then the `secret access key` which is also in the downloaded csv file.
When prompted for the region, you can look for region names on the management console. In my case I used **us-east-1**

## Create S3 Buckets
Navigate to **S3** on the management console. Click **Create bucket**, Choose **General purpose** write out <bucket_name>, this name has to be univerally unique from any other bucket names in the AWS. Leave other setting the same on the page. 
![Bucket](screenshots/creating_bucket.png)

You can uncheck the **Block all public access** section if you want your bucket to be seen by others. You can enable bucket versioning if you require the bucket to keep track of changes made inside it.
![Bucket](screenshots/create_bucket_2.png)

Click **Create bucket** to create the bucket.

## Upload data to S3 bucket

Click on the bucket created click `upload` button to get to import data into your bucket![customer churn bucket](screenshots/upload_to_S3.png)

Click on `Add files`and select the customer_churn dataset or drag the dataset on to the indicated space ![data upload window](screenshots/data_upload_window.png) After uploading the dataset, click `Upload` at the bottom of the page.

## AWS GLUE Crawler Setup
### Setup IAM role for the crawler
On the `AWS management console`, navigate to `IAM`, Click on `Roles` on the left of the page, underneath the `Access Management` header. Click on `Create role` to the top right of the page. ![IAM page](screenshots/Role_in_IAM.png)

Click on `AWS service` under `Trusted entity type` heading, under the `Use case` select `Glue` for `service or use case`, then click `next` at the bottom right of the page. ![Step 1](screenshots/page_1_roles.png)

On step 2, click on the search bar and search for `AWSGlueServiceRole`, ensure that the type is `AWS managed`,the second service to select is `AmazonS3FullAccess`. Click `next` bottom right of the page. ![](screenshots/step_2_glue.png)

On step 3, name your role, in my case it is named `AWSGlueChurnRole` scroll down to select `create role` at the bottom of the page. ![Final step of glue role](screenshots/step_3_glue.png)

### Creating Glue Database
Search for `AWS Glue` by either using the shortcut `Alt + S` on windows or by navigating to the top left of the page, to the search field. Click on `AWS Glue` and click on `Databases` below the `Data Catalog` heading. Click `Add database` towards the top right of the page. ![](screenshots/glue_database.png)

Write down the name of your database. In my case it is named `churn_db` Then click `create database`.![Glue Database](screenshots/create_glue_database.png)

### Creating Glue Crawler
Click `Crawlers` underneath the `Data Catalog` on the left pane. Then click `Create crawler` toward the top right of the page.
![Crawler creation page](screenshots/create_crawler_page.png)

After clicking `create crawler`, The first step requires to name the crawler, named it `churn-raw-crawler`, then click `Next`.
![step 1 crawler setup](screenshots/step_1_crawler.png)

In the second step page named `Choose data sources and classifiers`, Select `Add a data source` underneath the `Data Sources`. In the pop up page, select `Browse S3` and choose the customer churn bucket from S3 (`churn-project-ntsikelelo`), or enter the S3 path on the space provided. Then click `Add an S3 data source`.
![Step 2 crawler](screenshots/step_2_crawler.png)

Click on next to got to step 3 named `Configure security settings`. Click on empty field below `Existing IAM role` sub-heading and choose the created `AWSGlueChurnRole` then click `Next`. ![](screenshots/step_3_crawler.png)

Choose `churn_db` as `Target database`, write `raw_` on `Table name prefix - optional` field. Leave the `Frequency` on the `On demand` setting. If it was not pre selected like that change it to `On demand`. Then click on `Next`.![Set output and scheduling](screenshots/step_4_crawler.png).

Review that everything is as expected and then click `Create crawler` on the bottom right of the page. ![Review and create](screenshots/step_5_crawler.png)

### Running the crawler
Click the check box next to the created crawler and click `Run` at the top of the page. Wait approximately 2 minutes after pressing it. ![Running crawler](screenshots/running_crawler.png)

Click on `Databases` below the `Data Catalog` heading on the left pane then click on `churn_db`. ![](screenshots/confirming_database.png)

There should be a table that starts with `raw_` as specified for the crawler instructions. ![Glue Database Table](screenshots/crawler_table.png)

### Creating Glue ETL Job
Navigate to `AWS Glue`, `Jobs` then `Visual ETL`.  Click on `Visual ETL` below the `Create job` sub-heading. 
![](screenshots/creating_glue_job_1.png)

On the page that pops up, click on `Job details` tab and name your job `churn-etl-job`. Choose `AWSGlueChurnRole` in the `IAM role` of the job. Choose `Glue 4.0` for `Glue version` and leave `Language` selection as `Python 3`. Leave `Worker type` as `G 1X`. Scroll down and write `2` on the field beneath the `Requested number of workers`. Leave other settings as they are. This selection for the job details ensures that only free-tier resources are being used. ![Job details config for Glue](screenshots/Glue_role_job_details_1.png)

Click on the `Visual` tab and click on the big blue plus icon. This allows for addition of nodes to the Glue job. Search for `AWS Glue Data Catalog` and click on it, a mini pane will appear on the right of the screen with the heading `Data source properties - Data Catalog`, select `churn_db` as the Database and select `raw_churn_project_ntsikelelo` as the table. 
![Visual config of Glue](screenshots/glue_role_visual_config.png)

Click on the big blue plus icon again to add a node. Search for `Change Schema` formerly known as `Apply Mapping`