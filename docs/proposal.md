# UMBC DATA606 Capstone Project - Proposal
---

## 1. Title and Author

* **Title:** A Predictive Analytics Approach to Customer Attrition in the Telecommunications Sector
* **Author:** Gowtham Krishna Sai Relangi
* Prepared for the UMBC Data Science Master's Capstone, supervised by Dr. Chaojie (Jay) Wang.
* **My GitHub Repo:** [https://github.com/Relangi-Gowtham/UMBC-DATA606-Capstone]
* **My LinkedIn Profile:** [www.linkedin.com/in/gowtham-krishna-sai-relangi-8392691b5]
* **My Presentation:** [Soon to be uploaded]
* **My YouTube Video:** [Soon to be uploaded]

---

## 2. Background

### What's This All About?
Basically, I'm trying to figure out why customers cancel their services. In the telecom world, we call this **customer churn**, or attrition. The main idea is to use historical customer data—things like what services they have, their billing info, and some basic demographics—to build a model. This model will then give us a `Yes` or `No` prediction on whether a customer is at high risk of leaving.

### Why Does This Matter So Much?
Think about it: it costs a lot more to find a new customer than it does to keep an existing one happy. When a bunch of customers leave, it hurts a company's bottom line and their market standing. By using a predictive model, a company can stop being reactive and become **proactive**. They can reach out to at-risk customers with personalized offers or support before it's too late. It’s all about turning a problem into an opportunity to build stronger customer relationships.

### The Questions I'm Answering
For this project, I'm focused on a few key research questions:
* Can a machine learning model actually predict which customers will churn just by looking at their data?
* Out of all the information I have—like contract type or monthly charges—which factors are the most important for predicting churn?
* How do different machine learning methods, specifically **Random Forest**, **XGBoost**, and **Decision Trees**, stack up against each other when it comes to predicting churn?

---

## 3. Data

### Quick Facts
For this project, I'm using a popular **Telco Customer Churn dataset** that I found on Kaggle.

* **Size:** Around **977.4 KB**
* **Shape:** It has **7,043 rows** and **21 columns**.
* **What's a row?** Each row is a unique customer.

### My Data Dictionary
I put together this sheet to keep track of what each column means.

| Column Name | Data Type | What It Is | Potential Values |
| :--- | :--- | :--- | :--- |
| `customerID` | String | A unique ID for each customer. | e.g., "7590-VHVEG" |
| `gender` | String | Female or Male. | "Female", "Male" |
| `SeniorCitizen` | Integer | Is the person a senior citizen? | 0 (No), 1 (Yes) |
| `Partner` | String | Do they have a partner? | "Yes", "No" |
| `Dependents` | String | Do they have dependents? | "Yes", "No" |
| `tenure` | Integer | The number of months they've been a customer. | 0 to 72 |
| `PhoneService` | String | Do they have phone service? | "Yes", "No" |
| `MultipleLines` | String | Do they have multiple phone lines? | "Yes", "No", "No phone service" |
| `InternetService`| String | What kind of internet service do they have? | "DSL", "Fiber optic", "No" |
| `OnlineSecurity` | String | Do they have online security? | "Yes", "No", "No internet service" |
| `OnlineBackup` | String | Do they have online backup? | "Yes", "No", "No internet service" |
| `DeviceProtection`| String | Do they have device protection? | "Yes", "No", "No internet service" |
| `TechSupport` | String | Do they have tech support? | "Yes", "No", "No internet service" |
| `StreamingTV` | String | Do they have streaming TV? | "Yes", "No", "No internet service" |
| `StreamingMovies`| String | Do they have streaming movies? | "Yes", "No", "No internet service" |
| `Contract` | String | The type of contract they have. | "Month-to-month", "One year", "Two year" |
| `PaperlessBilling`| String | Do they use paperless billing? | "Yes", "No" |
| `PaymentMethod` | String | How do they pay? | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"|
| `MonthlyCharges` | Float | What they're charged each month. | 18.25 to 118.75 |
| `TotalCharges` | String | The total amount they've been charged so far. | Continuous numerical values |
| `Churn` | String | Did they leave the company last month? | "Yes", "No" |

### My Target & Features
* **My Target:** I’ve chosen **`Churn`** as the variable I want to predict. It's the most important one for my model!
* **My Features:** All the other columns, except for the `customerID`, are the features I'll use to train my model. They're what will help me make the churn prediction.
