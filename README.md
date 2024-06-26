# Predicting Used Car Prices Using CRISP-DM Framework
## Project Overview and Introduction
In the highly competitive used car market, accurate pricing is crucial for maximizing profits, attracting customers, and managing inventory effectively. Our project aims to leverage data analytics and machine learning to identify key drivers of used car prices and develop a predictive model to forecast these prices accurately. This will enable our business to make informed pricing decisions, enhance customer trust, and improve overall business performance.

## CRISP-DM Framework
We will use the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to guide our project. This widely adopted methodology provides a structured approach to data mining and ensures systematic and efficient analysis. The CRISP-DM process consists of six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

## Business Understanding

The used car market is highly dynamic, influenced by a variety of factors ranging from economic conditions to consumer preferences. For our business, accurately predicting used car prices is crucial for several reasons:

- <b>Inventory Management:</b> Knowing the right price helps in maintaining optimal inventory levels by ensuring cars are neither overstocked nor understocked.
- <b>Sales Strategy:</b> Accurate pricing can enhance competitiveness, attracting more customers and increasing sales volumes.
- <b>Customer Trust:</b> Transparent and fair pricing builds customer trust and loyalty.
- <b>Financial Planning:</b> Reliable price predictions aid in better financial forecasting and budgeting.

Given these aspects, our primary goal is to identify the key drivers that significantly influence the prices of used cars. Understanding these drivers will enable us to develop a robust pricing strategy, improve decision-making, and enhance overall business performance.

### Objectives

- <b>Identify Key Price Drivers:</b> Determine the primary factors that impact used car prices. This includes both quantitative attributes (e.g., mileage, age, brand, model, condition) and qualitative factors (e.g., market trends, economic indicators).
- <b>Develop a Predictive Model:</b> Create a predictive model that can accurately forecast the prices of used cars based on the identified key drivers. This model should be reliable, scalable, and easy to integrate into our existing business processes.
- <b>Enhance Pricing Strategy:</b> Utilize the insights from the predictive model to refine our pricing strategy. This involves setting competitive prices that reflect the true value of the cars while maximizing our profit margins.
- <b>Support Strategic Decision-Making:</b> Provide actionable insights to support strategic decisions related to procurement, sales, and marketing. This includes understanding seasonal trends, identifying high-demand vehicle types, and targeting specific customer segments.
- <b>Improve Customer Experience:</b> Offer transparent and fair pricing to customers, thereby enhancing their buying experience and building long-term relationships.
- <b>Optimize Inventory Management:</b> Use price predictions to manage inventory more effectively, ensuring the right mix of vehicles is available to meet customer demand without overstocking or understocking.

By achieving these objectives, our business aims to gain a competitive edge in the used car market, improve operational efficiency, and drive sustainable growth.

## Data Understanding
In the Data Understanding phase, we explored and familiarized ourselves with the dataset comprising 426,880 used car listings. This involved identifying key variables such as price, year, odometer readings, and categorical attributes like manufacturer, model, type, transmission, etc.. Initial data exploration allowed us to assess data quality, distribution, and relationships between variables, laying the groundwork for subsequent data preparation and modeling phases within the CRISP-DM framework.

### Data Collection
The dataset comprising 426,880 used car listings was sourced from Kaggle, a platform known for hosting diverse datasets contributed by the community.

### Data Description

The dataset contains information on 426,880 used cars, with 18 attributes detailing various aspects of each vehicle. Below is a detailed description of each column:

1. **id:** A unique identifier for each car listing.
2. **region:** The geographic region where the car is listed.
3. **price:** The listed price of the car in dollars.
4. **year:** The manufacturing year of the car.
5. **manufacturer:** The manufacturer or brand of the car (e.g., Ford, Toyota).
6. **model:** The model name of the car.
7. **condition:** The condition of the car (e.g., new, like new, excellent, good, fair, salvage).
8. **cylinders:** The number of cylinders in the car's engine.
9. **fuel:** The type of fuel the car uses (e.g., gas, diesel, electric, hybrid).
10. **odometer:** The mileage of the car (distance traveled in miles).
11. **title_status:** The status of the car's title (e.g., clean, salvage, rebuilt).
12. **transmission:** The type of transmission (e.g., automatic, manual).
13. **VIN:** The Vehicle Identification Number, a unique code used to identify individual motor vehicles.
14. **drive:** The type of drivetrain (e.g., 4wd, fwd, rwd).
15. **size:** The size category of the car (e.g., compact, mid-size, full-size).
16. **type:** The type or category of the car (e.g., sedan, SUV, truck).
17. **paint_color:** The exterior color of the car's paint.
18. **state:** The state where the car is listed.

### Data Exploration

In the Data Exploration phase, we conducted a thorough analysis to uncover patterns, relationships, and insights within the dataset. Key activities included:

- **Descriptive Statistics:** Calculated summary statistics for each numerical feature, such as mean, median, standard deviation, minimum, and maximum values. This provided a high-level overview of the data distribution and central tendencies.
- **Distribution Analysis:** Visualized the distributions of key variables (e.g., price, year, odometer) using histograms and density plots to understand their spread and identify any skewness or kurtosis. This helped in detecting potential outliers and understanding the general data shape.
- **Correlation Analysis:** Computed correlation coefficients between numerical variables to identify linear relationships. Visualized these correlations using heatmaps, which highlighted significant correlations (e.g., between year and price, odometer and price).
- **Missing Values Analysis:** Identified columns with missing values and calculated the proportion of missing data for each column. Visualized missing data patterns using heatmaps to understand the extent and distribution of missingness across the dataset.
- **Categorical Data Analysis:** Examined the distribution of categorical variables (e.g., manufacturer, fuel, transmission) using bar plots and pie charts. This analysis provided insights into the most common categories and their frequencies.
- **Outlier Detection:** Identified outliers in numerical features by visualizing box plots and scatter plots. Assessed the potential impact of these outliers on the analysis and modeling process.
- **Bivariate Analysis:** Explored relationships between pairs of variables (e.g., price vs. year, price vs. odometer) using scatter plots and box plots. This helped in understanding how different features interact and influence the target variable (price).

These exploratory analyses laid the foundation for subsequent data preparation and modeling steps, ensuring a comprehensive understanding of the dataset and guiding informed decisions throughout the project.

## Data Preparation

In the Data Preparation phase we focused on transforming the raw dataset into a clean and structured format suitable for modeling. This phase is crucial for ensuring the accuracy and reliability of our predictive model. Here’s a detailed overview of the steps taken:

- **Data Cleaning:**
    - **Handling Missing Values:** Missing values were identified and handled across various columns such as year, manufacturer, model, condition, cylinders, fuel type, odometer readings, title status, transmission type, drive type, size category, car type, paint color, and state.
    - **Handling Outliers:** remove, transforming (e.g., using log transformation), or cap outliers (set a limit on extreme values).
    - **Remove duplicates**
    - **Handling incorrect data types** None required
    - **Handling inconsistent data** (example: age shouldn't be negative), None required
- **Split data into training and test sets:**
This partitioning ensured that the model could be trained on a subset of data and tested for optimal performance. The primary reason for performing a train/test split before feature engineering and modeling is to prevent data leakage and to ensure that the evaluation of your model is accurate and indicative of its performance on unseen data.
    - Prevention of Data Leakage
    - Accurate Evaluation
    - Ethical Modeling Practices
- **Feature Engineering / Data Transformation:**
    - **Normalization/Scaling:** Numerical features such as price, vehicle_age, and odometer readings were scaled using techniques like Min-Max scaling to bring them within a standardized range, optimizing the performance of machine learning algorithms that are sensitive to varying scales.
    - **Polynomial Features:** Numerical features such as price, vehicle_age, and odometer were transformed using Polynomial features fundction with degree 2.
    - **Encoding Categorical Variables:**
        - **Target Encoding:** make_model was encoded using target encoding.
        - **one-hot encoding:** fuel, transmission, type, paint_color were encoded using one-hot encoding to convert them into numerical format.
        - **Ordinal Encoding:** condition, cylinders, title_status, drive, size were encoding using ordinal encoding.
    - **New features** were derived or modified from existing ones to enhance predictive power.
        - For example, creating a derived feature like **"vehicle age"** from the difference between the current year and the year of manufacture could provide deeper insights into pricing dynamics based on depreciation.
        - **make_model** was created to combine manufacturer and model columns.

By meticulously preparing the dataset in this manner, we established a solid foundation for building and evaluating predictive models that accurately forecast used car prices. This phase not only enhanced data quality but also streamlined subsequent phases of modeling, evaluation, and deployment within the CRISP-DM framework.

## Modeling
In the Modeling phase, we aimed to develop a robust predictive model to forecast used car prices based on the cleaned and transformed dataset. Key steps and considerations include:
- **Model Selection:**
We evaluated various regression models suitable for predicting continuous variables, such as Linear, Ridge, Lasso Regression.
Each model was selected based on its ability to handle the dataset's characteristics, interpretability, and potential for achieving high prediction accuracy.
- **Model Training:**
The selected models were trained using the training dataset, which was prepared during the data preparation phase.
Parameters were tuned using techniques like grid search combined with cross-validation to optimize model performance and prevent overfitting.

### Modeling Output
Linear Regression has the best performance among the four models with the lowest RMSE and the highest R2 value. It is the most suitable model for this dataset based on the given metrics. Similar output from Ridge and Lasso, which can further be evaluated for different alphas.

- **Results:**
    - Linear Regression: RMSE = 5676.3165, R2 = 0.6510
    - Ridge Regression: RMSE = 5676.3202, R2 = 0.6510
    - Lasso Regression: RMSE = 5676.6817, R2 = 0.6509
    - Elastic Net Regression: RMSE = 6160.6749, R2 = 0.5889
      
- **`RMSE (Root Mean Squared Error) = 5676.3165`**: This value indicates the average error in the predictions. In this context, it means that on average, the model's predictions are off by about 5676.3165 units from the actual values of the car price.

- **`R2 (R-squared) = 0.6510`**: This value indicates that approximately 65.10% of the variance in the target variable (price) is explained by the features in the model. While this is a good level of explanation, it indicates that there is still about 35% of unexplained variance, suggesting room for model improvement.

## Evaluation
In the Evaluation phase, we assessed the performance of the trained models to ensure they met our project objectives and business requirements:

- **Metrics:**
We used several evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) to quantify the accuracy of predictions. These metrics provided insights into how well the models predicted used car prices compared to actual values.

- **Validation:**
The models were validated using the test dataset, which was set aside during the data preparation phase.
This step ensured that the models generalized well to unseen data and avoided overfitting to the training dataset.

- **Variance Inflation Factor (VIF)**, a measure of multicollinearity in a set of regression variables was used but found not helpful in improving the model performance. 
  - **Interpretation of VIF Results**
The VIF results provided show that several features have inf (infinity) values, indicating perfect multicollinearity. This means that these features are exact linear combinations of other features in the dataset, leading to an inability to estimate the coefficients uniquely.

    - **Features with Infinite VIF**
      - **Paint Colors:** All paint colors have infinite VIF, suggesting they are perfectly collinear.
      - **Type Variables:** Vehicle types such as mini-van, hatchback, coupe, etc., also have infinite VIF.
      - **Fuel Types:** All fuel type categories have infinite VIF.
      - **Transmission Types:** Transmission types have infinite VIF.

    - **Results:** are rather worse after dropping 'inf' multi-collinearity columns as determined by VIF.
        - Linear Regression: RMSE = 5927.4558, R2 = 0.6194
        - Ridge Regression: RMSE = 5927.4559, R2 = 0.6194
        - Lasso Regression: RMSE = 5927.4650, R2 = 0.6194
        - Elastic Net Regression: RMSE = 6233.9973, R2 = 0.5790

- **Cross Validation and Hyperparameter Tuning** was performed to evaluate the models. The performance remains unchanged as expected since Linear Regression does not have hyperparameters to tune. It serves as a baseline for comparison. Based on the post-tuning metrics, Linear Regression is the best performing model with the lowest RMSE and highest R2 score, making it the preferred choice for predicting used car prices in this context.
    - Linear Regression: Cross-validation RMSE = 5608.2921
    - Best Ridge: {'alpha': 100.0}, RMSE = 5620.4871
    - Best Lasso: {'alpha': 0.1}, RMSE = 5620.5940
    
    - Evaluate the best models on the test set
        - Linear Regression: RMSE = 5676.3165, R2 = 0.6510
        - Ridge: RMSE = 5676.8223, R2 = 0.6509
        - Lasso: RMSE = 5676.3500, R2 = 0.6510
          
- **Best Model: Linear Regression with cross-validation**
    - Mean cross-validation score: 0.6627814416762869
    - Standard deviation of cross-validation score: 0.007372523909095059
    
    - **Sample Actual vs Predicted values**

      <img width="192" alt="image" src="https://github.com/mitbans/used-car-price-prediction/assets/166747739/3d93d1b3-027f-462c-85cd-5adf2f766e60">

    - **Actual vs Predicted values Plot**
<br>
<div align="center">
    <img width="906" alt="image" src="https://github.com/mitbans/used-car-price-prediction/assets/166747739/2fe37565-c5b8-4963-82f6-b0c650579411">
    <br>
    <br>
    <img width="883" alt="image" src="https://github.com/mitbans/used-car-price-prediction/assets/166747739/72ed88ec-1e5f-4284-b083-699c2a37962f">
</div>

<br>
- **Permutation Inportance** was used to assess the importance of the coefficients. It didn't make any difference in the performance of the model after removing negative permutation importance coefficients.
    - Linear Regression: RMSE = 5675.6693, R2 = 0.6510
    - Ridge: RMSE = 5675.6696, R2 = 0.6510
    - Lasso: RMSE = 5676.2251, R2 = 0.6510

## Conclusion
In conclusion, this project focused on predicting used car prices using a structured approach based on the CRISP-DM framework. We started with understanding business objectives and data collection, followed by thorough data preparation, modeling, evaluation, and deployment phases. Here's a summary of our findings and actionable insights:

### Interesting Findings

### Interesting Findings

| Feature      | Recommendation                                                                 | Coefficient Value          | Impact   | Interpretation                                                              |
|--------------|-------------------------------------------------------------------------------|----------------------------|----------|----------------------------------------------------------------------------|
| Fuel         | Collect more data on electric and hybrid models, and consider offering incentives for fuel-efficient options | Diesel - 4892.09          | High     | Fuel type Diesel is associated with higher prices, with electric and hybrid models also contributing to higher prices |
| Type         | Focus on marketing convertibles, offroad, pickup, truck, and coupe models     | Convertible - 2707.22      | High     | Convertibles, offroad, pickup trucks, trucks, and coupes are associated with higher prices |
| Make, Model  | Highlight popular makes and models in listings                                | 2539.73                    | High     | Certain makes and models can drastically increase the price                |
| Cylinders    | Emphasize performance aspects in high-cylinder vehicles                       | 1398.6                     | Medium   | Vehicles with more cylinders are often high-performance and can command higher prices |
| Condition    | Ensure accurate and detailed condition reports                                | 759.02                     | Medium   | Better condition typically results in higher prices                        |
| Odometer     | Highlight vehicles with lower mileage prominently                             | -2013                      | High     | Higher mileage significantly reduces the price                              |
| Title Status | Consider investigating the reasons for price differences and provide clear title status information | 1007.08                    | Medium   | Clear title status increases the vehicle's price                            |
| Transmission | Promote manual transmissions and educate buyers on their benefits             | Manual: 734, Auto: -643    | Medium   | Manual transmissions increase the price, while automatic transmissions slightly reduce it |
| Drive        | Highlight the benefits of specific drive types, such as 4WD                   | 575.37                     | Low      | Certain drive types (e.g., 4WD) can increase the price                      |
| Size         | Emphasize the advantages of larger vehicle sizes in listings                  | 428.59                     | Low      | Larger vehicle sizes can lead to a price increase                           |
| Paint Color  | Consider studying the popularity of colors in the market and highlight desirable colors | Yellow: 658, Custom: 599 | Low      | Popular or unique colors can increase the price                             |
| Vehicle Age  | Provide maintenance records and emphasize longevity for older vehicles        | -4832.72                   | Very High| Older vehicles significantly decrease in price                              |

## Actionable Insights

- **Optimize Pricing Strategy:**
    - Implement dynamic pricing based on mileage, vehicle age, and demand to maximize profits.
    - Monitor regional trends and adjust prices to align with local market conditions.
- **Enhance Inventory Management:**
    - Focus on high-demand models and manufacturers with strong resale value.
    - Use predictive models to forecast inventory turnover and manage stock levels efficiently.
- **Improve Customer Engagement:**
    - Tailor marketing efforts to highlight popular features like low mileage and specific models.
    - Provide transparent pricing and detailed vehicle histories to build trust and attract buyers.
- **Continuous Model Refinement:**
    - Regularly update the predictive model with new data to maintain accuracy.
    - Incorporate user feedback and market insights to enhance model performance.

By leveraging these insights, our business can improve its competitive edge, drive revenue growth, and enhance customer satisfaction.

## Future Work
- **Analyze Price Variations:** Investigate how prices change for specific makes and models to identify trends and optimize pricing strategies.
- **Examine Regional Drivers:** Explore state and regional data to understand local market drivers and adjust strategies accordingly.

## Repository Structure
- <code>data/vehicles.csv</code>: Contains dataset used in the analysis.
- <code>notebooks/used-car-price-prediction.jpynb</code>: Jupyter notebook with code.
- <code>README.md</code>: Summary of findings and link to notebook

## Notebook
The detailed analysis and code can be found in the Jupyter notebook <a href=" ">here</a>.
