
# Bank Marketing Campaign: Handling Missing Data and Categorical Encoding

This project is a continuation of a previous analysis on a bank marketing campaign dataset. The focus is on handling missing data, encoding categorical variables, and training a machine learning model to predict whether a customer will subscribe to a term deposit (`y`).

## Dataset Overview

The dataset includes the following features:

- `age`: Age of the client.
- `job`: Job type of the client.
- `marital`: Marital status of the client.
- `education`: Level of education.
- `default`: Whether the client has credit in default.
- `balance`: Balance of the client's bank account.
- `housing`: Whether the client has a housing loan.
- `loan`: Whether the client has a personal loan.
- `contact`: Communication type.
- `day`: Last contact day of the month.
- `month`: Last contact month of the year.
- `duration`: Last contact duration in seconds.
- `campaign`: Number of contacts performed during this campaign.
- `pdays`: Number of days since the client was last contacted from a previous campaign.
- `previous`: Number of contacts performed before this campaign.
- `poutcome`: Outcome of the previous marketing campaign.
- `y`: Target variable indicating whether the client subscribed to a term deposit (yes/no).

### Initial Data Inspection

The dataset is loaded, and an initial inspection is performed to understand the distribution and presence of missing values:

```python
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/bank_marketing.csv')
df.head()
```

### Handling Missing Data

The percentage of missing values is calculated for each column, and various strategies are employed to fill in missing data:

- **Job**: Missing values filled with `'unknown'`.
- **Education**: Missing values filled with `'primary'`.
- **Contact**: Missing values filled with `'unknown'`.
- **Pdays**: Missing values filled with `-1` to indicate no previous contact.
- **Poutcome**: Missing values filled with `'nonexistent'`.

These strategies were applied using custom functions:

```python
def fill_job(X_data):
    X_data['job'] = X_data['job'].fillna('unknown')
    return X_data

def fill_education(X_data):
    X_data['education'] = X_data['education'].fillna('primary')
    return X_data

def fill_contact(X_data):
    X_data['contact'] = X_data['contact'].fillna('unknown')
    return X_data

def fill_pdays(X_data):
    X_data['pdays'] = X_data['pdays'].fillna(-1)
    return X_data

def fill_poutcome(X_data):
    X_data['poutcome'] = X_data['poutcome'].fillna('nonexistent')
    return X_data

def fill_missing(X_data):
    X_data = fill_job(X_data)
    X_data = fill_education(X_data)
    X_data = fill_contact(X_data)
    X_data = fill_pdays(X_data)
    X_data = fill_poutcome(X_data)
    return X_data
```

These functions were applied to both the training and testing datasets.

### Encoding Categorical Variables

After filling in missing values, categorical variables were encoded using appropriate encoding techniques:

- **OneHotEncoder**: Used for categorical variables with non-ordinal data (`job`, `marital`, `contact`, `poutcome`).
- **OrdinalEncoder**: Used for ordinal data (`education`, `default`, `housing`, `loan`, `month`).

The encoders were trained on the training data and applied to both the training and testing datasets using a custom function:

```python
def encode_categorical(X_data):
    # Separate the numeric columns
    X_data_numeric = X_data.select_dtypes(include='number').reset_index()

    # Multicolumn encoders first
    job_encoded_df = pd.DataFrame(encode_job.transform(X_data['job'].values.reshape(-1, 1)), columns=encode_job.get_feature_names_out())
    marital_encoded_df = pd.DataFrame(encode_marital.transform(X_data['marital'].values.reshape(-1, 1)), columns=encode_marital.get_feature_names_out())
    contact_encoded_df = pd.DataFrame(encode_contact.transform(X_data['contact'].values.reshape(-1, 1)), columns=encode_contact.get_feature_names_out())
    poutcome_encoded_df = pd.DataFrame(encode_poutcome.transform(X_data['poutcome'].values.reshape(-1, 1)), columns=encode_poutcome.get_feature_names_out())

    # Concat all dfs together
    dfs = [X_data_numeric, job_encoded_df, marital_encoded_df, contact_encoded_df, poutcome_encoded_df]
    X_data_encoded = pd.concat(dfs, axis=1)

    # Add single column encoders
    X_data_encoded['education'] = encode_education.transform(X_data['education'].values.reshape(-1, 1))
    X_data_encoded['default'] = encode_default.transform(X_data['default'].values.reshape(-1, 1))
    X_data_encoded['housing'] = encode_housing.transform(X_data['housing'].values.reshape(-1, 1))
    X_data_encoded['loan'] = encode_loan.transform(X_data['loan'].values.reshape(-1, 1))
    X_data_encoded['month'] = encode_month.transform(X_data['month'].values.reshape(-1, 1))
    
    return X_data_encoded
```

### Model Training and Evaluation

The target variable `y` was also encoded using `OneHotEncoder`, and a RandomForest model was trained on the encoded data:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500)
model.fit(X_train_encoded, y_train_encoded)
```

### Overfitting Investigation

The model initially showed signs of overfitting with a perfect training score and a lower test score. To address this, the `max_depth` parameter was varied, and the balanced accuracy scores were observed:

```python
models = {'train_score': [], 'test_score': [], 'max_depth': []}

for depth in range(1, 10):
    models['max_depth'].append(depth)
    model = RandomForestClassifier(n_estimators=500, max_depth=depth)
    model.fit(X_train_encoded, y_train_encoded)
    y_test_pred = model.predict(X_test_encoded)
    y_train_pred = model.predict(X_train_encoded)

    models['train_score'].append(balanced_accuracy_score(y_train_encoded, y_train_pred))
    models['test_score'].append(balanced_accuracy_score(y_test_encoded, y_test_pred))

models_df = pd.DataFrame(models)
models_df.plot(x='max_depth')
```

A `max_depth` of 7 was found to be a good balance, reducing overfitting while maintaining decent accuracy on the test set:

```python
model = RandomForestClassifier(max_depth=7, n_estimators=100)
model.fit(X_train_encoded, y_train_encoded)
```

## Conclusion

This project demonstrates how to handle missing data, encode categorical variables, and address overfitting in a machine learning pipeline. The final model, after tuning, showed improved generalization to new data, indicating a better balance between bias and variance.

Further improvements could include more sophisticated techniques for handling missing data, feature engineering, or trying alternative models.
``` 

This README provides a detailed and structured overview of the continuation of your project, focusing on handling missing data, encoding categorical variables, and addressing overfitting in the model.
