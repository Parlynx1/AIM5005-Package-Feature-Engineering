## AIM5005 - Feature Engineering Assignment

### Author: Parlynx1  
**Email**: parlynx1@yu.edu  
**GitHub Repository**: [AIM5005-Package-Feature-Engineering](https://github.com/Parlynx1/AIM5005-Package-Feature-Engineering)

---

### Install
1. `git clone git@github.com:Parlynx1/AIM5005-Package-Feature-Engineering.git`
2. `cd AIM5005-Package-Feature-Engineering`
3. `make install`
4. You can ensure the tests pass by running `make` (which defaults to `make tests`)

### Use
- `make` will build and run all tests
- `make lr` will only run the tests for linear regression
- `make features` will run the test for features (this is your first assignment)

---

### Assignment Description

This project focuses on creating custom implementations of machine learning features, inspired by the `sklearn` API. The main objectives of this assignment include:

- **Implementing and Testing Feature Engineering Components**: The task was to create alternative implementations of common `sklearn` utilities such as the `StandardScaler`, `MinMaxScaler`, and `LabelEncoder`. These implementations mimic the `sklearn` API and include necessary methods such as `fit`, `transform`, `fit_transform`, and class-specific attributes like `classes_` for `LabelEncoder`.
  
- **Bug Fixes**: A bug was identified in the original `MinMaxScaler` implementation, and it was required to fix this bug to ensure proper scaling behavior.
  
- **Custom Tests**: Unit tests were written to ensure the correctness of the feature engineering components and to validate that the `LabelEncoder`, `StandardScaler`, and `MinMaxScaler` classes perform as expected.

---

### Modifications

#### 1. **StandardScaler Implementation**
   - Implemented a custom `StandardScaler` to standardize data by removing the mean and scaling to unit variance.
   - The custom implementation mimics the `sklearn` API with methods such as `fit`, `transform`, and `fit_transform`.

#### 2. **MinMaxScaler Bug Fix**
   - Fixed an issue in the `MinMaxScaler` implementation to ensure proper normalization of features between a given range (default is 0 to 1).

#### 3. **LabelEncoder Implementation**
   - Created a custom `LabelEncoder` class to encode categorical labels into numerical values.
   - Implemented necessary methods such as `fit`, `transform`, `fit_transform`, and the attribute `classes_` to store the classes encountered during fitting.

#### 4. **Test Cases**
   - Added test cases for `LabelEncoder` to validate correct encoding and decoding.
   - Test cases for `StandardScaler` and `MinMaxScaler` were modified to test the new and fixed implementations.

---

### Installation and Usage

 Clone this repository:
   ```bash
   git clone https://github.com/Parlynx1/AIM5005-Package-Feature-Engineering.git
