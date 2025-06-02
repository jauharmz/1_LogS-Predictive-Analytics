# Machine Learning Project Report - Jauhar Mumtaz

## Project Domain

The solubility of organic molecules in water is a critical physical property in the medical field. This property is directly related to absorption, a key parameter in the distribution of biologically active compounds in living organisms and the environment. Therefore, solubility significantly impacts bioavailability, efficacy, and commercial value of active compounds.

Measuring water solubility with high accuracy is costly, requiring significant time, instrumentation, expertise, and often limited by the availability of physical samples. Several methods for estimating water solubility (*S*) have been developed, one of which is the *General Solubility Equation (GSE)* introduced by [Sanghvi T. *et al.*, 2003](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.4c00685). This method estimates water solubility (*S*) as a function of melting point (*T*) and the octanol-water partition coefficient (*K*):

$$log(S) = -0.01 (T - 25°C) - log(K) + 0.50$$

The octanol partition coefficient (*P*) can be determined based on the compound's structure, but determining the melting point (*T*) still requires laboratory measurements. The GSE method is highly useful when melting point data is available, but for compounds known only by their structure, estimation methods that directly utilize molecular structure are needed.

Another developed method is the *Estimated Solubility (ESOL)*, a *machine learning* model developed by [Delaney JS, 2003](https://pubs.acs.org/doi/abs/10.1021/ci034243x). ESOL uses eight molecular descriptor parameters, such as *clogP*, molecular weight (*molWT*), number of rotatable bonds (*rb*), aromatic proportion (*ap*), non-carbon proportion, hydrogen bond donors and acceptors (*hbd, hba*), and polar surface area (*psa*). The model estimates solubility using the following equation:

$$\log(S) = 0.16 - 0.63 \log(P) - 0.0062 \text{MolWT} + 0.066 \text{RB} - 0.74 \text{AP}$$

Based on 2,874 training data points, ESOL provides more robust estimates than GSE, with the following results:

| Method | *R*² | SE   | MAE  |
|--------|------|------|------|
| ESOL   | 0.69 | 1.01 | 0.75 |
| GSE    | 0.67 | 1.05 | 0.81 |

ESOL results also indicate that the most significant parameter is *clogP*, followed by molecular weight (*molWT*), aromatic proportion (*ap*), and the number of rotatable bonds (*rb*).

With technological advancements, *machine learning* models continue to evolve in terms of dataset size, *hyperparameter tuning*, and model architecture. This study aims to develop a method for estimating molecular solubility in water using a larger dataset. In this research, only *SMILES* and *log S* variables are used as primary inputs. The [SMILES-enumeration-datasets](https://github.com/summer-cola/smiles-enumeration-datasets) dataset provides *SMILES* data with various molecular descriptors, ranging from 0D, 1D, 2D, to 3D descriptors, resulting in a total of 31 parameters used as inputs for various models.

This study implements various regression models based on *machine learning* and *deep learning*, such as Neural Network (*NN*), K-Nearest Neighbors (*KNN*), Random Forest (*RF*), Support Vector Regressor (*SVR*), Elastic Net (*EN*), Decision Tree (*DT*), Extreme Gradient Boosting (*XGBoost*), Extra Trees (*ET*), and Light Gradient-Boosting Machine (*LightGBM*). The best model will be selected based on Mean Absolute Error (*MAE*), Standard Error (*SE*), and Coefficient of Determination (*R²*), with model interpretation conducted using SHapley Additive exPlanations (*SHAP*) to identify the most significant parameters.

## Business Understanding

### Problem Statements

1. Predicting the water solubility (*logS*) of a *drug-like* molecule is a crucial step in *drug discovery* as it affects the efficiency and process of drug development. Can *logS* prediction be performed using simple *machine learning* or *deep learning* models with features extracted solely from *SMILES* annotations of a molecule?
2. Among various simple *machine learning* and *deep learning* models, which model achieves the lowest Mean Absolute Error (*MAE*) and Standard Error (*SE*), as well as a high Coefficient of Determination (*R²*) in predicting *logS* based on the used features?
3. From the eight features used in the [ESOL publication](https://pubs.acs.org/doi/abs/10.1021/ci034243x#), namely *clogP*, molecular weight (*molWT*), number of rotatable bonds (*rb*), aromatic proportion (*ap*), hydrogen bond donors and acceptors (*hbd, hba*), and polar surface area (*psa*), which feature has the most significant impact on *logS*? Are there other features that contribute significantly?

### Goals

1. Determine whether *logS* prediction can be performed using features extracted from *SMILES* annotations with simple *machine learning* or *deep learning* models.
2. Identify the simple *machine learning* or *deep learning* model with the best performance based on *MAE*, *SE*, and *R²* metrics.
3. Identify the features that most significantly influence *logS* (molecular solubility in water).

### Solution Statements

1. Predict *logS* using features extracted from *SMILES* annotations of molecules through *machine learning* or *deep learning* models.
2. Test and evaluate various models with predefined *hyperparameters* and select the best model based on *MAE*, *SE*, and *R²* metrics.
3. Extract feature weights using model interpretation techniques such as SHapley Additive exPlanations (*SHAP*) to identify the most significant parameters.

## Data Understanding

The dataset used to predict *logS* of a molecule is sourced from a GitHub dataset published by `summer-cola` under the repository named **`SMILES-enumeration-datasets`**. This dataset is accessible via the following [link](https://github.com/summer-cola/smiles-enumeration-datasets) and includes various physical properties of molecules, such as *logD*, *logP*, and *logS*. The dataset used for *logS* prediction is located in the `logS` directory with the file name `traintest.csv`, containing 7,954 rows of data.

### Variable Information

The dataset includes eight main variables with the following descriptions:

| **Variable**   | **Description**                                                                                         | **Example Value**         |
|----------------|---------------------------------------------------------------------------------------------------------|---------------------------|
| `Unnamed: 0`   | Automatically generated index when data is imported.                                                    | 0                         |
| `Compound ID`  | Unique ID to identify each compound in the dataset.                                                     | C4659                     |
| `InChIKey`     | Short alphanumeric code from *International Chemical Identifier* (InChI) for global identification.      | WIKXJKUZYYOTBP-UHFFFAOYSA-N |
| `SMILES`       | *Simplified Molecular Input Line Entry System*, a molecular structure notation based on *ASCII strings*. | CCCCC(COC(=O)N)(COC(=O)NC(C)C)C |
| `logS`         | Logarithmic value of water solubility (*S*), indicating the degree of solubility in water.               | -3.633501683             |
| `logP`         | Logarithmic value of the octanol-water partition coefficient (*P*), measuring molecular lipophilicity.   | 3.504                     |
| `MW`           | Molecular weight, the total atomic mass of the molecule (in Dalton/Da units).                           | 274.357                   |
| `smi`          | Alternative representation for *SMILES*.                                                                | C C C C C ( C O C ( = O ) N ) ... |

The dataset uses molecular descriptors based on structure (*SMILES*) to generate a total of 31 additional variables through calculations of 0D, 1D, 2D, and 3D descriptors, which are used as inputs for the models.

### Descriptor Variables

| **Variable**           | **Description**                                                                                  | **Example Value**     |
|------------------------|-------------------------------------------------------------------------------------------------|-----------------------|
| `logS`                 | Logarithmic value of molecular solubility (especially for drugs) in water.                       | -2.74                 |
| `molWt`                | Molecular weight (*Molecular Weight*).                                                           | 170.92                |
| `numAtoms`             | Number of heavy atoms (excluding hydrogen) in the molecule.                                      | 8                     |
| `molMR`                | *Molecular refractivity*, the molecule's ability to refract light, related to polarizability.     | 21.6                  |
| `rings`                | Number of rings in the molecular structure.                                                      | 0                     |
| `aromatic`             | Number of aromatic rings in the molecule.                                                        | 0                     |
| `ap`                   | Aromatic proportion, the ratio of aromatic atoms to total atoms.                                 | 0.0                   |
| `chiralC`              | Number of chiral centers (carbon) in the molecule.                                               | 0                     |
| `logP`                 | Logarithmic partition coefficient, measuring molecular polarity.                                 | 2.6496                |
| `hbd`                  | Number of hydrogen bond donors (*Hydrogen Bond Donor*).                                          | 0                     |
| `hba`                  | Number of hydrogen bond acceptors (*Hydrogen Bond Acceptor*).                                    | 0                     |
| `rb`                   | Number of rotatable bonds (*Rotatable Bond*).                                                    | 1                     |
| `tpsa`                 | *Topological Polar Surface Area*, the polar surface area of the molecule.                        | 0.0                   |
| `nh2`                  | Number of amine groups in the molecule.                                                         | 0                     |
| `oh`                   | Number of hydroxyl groups in the molecule.                                                       | 0                     |
| `balabanJ`             | Balaban index, a measure of molecular topological compactness.                                   | 4.020392              |
| `bertzCT`              | Bertz topological complexity, measuring molecular structure complexity based on graph theory.    | 67.01955              |
| `hallKierAlpha`        | Hall-Kier Alpha index, related to molecular shape and polarizability.                            | 0.3                   |
| `ipc`                  | Information Content Index, measuring molecular structure diversity.                              | 21.306059             |
| `chi0`                 | Chi 0 path index, measuring molecular topology based on atom count and type.                     | 7.0                   |
| `chi1`                 | Chi 1 path index, measuring atomic bonding patterns in the molecule.                             | 3.25                  |
| `kappa1`               | Molecular kappa 1 index, measuring molecular flexibility.                                        | 8.3                   |
| `kappa2`               | Molecular kappa 2 index, another variation for measuring molecular flexibility.                  | 1.91511               |
| `kappa3`               | Molecular kappa 3 index, a further variation for measuring molecular flexibility.                | 2.046098              |
| `fractionCSP3`         | Fraction of carbon atoms with sp³ hybridization.                                                 | 1.0                   |
| `asphericity`          | Asphericity, a measure of the molecule's deviation from a perfect sphere.                        | 0.072556              |
| `eccentricity`         | Eccentricity, measuring the asymmetry of atom distribution in the molecule.                      | 0.785158              |
| `inertialShapeFactor`  | Inertial shape factor, indicating molecular shape based on atom mass distribution.               | 0.003042              |
| `radiusOfGyration`     | Radius of gyration, measuring atom spread relative to the molecule's center of mass.             | 1.836359              |
| `spherocityIndex`      | Sphericity index, indicating how closely the molecule's shape resembles a sphere.                | 0.711911              |
| `ncp`                  | Non-carbon proportion relative to total atoms in the molecule.                                   | 0.75                  |
| `ecfp`                 | Extended Circular Fingerprints, a bit-based molecular representation.                            | [0, 0, 1, ...]        |

### Data Type Information

| **Data Type** | **Variables**                                                                                      | **Description**                                         |
|---------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| **Float**     | `logS`, `molWt`, `molMR`, `ap`, `logP`, `tpsa`, `balabanJ`, `bertzCT`, `hallKierAlpha`, `ipc`, `chi0`, `chi1`, `kappa1`, `kappa2`, `kappa3`, `fractionCSP3`, `asphericity`, `eccentricity`, `inertialShapeFactor`, `radiusOfGyration`, `spherocityIndex`, `ncp`. | Data from mathematical calculations and fractions.      |
| **Integer**   | `numAtoms`, `rings`, `aromatic`, `chiralC`, `hbd`, `hba`, `rb`, `nh2`, `oh`.                       | Data from integer counts.                              |
| **List**      | `ecfp`.                                                                                            | Data containing 2048-bit molecular interpretation (0 and 1). |

## Data Cleaning

### Handling Duplicate Data

After inspection, the dataset contains no missing data (*Null* or *NaN*) requiring imputation. However, one duplicate row was found. This duplicate was removed to ensure dataset integrity.

### Statistical Description

Below is the statistical description of the data before cleaning:

| | logS | molWt | numAtoms | molMR | rings | aromatic | ap | chiralC | logP | hbd | ... | kappa1 | kappa2 | kappa3 | fractionCSP3 | asphericity | eccentricity | inertialShapeFactor | radiusOfGyration | spherocityIndex | ncp |
|-|------|-------|----------|-------|-------|----------|-------|---------|------|-----|-----|--------|--------|--------|--------------|-------------|--------------|--------------------|------------------|-----------------|-----|
| count | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.00000 | 7954.000000 | 7954.000000 | ... | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7954.000000 | 7.954000e+03 | 7954.000000 | 7954.000000 |
| mean | -2.981528 | 292.151987 | 19.181795 | 75.840784 | 1.975107 | 1.195248 | 0.352023 | 0.97297 | 1.912550 | 1.239125 | ... | 3.924647 | 6.902837 | 5.105968 | 0.459514 | 0.392271 | 0.937606 | 0.002492 | 3.618187e+00 | 0.250457 | 0.270458 |
| std | 2.200720 | 138.909559 | 9.048712 | 34.724211 | 1.461655 | 0.982673 | 0.260995 | 2.21534 | 2.510816 | 1.513059 | ... | 2.528799 | 4.207856 | 22.793668 | 0.301113 | 0.193149 | 0.065069 | 0.010385 | 1.462360e+00 | 0.161920 | 0.133211 |
| min | -16.259392 | 16.043000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00000 | -46.668600 | 0.000000 | ... | 0.000000 | 0.000000 | -27.040000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.469447e-18 | 0.000000 | 0.000000 |
| 25% | -4.259388 | 197.190000 | 13.000000 | 51.231650 | 1.000000 | 0.000000 | 0.000000 | 0.00000 | 0.779440 | 0.000000 | ... | 2.475306 | 4.088493 | 2.171001 | 0.235294 | 0.239931 | 0.913155 | 0.000463 | 2.823452e+00 | 0.131913 | 0.181818 |
| 50% | -2.824600 | 273.798500 | 18.000000 | 72.044850 | 2.000000 | 1.000000 | 0.375000 | 0.00000 | 2.045950 | 1.000000 | ... | 3.534830 | 6.038287 | 3.400000 | 0.428571 | 0.374412 | 0.957519 | 0.001016 | 3.473871e+00 | 0.230156 | 0.250000 |
| 75% | -1.489481 | 359.713500 | 23.000000 | 93.985125 | 3.000000 | 2.000000 | 0.545455 | 1.00000 | 3.401700 | 2.000000 | ... | 4.818714 | 8.679717 | 5.344128 | 0.666667 | 0.532319 | 0.981270 | 0.002296 | 4.223524e+00 | 0.347257 | 0.333333 |
| max | 1.580000 | 1583.582000 | 109.000000 | 370.217200 | 16.000000 | 12.000000 | 1.000000 | 27.00000 | 20.854600 | 19.000000 | ... | 72.265273 | 62.805231 | 1128.960000 | 1.000000 | 1.000000 | 1.000000 | 0.339204 | 5.985842e+01 | 0.999963 | 1.000000 |

<p>8 rows × 31 columns</p>

### Handling Outliers

<p align="center">
  <img src="./images/outlier.png">
  Box plot of outliers for variables used in <b>ESOL</b>.
</p>

Outlier distribution was examined for the eight main variables from the ESOL publication, namely `logP`, `molWt`, `rb`, `ap`, `ncp`, `hbd`, `hba`, and `tpsa`. Based on *box plot* analysis, several variables have outlier values:

| **Variable** | **Data Distribution**                     | **Range**                         |
|--------------|-------------------------------------------|-----------------------------------|
| `logP`       | Centered around the range `0.8 - 3.5`     | `-46.6 - 20.8`                   |
| `molWt`      | Centered around the range `197.1 - 359.7` | `16 - 1583.5`                    |
| `rb`         | Centered around the range `3 - 9`         | `0 - 59`                         |
| `ap`         | Centered around the range `0 - 0.5`       | `0 - 1`                          |
| `ncp`        | Centered around the range `0.1 - 0.3`     | `0 - 1`                          |
| `hbd`        | Centered around the range `0 - 2`         | `0 - 19`                         |
| `hba`        | Centered around the range `2 - 4`         | `0 - 35`                         |
| `tpsa`       | Centered around the range `25.3 - 71.4`   | `25.3 - 601.8`                   |

**Interpretation**:
- Outlier values are considered relevant as they reflect the physical properties of molecules based on their structure.
- No outlier removal was performed to avoid losing significant information from the dataset.

### Univariate Analysis

<p align="center">  
  <img src="./images/Univariate_logS.png" alt="Histogram of logS">  
</p>  

**Label Variable (`logS`)**  
The distribution of the `logS` label shows a primary spread in the range `-7 - 0.1`, containing over 100 data points. This distribution covers categories from *poorly soluble* to *highly soluble* based on the [SwissADME](https://www.nature.com/articles/srep42717) scale:

$$insoluble \lt −10 \lt poorly \lt −6 \lt moderately \lt −4 \lt soluble \lt −2 \lt very \lt 0 \lt highly$$

<p align="center">
  <img src="./images/Univariate_rest.png"> 
</p>

<p align="center">  
  <small>Histogram plot of the <b>logS</b> variable as the label.</small>  
</p>  

**Input Features**  
Most input features exhibit skewed distributions. Features such as `rings`, `hbd`, `hba`, `rb`, and `chi0` show distinct clusters, indicating the presence of specific groups within the data.

#### Specific Feature Insights:
1. **Molecular Properties**:  
   - Features like `molWt`, `numAtoms`, and `molMR` show positive correlations. Larger molecules tend to have higher values for these features.

2. **Hydrogen Bonds**:  
   - `hbd` (number of hydrogen bond donors) and `hba` (number of hydrogen bond acceptors) exhibit a negative correlation. Molecules with more hydrogen bond donors tend to have fewer acceptors.

3. **Ring Structure**:  
   - `rings` and `aromatic` are closely related. All aromatic rings are included in the `rings` category, but not all rings are aromatic.

4. **Shape Description**:  
   - Features like `asphericity`, `eccentricity`, and `spherocityIndex` provide insights into the shape and distribution of molecules, with more complex molecules showing higher values for these features.

### Multivariate - Numerical Features

To understand relationships between numerical features, a correlation analysis was performed using a correlation matrix. The results are as follows:

<p align="center">  
  <img src="./images/corelation_matrix.png" alt="Correlation Matrix">  
</p>  

<p align="center">  
  <small>Correlation matrix plot between variables.</small>  
</p>  

#### Strong Positive Correlation with `logS`:
- Larger molecules (`molWt`, `numAtoms`, and `molMR`) tend to have lower solubility (`logS` is smaller).
- Molecules with more rings or aromatic structures tend to have lower solubility, likely due to resonance stabilization or free electron pair (*PEB*) effects.

#### Moderate Positive Correlation with `logS`:
- More lipophilic molecules (`logP` higher) tend to have lower solubility.
- Molecules with larger polar surface areas (`tpsa`) also tend to have lower solubility.

#### Weak or Insignificant Correlation with `logS`:
- Many other molecular descriptors have weak or insignificant correlations with `logS`, indicating minimal influence in solubility prediction.

#### High Correlation Between Descriptors:
- Features like `molWt`, `numAtoms`, `bertzCT`, `balabanJ`, `chi0`, and `chi1` show high positive correlations with each other, indicating that these features provide similar or related information about molecular structure.

## Data Preparation

### Unpack List Feature

The `ecfp` feature in the dataset contains a molecular representation in the form of a binary fingerprint list with a length of 2,048 bits. Each element has a value of `1` (True) or `0` (False). To facilitate processing by the model, this feature was *unpacked* into 2,048 separate columns. As a result, each column represents one element of the fingerprint, enabling its use as an individual numerical variable by machine learning algorithms.

### Data Splitting: Train and Test

The dataset was divided into two subsets:
1. **Train**: Used to train the model.
2. **Test**: Used to validate the model and evaluate performance.

The split ratio was `90:10`, resulting in the following data shapes:

| **Category**   | **Shape**           |
|----------------|---------------------|
| Train Data     | (`7157`, `2078`)    |
| Train Label    | (`7157`,)           |
| Test Data      | (`796`, `2078`)     |
| Test Label     | (`796`,)            |

### Scaling and Normalization

At this stage, numerical features of type `float` were standardized or normalized to improve consistency in scale across features. Meanwhile, features of type `int` were left unchanged to avoid losing important information from discrete data.

Various scaling and normalization techniques were tested, with evaluation based on *Negative Mean Squared Error (Negative MSE)*. The best technique was the **Quantile Transformer** with a **Uniform** distribution, achieving the best evaluation score of **`-1.4793369599164947`**.

The following table lists the tested algorithms and their effects on the data:

| **Algorithm**               | **Description**                                                                                  | **Effect on Data**                                      |
|-----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| **Standard Scaler**         | Scales data to have a mean of 0 and a standard deviation of 1.                                   | Aligns data (mean = 0, std = 1).                       |
| **Min-Max Scaler**          | Scales data to a specific range (default 0 to 1).                                                | Shifts data to the range [0, 1].                       |
| **Robust Scaler**           | Uses median and IQR to reduce the impact of outliers.                                            | Reduces the impact of outliers.                        |
| **Quantile Transformer**    | Transforms data to follow a specific distribution (Normal/Uniform).                              | Makes the distribution more uniform or normal.         |
| **Power Transformer**       | Applies a power transformation (Yeo-Johnson) to stabilize variance.                              | Reduces *skewness* to approach a Gaussian distribution. |

---

### Mathematical Formulas

Below are the mathematical formulas for each algorithm used:

#### **1. Standard Scaler**

$$z_i = \frac{x_i - \mu}{\sigma}$$

Where:
  - $x_i$: Original data value.
  - $z_i$: Scaled data value.
  - $\mu$: Mean of the entire data.
  - $\sigma$: Standard deviation of the entire data.

#### **2. Min-Max Scaler**

$$z_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}$$

Where:
  - $x_i$: Original data value.
  - $\min(x)$: Minimum value in the dataset.
  - $\max(x)$: Maximum value in the dataset.

#### **3. Robust Scaler**

$$z_i = \frac{x_i - Q_2}{Q_3 - Q_1}$$

Where:
  - $Q_2$: Median or second quartile.
  - $Q_1$: First quartile (25th percentile).
  - $Q_3$: Third quartile (75th percentile).

#### **4. Quantile Transformer (Normal Distribution)**

$$z_i = \Phi^{-1}\left(\frac{\text{rank}(x_i)}{n + 1}\right)$$

Where:
  - $\Phi^{-1}$: Inverse cumulative distribution function (quantile function).
  - $\text{rank}(x_i)$: Rank of the value $x_i$ in the dataset.
  - $n$: Total number of data points.

#### **5. Quantile Transformer (Uniform Distribution)**

$$z_i = \frac{\text{rank}(x_i)}{n + 1}$$

Where:
  - $\text{rank}(x_i)$: Rank of the value $x_i$ in the dataset.
  - $n$: Total number of data points.

#### **6. Power Transformer (Yeo-Johnson Distribution)**

$$
z_i = 
\begin{cases} 
\frac{(x_i + 1)^{\lambda} - 1}{\lambda}\text{, } & \text{if } x_i \geq 0 \text{, } \lambda \neq 0 \\\ 
\log(x_i + 1)\text{, } & \text{if } x_i \geq 0 \text{, } \lambda = 0 \\\ 
\frac{-(|x_i| + 1)^{2 - \lambda} - 1}{2 - \lambda}\text{, } & \text{if } x_i < 0 \text{, } \lambda \neq 2 \\\ 
-\log(|x_i| + 1)\text{, } & \text{if } x_i < 0 \text{, } \lambda = 2 
\end{cases}
$$

Where:
  - $x_i$: Original data value (including negative values if present).
  - $\lambda$: Transformation parameter determined via *Maximum Likelihood Estimation (MLE)*.
  - $\log$: Natural logarithm ($\ln$).

---

### Data Transformation

After applying the **Quantile Transformer (Uniform)**, the data distribution became more uniform. The minimum and maximum values were limited to `-5.1993` and `5.1993`, while the median and mean were within a balanced range, as shown in the following percentiles:

| **Percentile** | **Value**    |
|----------------|--------------|
| Minimum        | `-5.1993`    |
| Quartile-1     | `-0.6743`    |
| Median         | `-0.0009`    |
| Quartile-3     | `0.6737`     |
| Maximum        | `5.1993`     |

This transformation was applied to the training data first. The same scaling was then applied to the test data to ensure consistency and prevent *overfitting* due to differing data distributions.

## Modeling

Nine algorithms were used to build the models, consisting of eight *machine learning* algorithms and one *deep learning* algorithm:

| Model | Description | Feature Handling | Overfitting Risk | Suitable for High Dimensions? | Likelihood of Good Performance |
|-|-|-|-|-|-|
| **Neural Network**  | Layer-based neuron model; suitable for high complexity with tuning | Handles many features well, requires careful tuning | Medium-High | Yes, with proper regularization | **Medium**: High potential but risks overfitting with limited data |
| **K-Nearest Neighbors** (KNN) | Distance-based simple model; performance drops in high dimensions | Struggles with high dimensions | Low | No | **Low**: Likely to underperform due to high-dimensional issues |
| **Random Forest** (RF) | Ensemble tree-based model, robust to noise and easy to use | Handles many features well | Low (due to ensemble averaging) | Yes | **High**: Effective for high-dimensional and binary feature data |
| **Support Vector Regression** (SVR) | Margin-based model; sensitive to kernel and hyperparameters | Effective in high dimensions, requires tuning | Medium | Yes, but sensitive to kernel | **Medium**: Can perform well with proper parameter tuning |
| **ElasticNet** (EN) | Combines L1 (Lasso) and L2 (Ridge) regularization; suitable for sparse data | Suitable for sparse and high-dimensional data | Medium | Yes | **Medium**: Requires tuning; good for sparse data |
| **Decision Tree** (DT) | Simple tree-based model; prone to overfitting without pruning | Simple but prone to overfitting | High | No | **Medium-Low**: May struggle with small datasets and overfitting |
| **XGBoost** (XGB) | Fast and efficient gradient boosting; popular in competitions | Excellent for high-dimensional data | Low | Yes | **High**: Strong candidate due to robustness and feature selection |
| **Extra Trees** (ET) | Random Forest variation; faster due to random splitting | Similar to RF but less sensitive to noise | Low | Yes | **High**: Reliable and efficient for this data type |
| **LightGBM** (LGBM) | Histogram-based boosting model; highly efficient for large datasets | Excellent for binary/categorical high-dimensional features | Low | Yes | **High**: Strong candidate, efficient for binary features |

### Model Characteristics

1. **Overfitting Control**: Ensemble models like `RandomForest`, `ExtraTrees`, `XGBoost`, and `LightGBM` have built-in methods to prevent overfitting, making them suitable for small, high-dimensional datasets.
2. **Binary Feature Handling**: Models like `LightGBM` and `XGBoost` are highly efficient with sparse or binary features, such as the `ECFP` fingerprint.
3. **Dimension Sensitivity**: `KNN` and `Decision Trees` often struggle with high-dimensional data, making them less likely to perform well on this dataset.

### Hypothesis

* **Best Choices**: `RandomForest`, `ExtraTrees`, `XGBoost`, and `LightGBM` due to their robustness and ability to generalize well on small datasets.
* **Underperformers**: `KNN` and `Decision Trees` for high-dimensional data, unless reduced through techniques like PCA.

### Hyperparameters

| Model | Parameter | Range/Choices | Optimal Value |
|-|-|-|-|
| **NeuralNetR** | epochs | 34 - 35 | 34 |
|                | patience | 5 - 6 | 5 |
|                | batch_size | 89 - 90 | 89 |
|                | lr | 9e-4 - 2e-3 | 0.0023 |
|                | weight_decay | 5e-5 - 6e-5 | 0.0001 |
| | | | |
| **KNN**        | n_neighbors | 5 - 6 | 5 |
|                | p | 1 - 2 | 1 |
|                | weights | ['uniform', 'distance'] | distance |
| | | | |
| **RandomForest** | n_estimators | 95 - 96 | 96 |
|                | max_depth | 11 - 12 | 12 |
|                | min_samples_split | 3 - 4 | 4 |
|                | min_samples_leaf | 1 - 2 | 1 |
|                | bootstrap | [True, False] | True |
| | | | |
| **SVR**        | C | 47 - 48 | 48 |
|                | epsilon | 0.1 - 0.3 | 0.247 |
|                | gamma | ['scale', 'auto'] | scale |
|                | kernel | ['rbf', 'linear'] | rbf |
| | | | |
| **ElasticNet** | alpha | 0.1 - 0.2 | 0.101 |
|                | l1_ratio | 0.2 - 0.3 | 0.234 |
| | | | |
| **DecisionTree** | max_depth | 3 - 5 | 4 |
|                | min_samples_split | 3 - 4 | 3 |
|                | min_samples_leaf | 2 - 3 | 2 |
| | | | |
| **XGBoost**    | n_estimators | 179 - 180 | 179 |
|                | learning_rate | 0.09 - 0.15 | 0.092 |
|                | max_depth | 4 - 5 | 4 |
|                | min_child_weight | 1 - 2 | 1 |
| | | | |
| **ExtraTrees** | n_estimators | 198 - 199 | 199 |
|                | max_depth | 8 - 9 | 8 |
|                | min_samples_split | 2 - 3 | 2 |
|                | min_samples_leaf | 3 - 4 | 4 |
|                | bootstrap | [True, False] | False |
| | | | |
| **LightGBM**   | n_estimators | 119 - 120 | 119 |
|                | learning_rate | 0.1 - 0.2 | 0.15 |
|                | max_depth | 10 - 11 | 10 |
|                | num_leaves | 23 - 24 | 23 |
|                | min_child_weight | 1 - 2 | 1 |
|                | colsample_bytree | 0.8 - 0.9 | 0.9 |
|                | min_data_in_leaf | 29 - 30 | 29 |
|                | min_gain_to_split | 0.07 - 0.08 | 0.072 |

The `Range` for hyperparameters is narrow as it represents the final `validation` after iterative tuning across multiple ranges.

## Evaluation

### MAE, SE, and R² Metrics

Model evaluation used three parameters: Mean Absolute Error (MAE), Squared Error (SE), and R-squared (R²).

#### Mean Absolute Error (MAE)

- **Definition**: MAE is the average absolute difference between actual and predicted values. This metric provides a simple measure of prediction error without considering the direction of the error.
- **Formula**:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

- **Interpretation**: MAE is easy to understand as it shows the average magnitude of errors in the units of the target variable. A lower value indicates better model accuracy.

#### Squared Error (SE)

- **Definition**: SE is the sum of squared differences between actual and predicted values. This metric gives more weight to larger errors, making it useful for models aiming to minimize significant deviations.
- **Formula**:

$$\text{SE} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

- **Interpretation**: SE emphasizes larger errors, making it useful for identifying models that reduce significant deviations. It is typically used as an intermediate calculation (e.g., for MSE or RMSE) and does not provide a direct measure like MAE.

#### R-squared (R²)

- **Definition**: R² represents the proportion of variance in the target variable that can be explained by the features. It measures how well the model captures data variability.
- **Formula**:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

- **Interpretation**: R² ranges from 0 to 1, with 1 indicating perfect prediction. Higher values indicate better model performance, as they show the model explains a larger portion of the variance in the target variable.

#### Notes on Variables and Symbols:
- \( y_i \): Actual (observed) value for the \( i \)-th data point.
- \( \hat{y}_i \): Predicted value for the \( i \)-th data point.
- \( \bar{y} \): Mean of the actual values.
- \( n \): Total number of data points.
- \( \sum \): Summation operator.
- \( |\cdot| \): Absolute value function.

<p align="center">
  <img src="./images/matriks_evaluasi_model.png">
  Bar plot of evaluation metrics (<b>MAE</b>, <b>SE</b>, and <b>R²</b>) for each model.
</p>

- **Overall Best Models**: `SVR` and `ElasticNet (EN)` show the best generalization across all metrics (MAE, SE, $R^2$), making them the most reliable choices for test data.
- **Robust but Slightly Overfit Models**: `NeuralNetwork (nnR)`, `RandomForest (rfR)`, `ExtraTrees (etR)`, and `XGBoost (xgbR)` perform well but show slight overfitting, which can be improved with further optimization.
- **Overfit Models**: `KNN (knnR)` and `DecisionTree (dtR)` exhibit significant overfitting with poor generalization on test data, requiring improvements to enhance test performance.

<p align="center">
  <img src="./images/matriks_evaluasi_model_ESOL_baseline.png">
  Bar plot of evaluation metrics (<b>MAE</b>, <b>SE</b>, and <b>R²</b>) for each model compared to the ESOL baseline.
</p>

Based on the performance analysis of models with and without the ESOL baseline, **`SVR`** and **`ElasticNet (EN)`** are the best models due to their strong generalization across all metrics (MAE, SE, and $R^2$), matching or surpassing the ESOL baseline. Models like **`NeuralNetwork`**, **`RandomForest`**, **`ExtraTrees`**, and **`XGBoost`** show strong performance but slight overfitting, which can be improved with regularization. Conversely, **`KNN`** and **`DecisionTree`** tend to overfit with poor test performance, requiring significant adjustments to improve generalization. These results provide guidance for selecting the best model based on solubility prediction needs.

### Predictive vs. Actual Comparison

For a detailed analysis of model behavior based on predictions, the following is a scatter plot of `Predicted` vs. `Actual` values with $R^2$ values.

<p align="center">
  <img src="./images/pred_vs_act.png">
  Scatter plot of <b>predicted</b> vs. <b>actual</b> values for each model.
</p>

Based on **R²** values, models like `RandomForest`, `SVR`, `XGBoost`, and `LightGBM` show excellent performance in predicting data, with **R²** values on test data approaching or exceeding 0.79. **LightGBM** has the best performance with an **R²** of 0.813, followed by **XGBoost** (0.798) and **RandomForest** (0.792). The **NeuralNetwork** also shows solid performance with an **R²** of 0.774, close to the top-performing models.

In contrast, while **KNN** has a decent **R²** value (0.708), it shows signs of **overfitting**. This is evident from the perfect **MAE** on training data (0.0) but a higher value on test data (0.820), indicating an inability to capture patterns in unseen data. **DecisionTree** has the lowest performance among all models, with an **R²** of 0.616, and its prediction pattern appears less flexible, as seen in the graph with many constant predicted values along the `y`-axis.

Although models like **ExtraTrees** perform reasonably well (**R²** 0.769), they show limitations in certain prediction ranges, particularly at extreme values. **ElasticNet** also performs adequately with an **R²** of 0.704 but is relatively lower compared to models like **SVR** (0.791).

The performance of **XGBoost** and **LightGBM** is already excellent, but with more optimal **hyperparameter tuning** (e.g., number of trees, maximum depth, or learning rate), these models could achieve even better results, especially on complex datasets.

Overall, the prediction patterns indicate that data distribution and model parameters significantly impact the final results, with some models being more sensitive to certain parameters than others.

### Random Data Tester

| | 4823 | 66 | 6380 | 3674 | 4273 |
|-|-|-|-|-|-|
| **True LogS** | -2.003185 | -1.687449 | -1.944657 | -2.490308 | -3.41 |
| NeuralNetwork | -3.575 | -2.375 | -2.675 | -2.825 | -1.953 |
| KNN           | -2.235 | -2.517 | -2.301 | -3.093 | -2.13 |
| RandomForest  | -2.87 | -2.713 | -2.733 | -2.553 | -2.925 |
| SVR           | -3.091 | -2.32 | -2.413 | -2.769 | -1.983 |
| ElasticNet    | -3.322 | -2.368 | -3.36 | -3.19 | -2.947 |
| DecisionTree  | -3.992 | -2.886 | -2.774 | -3.882 | -2.774 |
| XGBoost       | -2.689 | -2.785 | -2.787 | -2.593 | -3.011 |
| ExtraTrees    | -2.917 | -2.539 | -2.763 | -2.812 | -2.732 |
| LGBM          | -2.472 | -2.774 | -2.611 | -2.686 | -2.708 |
| Closest Model 1 | KNN | SVR | KNN | RandomForest | XGBoost  |
| Closest Model 2 | LGBM | ElasticNet | SVR | XGBoost | ElasticNet |
| Closest Model 3 | XGBoost | NeuralNetwork | LGBM | LGBM | RandomForest |

It is evident that top-performing models like `RandomForest`, `SVR`, `XGBoost`, and `LightGBM` frequently appear in the `Top 3` in the above `Random Test`.

### Feature Importance

At this stage, `Feature Importance` was measured using `SHAP` for the best models based on 'R²', 'MAE', and 'SE' values compared to `ESOL` and `GSE`.

| Model          | R²   | SE   | MAE  |
|----------------|-------|------|------|
| ESOL           | 0.69  | 1.01 | 0.75 |
| GSE            | 0.67  | 1.05 | 0.81 |
| | | |
| LGBM           | 0.82  | 0.03 | 0.60 |
| XGBoost        | 0.80  | 0.03 | 0.65 |
| RandomForest   | 0.80  | 0.03 | 0.67 |
| SVR            | 0.80  | 0.03 | 0.63 |
| ExtraTrees     | 0.78  | 0.04 | 0.73 |
| NeuralNetwork  | 0.77  | 0.04 | 0.66 |

1. LGBM

<p align="center">
  <img src="./images/1_shap_lgbm.png">
</p>

The `SHAP` plot above shows the top 10 features with the highest contribution weights, with three features used in `ESOL`: `logP`, `molWt`, and `hba`. The top three features are `logP`, `molMR`, and `balabanJ`.

2. XGBoost

<p align="center">
  <img src="./images/2_shap_xgb.png">
</p>

Similar to `LGBM`, the `SHAP` plot above shows the top 10 features with the highest contribution weights, with three features used in `ESOL`: `logP`, `molWt`, and `hba`. The top three features are `logP`, `molMR`, and `balabanJ`.

3. RandomForest

<p align="center">
  <img src="./images/3_shap_rf.png">
</p>

Similar to `LGBM` and `XGBoost`, the `SHAP` plot above shows the top 10 features with the highest contribution weights, with three features used in `ESOL`: `logP`, `molWt`, and `hba`. The top three features are `logP`, `molMR`, and `balabanJ`.

4. SVR

<p align="center">
  <img src="./images/4_shap_svr.png">
</p>

The `SHAP` plot above shows the top 10 features with the highest contribution weights, with six features used in `ESOL`: `numAtoms`, `logP`, `molWt`, `hba`, `hbd`, and `tpsa`. The top three features are `numAtoms`, `logP`, and `molWt`.

5. ExtraTrees

<p align="center">
  <img src="./images/5_shap_et.png">
</p>

The `SHAP` plot above shows the top 10 features with the highest contribution weights, with four features used in `ESOL`: `logP`, `tpsa`, `numAtoms`, and `molWt`. The top three features are `logP`, `molMR`, and `bertzCT`.

6. NeuralNetwork

<p align="center">
  <img src="./images/6_shap_nn.png">
</p>

The `SHAP` plot above shows the top 10 features with the highest contribution weights, with six features used in `ESOL`: `logP`, `numAtoms`, `molWt`, `tpsa`, `hba`, and `rb`. The top three features are `logP`, `numAtoms`, and `molWt`.

Visualizations from the best models (`LGBM`, `XGBoost`, and `RandomForest`) show that `logP` is the most influential feature in predicting `logS`. This aligns with known `logS` formulas, such as:

$$log(S) = -0.01 (T - 25°C) - log(P) + 0.50$$

`logP` (hydrophobicity) is a primary factor. Compounds with higher `logP` values (more hydrophobic) generally have lower water solubility, explaining why `logP` significantly influences `logS` predictions.

Other features, such as `molMR` (molar refractivity) and `molWt` (molecular weight), are also relevant, as seen in another formula:

$$\log(S) = 0.16 - 0.63 \log(P) - 0.0062 \text{MolWT} + 0.066 \text{RB} - 0.74 \text{AP}$$

This formula further emphasizes the inverse relationship between `logP` and solubility. The consistent importance of `logP` in empirical formulas and machine learning models highlights its central role in reflecting molecular interactions with solvents that determine solubility.

The following is the total presence of features in the top 10 features for each model:

| Feature            | Count |
|--------------------|-------|
| **logP**           | 6     |
| **molMR**          | 6     |
| **molWt**          | 6     |
| **hba**            | 5     |
| **kappa1**         | 5     |
| **balabanJ**       | 4     |
| **bertzCT**        | 4     |
| **ncp**            | 4     |
| **numAtoms**       | 3     |
| **chi0**           | 3     |
| **tpsa**           | 3     |
| **chi1**           | 2     |
| **hallKierAlpha**  | 2     |
| **oh**             | 2     |
| **fractionCSP3**   | 1     |
| **hbd**            | 1     |
| **ipc**            | 1     |
| **rb**             | 1     |
| **kappa2**         | 1     |

## Conclusion

1. Predicting *logS* using descriptors from *SMILES* molecular data is feasible with *machine learning* and *deep learning* models.
2. Models such as `LGBM`, `XGBoost`, `RandomForest`, `SVR`, `ExtraTrees`, and `NeuralNetwork` with *hyperparameter tuning* outperform the *ESOL* baseline.
3. Key *ESOL* features like `logP`, `molWt`, and `hba` frequently appear in the top 10 most significant features. However, across all models, `logP`, `molWt`, and `molMR` consistently appear.

## Recommendations

1. To improve descriptor accuracy, molecular geometry can be optimized using *Quantum Mechanics* approaches like `B3LYP` instead of `MMFF94` or `UFF`, albeit at a higher computational cost.
2. Further development can focus on aspects such as: focusing on a single model with *hyperparameter tuning*, data engineering (*K-Fold Cross Validation*, *PCA*, *Standardization*), selecting and using significant descriptors, comparing model performance based on metrics against published formulas or models like *GSE*, *ESOL*, and *SwissADME*.
3. Consider molecular properties based on *Lipinski’s Rule of 5*.
4. Derive a *Final Equation* from the developed model and compare it with published equations like *GSE* and *ESOL*.

## References

1. Ahmad I, Kuznetsov AE, Pirzada AS, Alsharif KF, Daglia M, Khan H. 2023. Computational pharmacology and computational chemistry of 4-hydroxyisoleucine: Physicochemical, pharmacokinetic, and DFT-based approaches. *Front Chem.* 11 April:1–15. [doi:10.3389/fchem.2023.1145974](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2023.1145974/full).

2. Daina A, Michielin O, Zoete V. 2017. SwissADME: a free web tool to evaluate pharmacokinetics, drug-likeness and medicinal chemistry friendliness of small molecules. *Sci Rep.* 7(1):42717. [doi:10.1038/srep42717](https://www.nature.com/articles/srep42717).

3. Delaney JS. 2004. ESOL: Estimating aqueous solubility directly from molecular structure. *J Chem Inf Comput Sci.* 44(3):1000–1005. [doi:10.1021/ci034243x](https://pubs.acs.org/doi/10.1021/ci034243x).

4. Ranjith D RC. 2019. SwissADME predictions of pharmacokinetics and drug-likeness properties of small molecules present in *Ipomoea mauritiana Jacq.* *J Pharmacogn Phytochem.* 8(5):2063–2073. [https://www.phytojournal.com/archives/2019.v8.i5.9904/swissadme-predictions-of-pharmacokinetics-and-drug-likeness-properties-of-small-molecules-present-in-ltemgtipomoea-mauritiana-ltemgtjacq](https://www.phytojournal.com/archives/2019.v8.i5.9904/swissadme-predictions-of-pharmacokinetics-and-drug-likeness-properties-of-small-molecules-present-in-ltemgtipomoea-mauritiana-ltemgtjacq).

5. Sanghvi T, Jain N, Yang G, Yalkowsky SH. 2003. Estimation of Aqueous Solubility By The General Solubility Equation (GSE) The Easy Way. *QSAR Comb Sci.* 22(2):258–262. [doi:10.1002/qsar.200390020](https://onlinelibrary.wiley.com/doi/10.1002/qsar.200390020).
