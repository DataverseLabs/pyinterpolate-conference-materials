# The Deconvolution of the Aggregated Data into the Fine-Scale Blocks with Pyinterpolate

## Introduction

Do you need high-resolution data for your machine learning, but you have only areal aggregates? Would you like to present continuous maps instead of choropleth maps? We can transform county-level data into smaller blocks with Pyinterpolate. We will learn how to perform Poisson Kriging on the areal dataset during workshops.

## Rationale - why should I use pyinterpolate?

Choropleth maps representing areal aggregates are standard in the social sciences. We aggregate data over areas for administrative purposes and protect citizens' privacy. Unfortunately, those aggregated datasets can be misleading:

- Administrative units, especially in Europe, vary significantly in shape and size,
- Large units tend to be visually more important than smaller areas,
- It is hard to integrate areal data into machine learning pipelines with data at a smaller and regular scale.

There is a solution for the processes that are spatially correlated and represent rates. One example is the disease incidence rate map. An incidence rate is the number of disease cases per area divided by the total population in this area and multiplied by the constant number of 100,000. Through the denominator (total population), we can divide our space into smaller blocks – in this case, the population blocks. Then we regularize the semivariogram of areal data with the population density semivariogram to obtain a final model that considers fine-scale population blocks and can predict disease rates at a smaller scale. After this transformation, we can:

- show a continuous map of disease rates,
- avoid problems with the visual discrepancy between different areas' sizes,
- use data with better spatial resolution as an input for machine learning pipelines; for example, we can merge data with the remotely sensed information.

We will learn how to transform areal aggregates into smaller blocks during workshops. We will use the Pyinterpole package. We will discuss the most dangerous modeling pitfalls and what can be done with the output data. If you are an expert in the economy, social sciences, public health, or similar fields, this workshop is for you.

Pyinterpolate is a Python package for spatial interpolation. It is available here: https://pypi.org/project/pyinterpolate/

What can be achieved with pyinterpolate? Look at this map:

![Transformation of choropleth map into point-support model](data/fig1_example.png  "Transformation of choropleth map into point-support model")

## Setup

We will work in `Google Colab` (GC) environment. Setup in GC is not different than a setup in Linux or MacOS systems but we'll do it step-by-step to be sure that everyone can follow.

First of all, **pyinterpolate** in a stable version (*0.2.5-post1*) requires Python in version `3.7.x`. It could be a tricky to configure it locally, but if you use `conda` or `pipenv` it shouldn't be a problem. GC has Python in version `3.7.13` by default, so we are ready to go.

As a sanity check, you may write in GC:

```shell
!python3 --version
```

and expected output is:

```shell
>>> Python 3.7.13
```

> For the **future**, you may install different versions of Python in your Google Colab. Fortunately for us, the future stable release of **pyinterpolate** will work with Python 3.7, Python 3.8 and Python 3.9, so probably there won't be any problem with the installation.

It doesn't matter if you use your local environment or GC, your OS must have `libspatialindex.so` package installed. In GC you should type every time when fresh run starts:

```shell
!sudo apt install libspatialindex-dev
```

Now we are sure, that our system is configured properly. We can install the package!

```shell
!pip install pyinterpolate
```

> **Pyinterpolate** depends on the packages *matplotlib* and *numpy* - you may be forced to restart kernel after the installation in GC because those packages were included in the base setup of your environment. Just click `RESTART RUNTIME` button and everything should work fine! In local environment it shouldn't be a case.

We are ready to go!

## Datasets

We will use two different datasets:

1. `meuse` dataset [1] used for point kriging of zinc concentrations.
2. `Breast Cancer Rates in New England` [2] used for block deconvolution, `Population point-support data` [3]. Analysis based on [4].

## How Kriging works - practical PoV

Kriging is an interpolation method and it allows us to find values at unseen and unsampled locations. The technique is based on the semivariance (or covariance) model of spatial process. It is similar to the Inverse Distance Weighting algorithm in a way that the unknown value is a weighted average of known observations, but in contrary to IDW, Kriging weights are different at different distances. Moreover, Kriging has two important properties:

- it generates unbiased predictions, in practice if we "predict" values of known points we should get the exact value,
- it returns not only regressed value, but variance error too. It is a measure of interpolation uncertainty.

In practice, data analyst always follows a set of steps to make a Kriging prediction:

- create experimental variogram of a known points,
- based on the experimental variogram: model theoretical variogram model (only set of models is available for this, not every function can be applied at this step!),
- apply theoretical model to weights in the Kriging system.

**Pyinterpolate** simplifies those steps, but we shouldn't treat Kriging as an algorithm that can run without a supervision. The critical step, that should be done by human, is to choose valid semivariogram model (or just approve model that has been selected automatically).

Let's go through the initial example, where we will model zinc concentrations from the point measurements.

## The point Kriging example - zinc concentrations

### 1. Import Python packages

```python
import numpy as np
import pandas as pd
import geopandas as gpd

from pyinterpolate.semivariance import calculate_semivariance  # experimental semivariogram
from pyinterpolate.semivariance import TheoreticalSemivariogram  # theoretical models
from pyinterpolate.kriging import Krige
from shapely.geometry import Point

import matplotlib.pyplot as plt
```

- **numpy**: data transformations,
- **pandas**: read tabular data,
- **geopandas**: work with spatial tables,
- `calculate_semivariance`: get experimental semivariance model,
- `TheoreticalSemivariogram`: create theoretical variogram model,
- `Krige`: Kriging interpolator,
- `shapely.geometry.Point`: data transformations,
- `matplotlib.pyplot`: show intermediate steps and results.

### 2. Read data

```python
df = pd.read_csv('/content/drive/MyDrive/data_geopython_2022/meuse.csv')
```

### 3. Select columns

It can be done in the second step with `usecols` parameter of the `pandas.read_csv()` function.

```python
# Get only x, y and zinc concentration columns

df = df[['x', 'y', 'zinc']]
```

### 4. Transform data

Take **log** of the interpolated zinc concentrations.

```python
# Transform zinc column

df['zinc'] = np.log(df['zinc'])
```

> Why we did it? Kriging doesn't work well with highly skewed data. Log-transform may be required.

### 5. Calculate experimental variogram

For this step we must set two parameters:

- `step_size`: to control how many lags has our variogram (we should smooth variability but preserve a general trend, examples below).
- `max_range`: close points tend to be correlated, but at a long distance it is rarely a case! Usually we should assume, that the max range of spatial correlation is about a half of a study extent. This is a good practice from the computational perspective because in the mist cases the weights at larger distances are too small to have any effect on a predicted value.

**Wrong**: here we model experimental variogram with a very big step, and maximum range that exceeds the study extent:

```python
step_size = 3000
max_r = 160_000
variogram = calculate_semivariance(df.values, step_size=step, max_range=max_r)

# Plot experimental semivariogram

plt.figure(figsize=(12, 6))
plt.plot(variogram[:, 0], variogram[:, 1], '--o')
plt.title('Semivariogram')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.show()
```

> Why we did it? A common error is when we set a wrong range and step size because we didn't check coordinate reference system. For example, data has geographical coordinates but we expect it to be in a metric system.

**Wrong**: too small step size.

```python
step = 20
max_r = 1600
variogram = calculate_semivariance(df.values, step_size=step, max_range=max_r)

# Plot experimental semivariogram

plt.figure(figsize=(12, 6))
plt.plot(variogram[:, 0], variogram[:, 1], '--o')
plt.title('Semivariogram')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.show()
```

> Why we did it? To show that too small step generates too much noise. We are interested in the smooth trend, and not a small-scale variability. However, this variogram is much better than the previous one. But we can do it better!

**Correct**:

```python
step = 100
max_r = 1600
variogram = calculate_semivariance(df.values, step_size=step, max_range=max_r)

# Plot experimental semivariogram

plt.figure(figsize=(12, 6))
plt.plot(variogram[:, 0], variogram[:, 1], '--o')
plt.title('Semivariogram')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.show()
```

The correct example allows us to model Theoretical Variogram.

### 6. Fit theoretical model into experimental variogram

This is the most important step of analysis. We may think of of it as of the model training in a classic Machine Learning approach and further interpolation results are directly related to the semivariogram that we choose and its parameters.

Here is a sample experimental variogram and fitted theoretical model:

![Experimental and theoretical variograms](data/fig2.png  "Comparison of experimental variogram and theoretical model")

Our role is to find:

- **nugget** (or bias) at a distance 0 (point with itself). Usually it is set to 0, but for some cases it has a value > 0.
- **sill** - the value of dissimilarity where points stop to affecting each other,
- **range** - the distance when variogram reaches its sill.
- **model** - theoretical function that utilizes **nugget**, **sill** and **range** in some way. In practice, there is a limited set of functions that can be applied to semivariogram modeling and they are  known as conditional negative semi-definite functions. In practice, we cannot simply use any function to describe variogram because it could lead to negative variances at some distances.

Models available in version 0.2.5.post-1:

- circular,
- cubic,
- exponential,
- gaussian,
- linear,
- power,
- spherical.

The comparison between models is presented in the tutorial given in supplementary materials [S1].

We will use `.find_optimal_model()` method to allow algorithm to choose for us the best **range**, **sill** and **model** (**nugget** is set to 0 at a distance 0). We pass only one parameter: `number_of_ranges` that is the number of different distances tested for the lowest possible model error.

First, we initialize model:

```python
number_of_rngs = 64


theo_pyint = TheoreticalSemivariogram(
    points_array=df.values,
    empirical_semivariance=variogram
    )
```

Then, we allow algorithm to search the best range and model for our experimental data:

```python
opt_pyint = theo_pyint.find_optimal_model(
    number_of_ranges=number_of_rngs,
    number_of_sill_ranges=number_of_rngs
    )

print(f'Optimal model is {opt_pyint}')
```

```shell
>>> Optimal model is spherical
```

We can plot semivariogram with `.show_semivariogram()` method:

```python
theo_pyint.show_semivariogram()
```

We won't spend here a lot of time during the workshops, but you should be aware, that the next stable release (0.3.0) will have a great number of methods to control semivariogram modeling, that is the core operation in Kriging.

### 7. Build and test Kriging model

Kriging is not a single technique and there are multiple methods to *Krige*. The most popular is *Ordinary Kriging*, but **pyinterpolate** has also *Simple Kriging* and *Poisson Kriging* techniques included.

We use Ordinary Kriging, if you want to learn more, feel free to check supplementary materials [S2].

`Krige` object takes two parameters: semivariogram model and known points. We can check if our Kriging model works as it should with a simple test. We use one known point coordinates and predict value at this place. If it's equal to the training value then our model works fine!

```python
# Initialize model

model = Krige(semivariogram_model=theo_pyint, known_points=df.values)

# Get one sample for test
unknown = df.sample().values[0]
unknown_val = unknown[-1]
unknown_loc = unknown[:-1]

# Set Kriging parameters
nn = 32
nmin = 8

# Make a prediction
ok_pred = model.ordinary_kriging(unknown_location=unknown_loc,
                                 max_no_neighbors=nn)
                                 
# Compare predicted value to real observation
print(np.equal(ok_pred[0], unknown_val))
```

```shell
True
```

### 8. Make predictions

If our test returned `True` then we are free to interpolate values at unseen locations! To do so, we must pass unknown points into a model and store predictions and prediction error.

```python
# Read point grid

data = pd.read_csv('/content/drive/MyDrive/data_geopython_2022/meuse_grid.csv')
data = data[['x', 'y']].values

# Set output array and a Krige object

interpolated_results = []

model_pyint = Krige(semivariogram_model=theo_pyint, known_points=df.values)
```

Now we can make a predictions!

```python
for pt in data:
    try:
        pred, err = model_pyint.ordinary_kriging(unknown_location=pt,
                                                 max_no_neighbors=nn,
                                                 min_no_neighbors=nmin)[:2]
    except ValueError:
        pred, err = np.nan, np.nan
        
    interpolated_results.append([pt[0], pt[1], pred, err])
    
predictions = gpd.GeoDataFrame(interpolated_results,
                           columns=['x', 'y', 'pred', 'err'])

```

Voila! We have an array of predictions and an array of prediction errors. Now, we can transform `x` and `y` coordinates to `Point` and plot results on a map!


```python
predictions['geometry'] = gpd.points_from_xy(predictions['x'], predictions['y'])
predictions.set_geometry('geometry', inplace=True)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 8))
predictions.plot(ax=axes[0], column='pred', legend=True, vmin=4.5, vmax=7.6)
predictions.plot(ax=axes[1], column='err', cmap='YlOrRd', legend=True, vmin=0, vmax=0.5)
plt.show()
```

![Ordinary Kriging Output](data/fig3.png  "Ordinary Kriging Output")

> Important: Kriging "superiority" over "normal" machine learning models is its explainability - we can describe spatial process in a term of nugget, sill, variance - and, even more important, the variance error map that shows uncertainty of interpolated results in a given areas. For some applications this explainability and uncertainty is crucial (public health, mining, weather forecasts).

# The block kriging example - deconvolution of choropleth map into small-scale point-support map

The process of block deconvolution is described in [4]. In practice, it allows us to transform block rates (or volumes) into a point-support map.

![Area-to-Point Poisson Kriging Output](data/fig4.png  "Area-to-Point Poisson Kriging Output")

**Pyinterpolate** has three algorithms used for block Kriging:

1. *Centroid-based Poisson Kriging*: used for areal interpolation and filtering. We assume that each block can collapse into its centroid. It is much faster than Area-to-Area and Area-to-Point Poisson Kriging but introduces bias related to the area's transformation into single points [S3].
2. *Area-to-Area Poisson Kriging*: used for areal interpolation and filtering. The point-support allows the algorithm to filter unreliable rates and makes final areal representation of rates smoother [S4].
3. *Area-to-Point Poisson Kriging*: where areal support is deconvoluted in regards to the point support. Output map has a spatial resolution of the point support while coherence of analysis is preserved (sum of rates is equal to the output of Area-to-Area Poisson Kriging). It is used for point-support interpolation and data filtering [S5].

During this workshops, we are going to explore *Area-to-Point* technique, that requires from us semivariogram deconvolution. As you see, semivariogram analysis and modeling is, again, a significant part of our job.

### 1. Import packages and modules

```Python
import numpy as np
import pandas as pd
import geopandas as gpd

from pyinterpolate.io_ops import prepare_areal_shapefile, get_points_within_area  # Prepare data
from pyinterpolate.semivariance import calculate_semivariance  # Experimental semivariogram
from pyinterpolate.semivariance import RegularizedSemivariogram  # Semivariogram regularization class
from pyinterpolate.semivariance import TheoreticalSemivariogram
from pyinterpolate.kriging import ArealKriging

import matplotlib.pyplot as plt
```

### 2. Set paths to datasets and the names of the required columns

Area-to-Point Kriging requires two data sources:

1. Polygons (blocks) with rates, usually presented as a choropleth map.
2. Point-support data, representing denominator of our rates. It could be a population (where we analyze number of disease cases per population), time (where we analyze number of endangered species counts per time spent on observations in a given region), or depth (number of grade counts per depth of the sampling hole).

Algorithm must know index and value column names of a block and value column name of a point-support.

```python
BLOCKS = '/content/drive/MyDrive/data_geopython_2022/cancer_data.shp'
POINT_SUPPORT = '/content/drive/MyDrive/data_geopython_2022/cancer_population_base.shp'

BLOCK_ID = 'FIPS'
BLOCK_VAL_COL = 'rate'
PS_VAL_COL = 'POP10'
```

### 3. Load blocks and point-support into module

Blocks data must be preprocessed, the same for the point-support. Both data sources are transformed into numpy arrays.

Block data structure is:

```
[area_id, area_geometry, centroid coordinate x, centroid coordinate y, value]
```

Point-support structure is:

```
[area_id, [[point_position_x, point_position_y, value], ...]]
```

Basically, we get block centroids and we group point-support within blocks for the modeling.

```python
areal_data_prepared = prepare_areal_shapefile(BLOCKS, BLOCK_ID, BLOCK_VAL_COL)

points_in_area = get_points_within_area(BLOCKS,
                                        POINT_SUPPORT,
                                        areal_id_col_name=BLOCK_ID,
                                        points_val_col_name=PS_VAL_COL)
```

### 4. Check block data experimental variogram

Before we start modeling, we should first check block variogram and a point-support variogram to be sure that our data has any spatial dependency (at both levels) and to find the best step size and maximum range for modeling.

First, we will check the experimental variogram of blocks - it is derived from block centroids.

```python
maximum_range = 300000
step_size = 40000

dt = areal_data_prepared[:, 2:]  # x, y, val
exp_semivar = calculate_semivariance(data=dt, step_size=step_size, max_range=maximum_range)

# Plot experimental semivariogram

plt.figure(figsize=(12, 7))
plt.plot(exp_semivar[:, 0], exp_semivar[:, 1])
plt.title('Experimental semivariogram od areal centroids')
plt.xlabel('Range - in meters')
plt.ylabel('Semivariance')
plt.show()
```

![Areal data experimental variogram](data/fig5.png  "Areal data experimental variogram")

It seems to be ok, so we can make a next step: let's check population units:

```python
def build_point_array(points):
    a = None

    for rec in points:
        if a is None:
            a = rec.copy()
        else:
            a = np.vstack((a, rec))

    return a

maximum_point_range = 300000
step_size_points = 10000


pt = build_point_array(points_in_area[:, 1])  # x, y, val
exp_semivar = calculate_semivariance(data=pt, step_size=step_size_points, max_range=maximum_point_range)

# Plot experimental semivariogram

plt.figure(figsize=(12, 7))
plt.plot(exp_semivar[:, 0], exp_semivar[:, 1])
plt.title('Experimental semivariogram od population data')
plt.xlabel('Range - in meters')
plt.ylabel('Semivariance')
plt.show()
```

![Point-support experimental variogram](data/fig6.png  "Point-support experimental variogram")

This variogram is fine too, but as you may have noticed, it's variance is orders of magnitude larger than semivariances of block data. Our role is to transform this variogram, and find the theoretical model, that will describe blocks process at a scale of a point-support.

### Fit model

At this point, we have blocks data, point-support, information about the step size and maximum range of blocks data. We can initilize `RegularizedSemivariogram()` model.

```python
reg_mod = RegularizedSemivariogram()
```

The first step is to fit initial variograms and check how big is an error between those. This process takes some time, because we perform multiple heavy-computing operations on a set of points:

- we calculate "inner variograms" of each blocks,
- we calculate variograms between blocks based on the point-support within a specific block.

The `.fit()` method takes 6 parameters:

1. **areal_data**: transformed block data,
2. **areal_step_size**: step size of experimental variogram of blocks data,
3. **max_areal_range**: a maximum range of of 

## Bibliography

[1] Pebesma, E. The meuse data set: a tutorial for the gstat R package. URL: [here](https://cran.r-project.org/web/packages/gstat/vignettes/gstat.pdf)

[2] National Cancer Institute. Incidence Rates Table: Breast Cancer: United States. URL: [here](https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=00&areatype=county&cancer=055&race=00&sex=2&age=001&stage=999&year=0&type=incd&sortVariableName=rate&sortOrder=default&output=0#results)

[3] United States Census Bureau. Centers of Population for the 2010 Census. URL: [here](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2010.html)

[4] Goovaerts, P. Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units. Mathematical Geosciences, volume 40, year 2007. DOI: 10.1007/s11004-007-9129-1

## Supplementary Materials

[S1] [Fit semivariogram](https://github.com/DataverseLabs/pyinterpolate/blob/main/tutorials/Semivariogram%20Estimation%20(Basic).ipynb
)

[S2] [Comparison of Ordinary and Simple Kriging](https://github.com/DataverseLabs/pyinterpolate/blob/main/tutorials/Ordinary%20and%20Simple%20Kriging%20(Basic).ipynb)

[S3] [Poisson Kriging - centroid based approach - tutorial](https://github.com/DataverseLabs/pyinterpolate/blob/main/tutorials/Poisson%20Kriging%20-%20Centroid%20Based%20(Advanced).ipynb)

[S4] [Poisson Kriging - Area to Area Kriging](https://github.com/DataverseLabs/pyinterpolate/blob/main/tutorials/Poisson%20Kriging%20-%20Area%20to%20Area%20(Advanced).ipynb)

[S5] [Poisson Kriging - Area to Point Kriging](https://github.com/DataverseLabs/pyinterpolate/blob/main/tutorials/Poisson%20Kriging%20-%20Area%20to%20Point%20(Advanced).ipynb)





















