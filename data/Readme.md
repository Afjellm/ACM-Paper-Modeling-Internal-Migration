## District-Level Metadata Overview

This table describes the data created for the paper submitted for https://sigsim.acm.org/conf/pads/2026/ by Alexander
Jell.

| **Column Name**                 | **Description**                                                          | **Data Type** | **Additional Information**                                                                                              |
|---------------------------------|--------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------|
| **population_total**            | Total population at start of year                                        | `int`         | **Available years:** 2002–2024<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **population_within_age_group** | Total population at start of year for a given age group                  | `int`         | **Available years:** 2002–2024<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **distance**                    | Road distance in meters                                                  | `float`       | **Available years:** static<br>**Missing values:** no<br>**Source:** OpenStreetMap                                      |
| **unemp**                       | Unemployment rate                                                        | `float [0–1]` | **Available years:** 2002–2024<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **rental_price**                | Average rental price per m²                                              | `float`       | **Available years:** 2020–2024<br>**Missing values:** yes<br>**Source:** willhaben.at                                   |
| **gross_income**                | Average yearly gross income                                              | `float`       | **Available years:** 2016–2023<br>**Missing values:** no<br>**Source:** AMS                                             |
| **gdp**                         | Average yearly GDP per capita                                            | `float`       | **Available years:** 2000–2023<br>**Missing values:** no<br>**Source:** AMS                                             |
| **schools**                     | Number of schools                                                        | `int`         | **Available years:** static*<br>**Missing values:** no<br>**Source:** Statistik Austria                                 |
| **real_estate_price**           | Average price of real estate per m²                                      | `float`       | **Available years:** 2015–2022<br>**Missing values:** yes<br>**Source:** Statistik Austria                              |
| **land_price**                  | Average price of land per m²                                             | `float`       | **Available years:** 2015–2023<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **dcuc**                        | Car travel time to the next big city (minutes)                           | `float`       | **Available years:** static<br>**Missing values:** no<br>**Source:** OSM                                                |
| **dtuc**                        | Train travel time to the next big city (minutes)                         | `float`       | **Available years:** static<br>**Missing values:** no<br>**Source:** Google Maps                                        |
| **c_x**                         | Number of companies within size class x                                  | `int`         | **Available years:** 2011–2023<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **rural_level**                 | Classification of district as rural or urban                             | `string`      | **Available years:** static<br>**Missing values:** no<br>**Source:** Statistik Austria                                  |
| **u5km**                        | Universities within 5 km radius of district center                       | `int`         | **Available years:** static<br>**Missing values:** no<br>**Source:** oesterreich.gv.at                                  |
| **c5km**                        | Colleges within 5 km radius of district center                           | `int`         | **Available years:** static<br>**Missing values:** no<br>**Source:** oesterreich.gv.at                                  |
| **w_permit_x**                  | Number of permitted building constructions (by apartment size *x*)       | `int`         | **Available years:** 2010–2023<br>**Missing values:** no<br>**Source:** Statistik Austria                               |
| **green_ratio**                 | Share of attractive green spaces compared to the total district area     | `float [0-1]` | **Available years:** static<br>**Missing values:** no<br>**Source:** Corine Land Cover 2018                             |
| **rental_rate**                 | Share of people renting their home                                       | `float [0-1]` | **Available years:** 2021-2023<br>**Missing values:** no<br>**Source:** Statisik Austria                                |
| **flood_risk**                  | Population weighted share of flooding zone area                          | `float [0-1]` | **Available years:** static<br>**Missing values:** no<br>**Source:** Federal institute of Environment (Umweltbundesamt) |
| **download_speed**              | Population weighted average download speed of the best internet provider | `int`         | **Available years:** 2010-2023<br/>**Missing values:** no<br>**Source:** data.gv.at                                     |
| **is_adjacent**                 | Adjacency of 2 districts based on official borders                       | `bool`        | **Available years:** static<br>**Missing values:** no<br>**Source:** Statisik Austria                                   |
| **is_covid_year**               | Set to 1 for years 2020-2022                                             | `bool`        | **Available years:** static<br>**Missing values:** no<br>**Source:** Statisik Austria                                   |
 
## Data Interpolation

Some of the data is not complete from the source (missing values). To still feed full dataframes into the models the following values have been interpolated using simple interpolation techniques:

### Rental rates

Rental rates are derived from Statistik Austria (table: *“Housing Census – Persons with attributes of the buildings and dwellings – Time series from 2011”*).

For each district, the rental rate is calculated as:  
> number of people living in rented dwellings ÷ total population  

The dataset provides values only for 2011, 2021, 2022, and 2023. Missing years (2012–2020) are estimated using linear interpolation between 2011 and 2021:  

$$
\hat{q}_t = q_{2011} + \frac{q_{2021} - q_{2011}}{2021 - 2011} \cdot (t - 2011)
$$

The resulting data shows a strong urban–rural gradient, with rental rates up to ~80% in cities and around ~8% in rural districts.

### Real estate prices

Real estate price data (2015–2023) is sourced from Statistik Austria (.ods dataset). Apartment prices are used due to their lower share of missing values (~9%).

Missing apartment prices are imputed using house price trends. Assuming a proportional relationship between house and apartment prices, missing values are estimated using adjacent years:

Backward fill:
$$
\hat{p}_{apartment,t} = p_{apartment,t+1} \cdot \frac{p_{house,t}}{p_{house,t+1}}
$$

Forward fill:
$$
\hat{p}_{apartment,t} = p_{apartment,t-1} \cdot \frac{p_{house,t}}{p_{house,t-1}}
$$

### Rental prices

The data is missing values for the districts Rust, Bludenz and Reutte. The values for
these districts were interpolated using averages from the neighboring districts. Also, the
data starts at the year 2020. For model fitting and training, data from the year 2020 has
been used for all previous years. This can be improved as soon as more data is available