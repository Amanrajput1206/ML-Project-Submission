## Dataset Description

## Background

**VehicleVision: Robust Vehicle Classification** (from the Kaggle Overview page)

Can you build a machine learning model that can reliably identify a vehicle type from its specifications and performance measurements?

You are given vehicle measurements such as engine displacement, horsepower, vehicle weight, wheelbase, fuel efficiency, cargo capacity, acceleration, braking distance, tire width, and aerodynamic characteristics. The task is to classify each vehicle into one of three categories:

| Value | Vehicle Type |
|---|---|
| 0 | Sedan |
| 1 | SUV |
| 2 | Sports Car |

**The challenge** (verbatim points from Kaggle):

- The training data may contain data-quality issues such as duplicate vehicle measurements.
- `vehicle_id` identifies a record, but it should not be treated as a predictive vehicle characteristic.
- **The hidden evaluation data introduces measurement uncertainty.** The final evaluation samples are collected under a different measurement environment, where the recorded vehicle characteristics may contain additional random measurement error. Therefore, a model that performs well on the training data is not necessarily the model that will perform best on the final evaluation.
- This is a synthetic educational dataset: realistic names and ranges, but the records do not represent actual vehicles.

**Evaluation:** classification **accuracy** (fraction of predictions that exactly match the hidden classes).

**Submission:** `vehicle_id,vehicle_class`, one predicted class per test row.

## Data Description

The dataset folder contains the following files:

- **train.csv:** 10000 x 22
- **test.csv:** 2500 x 21
- **sample_submission.csv:** 2500 x 2 (`vehicle_id,vehicle_class`)

Files were renamed from the downloaded names (`vehicle_train_10000.csv`, `vehicle_test.csv`, `vehicle_sample_submission.csv`) to match `dataset/regression/`. The Kaggle page calls them `vehicle_train.csv`, `vehicle_public_test.csv`, `vehicle_sample_submission.csv`.

### Columns

| Column Name                    | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `vehicle_id`                   | Unique identifier (train 1–10000, test 10001–12500). |
| `engine_displacement_l`        | Engine displacement in litres.                    |
| `horsepower_hp`                | Engine power in horsepower.                       |
| `vehicle_weight_kg`            | Vehicle weight in kg.                             |
| `wheelbase_mm`                 | Wheelbase in mm.                                  |
| `vehicle_length_mm`            | Overall length in mm.                             |
| `vehicle_width_mm`             | Overall width in mm.                              |
| `vehicle_height_mm`            | Overall height in mm.                             |
| `fuel_tank_capacity_l`         | Fuel tank capacity in litres.                     |
| `city_fuel_efficiency_kmpl`    | City fuel efficiency in km/l.                     |
| `highway_fuel_efficiency_kmpl` | Highway fuel efficiency in km/l.                  |
| `engine_cylinders`             | Number of engine cylinders (integer, 3–10).       |
| `torque_nm`                    | Engine torque in Nm.                              |
| `ground_clearance_mm`          | Ground clearance in mm.                           |
| `turning_radius_m`             | Turning radius in metres.                         |
| `cargo_capacity_l`             | Cargo capacity in litres.                         |
| `acceleration_0_100_kmh_s`     | 0–100 km/h acceleration time in seconds.          |
| `braking_distance_m`           | Braking distance in metres.                       |
| `tire_width_mm`                | Tyre width in mm.                                 |
| `drag_coefficient`             | Aerodynamic drag coefficient.                     |
| `battery_capacity_kwh`         | Battery capacity in kWh.                          |
| `vehicle_class`                | **Target.** 0 = Sedan, 1 = SUV, 2 = Sports Car (train only). |
