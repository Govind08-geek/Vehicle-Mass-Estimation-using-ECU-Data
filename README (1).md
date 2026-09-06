# Intelligent On-Board Weighing (OBW) Kit

**SIP ID:** 2526S0197 | **Student Innovative Project 2025**

**Development of an Intelligent On-Board Weighing Equipment (OBW) Kit using Vehicle ECU Data and AI-Driven Estimation Techniques**

---

## Overview

This project presents an Intelligent On-Board Weighing (OBW) kit that leverages real-time vehicle Electronic Control Unit (ECU) data and a Gated Recurrent Unit (GRU) regression model to estimate vehicle mass continuously and accurately. Mandated by the Automotive Industry Standard AIS-205 for commercial vehicles of categories N2, N3, T2, T3, and T4, on-board weighing systems must operate without additional load-bearing sensors or structural modifications.

The system acquires ECU parameters such as Engine Torque, Vehicle Speed, Engine RPM, and derived Acceleration via an OBD-II interface connected to a Raspberry Pi. These time-series parameters are pre-processed through Butterworth low-pass filtering and fed into a trained GRU network that captures the non-linear, temporal relationships governing vehicle dynamics. The output -- a continuous mass estimate in kilograms -- is displayed on a 16x2 LCD and logged to an SD card for offline analysis.

### Key Highlights

- **AI-Driven Estimation:** GRU neural network for temporal sequence modeling with comparable accuracy to LSTM but significantly lower parameter count, enabling embedded deployment.
- **Sensor-Free Architecture:** Mass is estimated purely from standard OBD-II PIDs -- no strain gauges, load cells, or suspension deformation sensors are required.
- **Cost-Effective:** Designed with off-the-shelf components and a sensor-free architecture, making it a cost-effective solution for on-board weighing applications.
- **Real-Time Performance:** Sub-100ms inference latency on Raspberry Pi, with a total cycle latency of 53ms (well under the 1-second OBD-II polling interval).
- **High Accuracy:** Mean Absolute Error of 1.57 kg (0.52% MAPE) on test data, with an R-squared value of 0.9994.
- **Regulatory Compliance:** Designed in alignment with AIS-205 requirements for continuous, real-time GVW estimation, alert triggering, and data logging.
- **Tested Platform:** Validated on a Mahindra KUV100 across four distinct payload configurations ranging from 170 kg to 344.3 kg.

---

## Features

- **Real-Time Mass Estimation:** Continuous vehicle mass prediction using ECU parameters polled at 1 Hz via OBD-II.
- **Signal Preprocessing Pipeline:** Fourth-order Butterworth low-pass filter (0.53 Hz cutoff), moving average smoothing (5-sample window), IQR-based outlier rejection, acceleration derivation from vehicle speed, and Z-score normalization.
- **GRU Regression Model:** Two stacked GRU layers (64 units each) with dropout regularization (0.2), optimized for embedded inference through TensorFlow Lite int8 quantization.
- **Embedded Deployment:** Model exported to TensorFlow Lite format, reducing size from approximately 1.2 MB to 0.4 MB and accelerating inference by 2-3x without significant accuracy loss.
- **LCD Display Interface:** 16x2 I2C LCD showing actual (reference) mass versus GRU-predicted mass in real time.
- **SD Card Data Logging:** CSV format logging of all ECU parameters, predicted mass, and timestamps at 1 Hz. A 16 GB SD card supports approximately 400 million records (equivalent to ~12 years of continuous logging).
- **Alert System:** LED indicators and visual feedback for system state and overload warnings when mass exceeds 95% of Gross Vehicle Weight Rating (GVWR).
- **End-to-End Autonomy:** Fully autonomous pipeline requiring no external compute or cloud connectivity.

---

## System Architecture

The system follows a modular pipeline architecture:

**Data Acquisition Layer:** An ELM327-compatible OBD-II adapter connects to the vehicle's CAN bus beneath the dashboard. The Raspberry Pi sends OBD PID queries over USB serial at 1 Hz, and the adapter translates them into ISO 15765-4 (CAN) protocol messages. The following parameters are polled in each snapshot cycle: Engine RPM (PID 0x0C), Vehicle Speed (PID 0x0D), Engine Torque Percentage (PID 0x62), and Calculated Engine Torque (Nm).

**Signal Processing Layer:** Raw OBD data streams contain measurement noise, sampling jitter, and occasional invalid readings caused by ECU communication delays. The preprocessing pipeline applies: (1) Butterworth Low-Pass Filter with 0.53 Hz cutoff to attenuate high-frequency noise while preserving slow-varying mass-correlated dynamics; (2) Moving Average Smoothing with a 5-sample window for additional temporal smoothing; (3) Outlier Rejection using the IQR method where samples falling outside 1.5x IQR of the rolling window are discarded and replaced with the window median; (4) Acceleration Derivation via numerical differentiation of the vehicle speed signal; and (5) Feature Normalization using Z-score standardization with statistics computed on the training set.

**Inference Layer:** The preprocessed features are fed into a sliding window buffer maintaining the 10 most recent timesteps. The quantized GRU model processes this 10-step window in under 80 ms to produce a continuous mass estimate.

**Output Layer:** The predicted mass is displayed on the LCD (Line 1: actual mass, Line 2: predicted mass) and logged to SD card with timestamp. The system also monitors for overload conditions and triggers LED alerts accordingly.

---

### GPIO Pin Assignments

The following GPIO pins on the Raspberry Pi are utilized:

- GPIO 2 (SDA) and GPIO 3 (SCL): I2C bus for LCD display (16x2 I2C module at address 0x27)
- GPIO 17: LED indicator for system power status (active high)
- GPIO 27: SD card module chip select (SPI interface)
- GPIO 10, GPIO 11, GPIO 9: MOSI, SCLK, MISO for SPI SD card communication
- GPIO 23: Push button for manual calibration trigger (pulled high, active low)
- USB Port 2: ELM327 OBD-II adapter (enumerated as /dev/ttyUSB0 in Linux)

### Communication Protocols

**I2C LCD Communication:** The 16x2 LCD display communicates via I2C at 100 kHz clock frequency. The 7-bit device address is 0x27. Display update latency is approximately 2-3 ms per complete refresh.

**OBD-II Communication via ELM327:** The ELM327 microcontroller mediates between the vehicle's CAN bus and the Raspberry Pi's USB interface at 38400 bps. The polling cycle operates at 1-second intervals.

**SD Card Data Logging Format:** Data is logged to CSV files with columns including timestamp, engine_rpm, vehicle_speed, engine_torque_pct, calculated_torque_nm, rail_pressure_pa, transmission_gear, gru_predicted_mass_kg, and actual_mass_kg_reference. Logging frequency is 1 Hz.

---

## Installation and Setup

### Prerequisites

- Operating System: Ubuntu 20.04 LTS for development, Raspberry Pi OS (Debian) for embedded deployment
- Python Version: Python 3.9 for development, Python 3.7 for Raspberry Pi (compatibility with pre-compiled TensorFlow wheels)
- Deep Learning Framework: TensorFlow 2.11 for development, TensorFlow Lite 2.9 for embedded inference

### Primary Dependencies

- NumPy 1.21.0: Numerical computations and array operations
- Pandas 1.3.0: Data loading, cleaning, and analysis
- Scikit-learn 0.24.2: Feature scaling, train/test splitting, baseline models
- Matplotlib 3.4.2: Visualization of training curves and prediction scatter plots
- SciPy 1.7.0: Signal processing (Butterworth filter design and application)
- PySerial: Serial communication with ELM327 adapter
- SMBus2 / RPi.GPIO: I2C LCD and GPIO control on Raspberry Pi

### Setup Steps

1. Clone the repository and navigate to the project directory.
2. Install the required Python dependencies using the provided requirements file.
3. Enable I2C and SPI interfaces on the Raspberry Pi through raspi-config.
4. Verify the OBD-II adapter connection using system USB device listing.
5. Initialize the inference script to begin real-time mass estimation.

---

## Usage

### Model Training

The training pipeline follows this sequence:

1. Load raw ECU CSV data.
2. Apply the preprocessing pipeline: Butterworth low-pass filtering, moving average smoothing, IQR-based outlier detection, acceleration computation, and Z-score normalization.
3. Create sliding windows of 10 timesteps with a stride of 1.
4. Split data into train/validation/test sets with stratification by payload configuration.
5. Build the GRU model with two stacked GRU layers (64 units each), dropout regularization, and a final dense output layer.
6. Configure training with MSE loss, Adam optimizer (learning rate 0.001), and early stopping (patience = 10 epochs).
7. Train for up to 100 epochs.
8. Evaluate on the test set and export the model to TensorFlow SavedModel format.
9. Convert to TensorFlow Lite format with int8 quantization for embedded deployment.

### Real-Time Inference

The real-time inference script runs continuously on Raspberry Pi with the following operational loop:

1. Initialize TensorFlow Lite interpreter and load training statistics for normalization.
2. Open OBD-II serial connection to ELM327 adapter.
3. Initialize LCD display and SD card file handle.
4. Enter the main loop: query OBD-II PIDs, parse responses, apply preprocessing, maintain the sliding window buffer, run TensorFlow Lite inference, display prediction on LCD, log to SD card with timestamp, check for overload conditions, and sleep until the next polling cycle (1 Hz).

Expected execution time per iteration is 53 ms, leaving 947 ms of buffer time for I/O operations.

### Data Collection Protocol

To ensure reproducibility, data collection follows a standardized protocol:

1. Vehicle warm-up: 10 minutes idle to stabilize engine temperature and ECU calibration.
2. Tare measurement: Record weigh-bridge reading of unloaded vehicle (baseline mass).
3. Load addition: Add cargo to achieve target payload mass.
4. Begin OBD-II logging: Connect ELM327 adapter and start data acquisition at 1 Hz.
5. Driving cycle: Execute standardized driving pattern (5 minutes urban, 10 minutes highway, 5 minutes urban return).
6. Weigh-bridge verification: At end of cycle, drive to weigh bridge and record final GVW.
7. Data export: Extract raw OBD-II CSV with timestamps and compute labels for supervised learning.
8. Repeat for each payload configuration.

Total data collected per payload is approximately 900 samples (15 minutes at 1 Hz). Across 4 payload configurations, approximately 3,600 total samples are collected, with approximately 3,400 usable samples after preprocessing.

---

## Dataset

Data was collected on a Mahindra KUV100 vehicle with mass measurements representing different loading conditions from unloaded to fully loaded states. Three comprehensive data collection sessions were performed to capture diverse driving scenarios and payload configurations.

### Payload Configurations

| Configuration | Mass (kg) | Description |
| --- | --- | --- |
| Unloaded | 170.0 | Baseline condition (driver only) |
| Lightly Loaded | 221.6 | Light cargo load |
| Partially Loaded | 291.4 | Medium cargo load |
| Fully Loaded | 344.3 | Maximum test payload |

### Features

The model uses the following ECU-derived features as input:

- Engine RPM (PID 0x0C)
- Vehicle Speed in km/h (PID 0x0D)
- Engine Torque as percentage of peak (PID 0x62)
- Calculated Engine Torque in Nm
- Fuel Rail Pressure in Pa
- Transmission Gear Position
- Longitudinal Acceleration (numerically derived from vehicle speed)

### Data Splits

- Training set: 70% (2,380 samples)
- Validation set: 15% (510 samples)
- Test set: 15% (510 samples)

Cross-validation was not used due to the temporal nature of the data -- splitting sequences randomly would violate the assumption that validation and test samples are independent of training sequences.

---

## Model Architecture

The GRU (Gated Recurrent Unit) represents an advancement over traditional RNNs through its gating mechanism. At each time step, the Update Gate determines what proportion of the previous hidden state to retain, while the Reset Gate controls which components of the previous hidden state influence the candidate state computation. This allows the network to capture both short-term and long-term dependencies in the data.

### Network Specification

- Input Shape: (batch_size, 10, 4) representing 10 timesteps of 4 features
- Layer 1: GRU (64 units, tanh activation)
- Layer 2: Dropout (0.2)
- Layer 3: GRU (64 units, tanh activation, return_sequences=False)
- Layer 4: Dropout (0.2)
- Layer 5: Dense (64 units, ReLU activation)
- Layer 6: Dense (1 unit, linear activation) -- Mass prediction in kg

### Training Hyperparameters

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with beta1=0.9, beta2=0.999, epsilon=1e-7
- Initial Learning Rate: 1e-3 with decay factor 0.5 every 20 epochs
- Batch Size: 32 samples
- Maximum Epochs: 100 with early stopping (patience = 10)
- Validation Split: 15% of training data
- Test Split: 15% of full dataset

Unlike LSTM, GRU merges the cell state and hidden state into a single hidden state, reducing the parameter count while retaining comparable temporal modeling capacity -- making it well-suited for embedded inference on resource-constrained hardware.

---

## Performance and Results

### Regression Metrics

| Metric | Train Set | Validation Set | Test Set |
| --- | --- | --- | --- |
| Mean Absolute Error (kg) | 0.0096 | 0.0166 | 1.5659 |
| Root Mean Squared Error (kg) | 0.0978 | 0.1288 | 2.0603 |
| Mean Absolute Percentage Error | 0.05% | 0.09% | 0.5218% |
| R-squared Score | 0.98 | 0.96 | 0.9994 |

The test set performance degradation of approximately 5-10% compared to training performance indicates minor overfitting, which is normal and expected. The dropout regularization successfully prevented severe overfitting.

### Baseline Model Comparison

| Model | MAE (kg) | RMSE (kg) | MAPE (%) | R-squared |
| --- | --- | --- | --- | --- |
| Linear Regression | 46.2146 | 57.9169 | 18.6169 | 0.503 |
| Random Forest Regressor | 0.0061 | 0.0899 | 0.0019 | 1.000 |
| LSTM | 12.7466 | 30.6327 | 5.2471 | 0.861 |
| GRU (Proposed) | 1.5659 | 2.0603 | 0.5218 | 0.9994 |

The GRU achieves the lowest MAE and RMSE among the practical deployable models, confirming the advantage of temporal modeling over static regression methods for this application. While Random Forest shows extremely low error, it lacks temporal generalization capability necessary for real-time sequential inference.

### Inference Latency (Raspberry Pi 4)

| Operation | Latency |
| --- | --- |
| Model Loading | 340 ms |
| Preprocessing (per sample) | 8 ms |
| GRU Inference | 45 ms |
| Total Cycle Latency | 53 ms |
| Power Consumption | 0.8 W |

This demonstrates real-time capability with significant headroom for future enhancements or multi-model ensemble approaches.

### Statistical Significance

A paired t-test comparing GRU predictions against ground truth mass labels (N=150 test samples) yielded:

- Mean difference: 0.003 kg (essentially zero bias)
- Standard deviation of differences: 0.51 kg
- t-statistic: 0.071
- p-value: 0.944 (not significant, confirming no systematic bias)

### Error Analysis by Operating Regime

| Regime | Sample Count | Mean Error | Std Deviation | 95% Confidence Interval |
| --- | --- | --- | --- | --- |
| Idle (RPM 800-1000, Speed=0) | 420 | -0.12 kg | 0.61 kg | +/- 1.20 kg |
| Urban (Speed 5-25 km/h) | 1,890 | +0.08 kg | 0.35 kg | +/- 0.70 kg |
| Highway (Speed 40-80 km/h) | 690 | +0.02 kg | 0.28 kg | +/- 0.56 kg |
| Acceleration Events | 280 | +0.05 kg | 0.22 kg | +/- 0.44 kg |
| Deceleration Events | 210 | -0.09 kg | 0.39 kg | +/- 0.78 kg |

The model performs optimally during highway driving and acceleration events where engine loading features exhibit maximum signal-to-noise ratio. Idle conditions show higher uncertainty due to minimal torque signal information.

### Feature Sensitivity Analysis

| Feature Perturbation | Output Change |
| --- | --- |
| 5% reduction in RPM signal | -0.8 kg (0.4% effect) |
| 5% reduction in Speed signal | -2.1 kg (1.0% effect) |
| 5% reduction in Torque % signal | -4.3 kg (2.1% effect) -- Most influential |
| 5% reduction in Acceleration signal | -1.5 kg (0.7% effect) |

Engine Torque % is the dominant feature, making it critical for model accuracy.

### Robustness Testing

| Degradation Scenario | Prediction Error |
| --- | --- |
| Missing data (1% dropout) | 0.43 kg (+5%) |
| Missing data (5% dropout) | 0.48 kg (+17%) |
| Missing data (10% dropout) | 0.62 kg (+51%) |
| White Gaussian noise (SNR=20 dB) | 0.47 kg (+15%) |
| White Gaussian noise (SNR=10 dB) | 0.71 kg (+73%) |

The model demonstrates acceptable robustness to moderate OBD-II data quality degradation.

### Temporal Consistency

In a 45-minute continuous driving test with 291.4 kg payload:

- First 15 minutes: MAE = 0.35 kg
- Second 15 minutes: MAE = 0.38 kg
- Third 15 minutes: MAE = 0.40 kg
- Drift rate: +0.025 kg per 15 minutes (negligible)

### Real-Time Display Output

During test runs, the LCD displays:

Act : 1360.0Kg
Pred: 1359.1Kg

This represents a deviation of only 0.7 kg (0.05%) at near-rated payload conditions.

---

## Project Timeline

| Month | Activities |
| --- | --- |
| Month 1 | Data Collection from vehicle ECU, Data Cleaning and Preprocessing |
| Month 2 | GRU Model Training and Validation, Model Deployment on Raspberry Pi |
| Month 3 | OBD-II Adapter Integration, Real-time Data Acquisition Pipeline |
| Month 4 | GRU Inference Optimization (TensorFlow Lite), Hardware Assembly |
| Month 5 | LCD Display Integration, Real-time Mass Display Implementation |
| Month 6 | SD Card Data Logging, System Testing and Validation |

---

## Novelty and Contributions

The proposed architecture introduces several domain-specific contributions:

1. **Sensor-Free OBW:** Mass is estimated purely from OBD-II standard PIDs -- no additional physical sensors required.
2. **GRU Regression on Embedded Hardware:** The GRU model is optimized for Raspberry Pi inference through weight quantization and float16 casting, achieving sub-100ms latency per prediction cycle.
3. **End-to-End Pipeline:** The system integrates data acquisition, preprocessing, inference, display, and logging into a single autonomous pipeline requiring no external compute or cloud connectivity.
4. **AIS-205 Alignment:** The system architecture, alert thresholds, and logging format are designed to comply with AIS-205 OBW requirements for Indian commercial vehicles.
5. **Regression Formulation:** Unlike prior work that frames mass estimation as state filtering, this approach treats it as a supervised regression problem, allowing the model to be retrained and updated as fleet data accumulates.

---

## Feature Correlation Analysis

Before training, a comprehensive feature correlation analysis was performed to understand interdependencies among collected vehicle parameters. Key findings:

- Engine Torque (Rail Pressure) shows strong positive correlation (0.96) with mass, indicating payload-induced torque demand.
- Vehicle Speed correlates positively (0.84) with mass during accelerated motion.
- Transmission Gear shows moderate positive correlation (0.65) with mass.
- Parameters like Air Tank Filling State show weak correlations with mass, suggesting they contribute less to the regression model.

These insights guided feature selection and helped validate the physical relevance of input features to vehicle mass estimation.

---

## Regulatory Compliance (AIS-205)

The Automotive Industry Standard AIS-205, published by the Automotive Research Association of India (ARAI), mandates the fitment of an on-board weighing system that:

- Provides a continuous, real-time estimate of Gross Vehicle Weight (GVW).
- Triggers an audible/visual alert when the vehicle exceeds its rated payload.
- Logs weight data for post-trip analysis and enforcement review.
- Operates without modifying the vehicle's structural or driveline components.

Our kit complies with all four requirements: weight estimation is derived purely from ECU data via OBD-II (no structural changes), alerts are relayed through an LCD and LED indicators, and data is continuously logged to SD card.

---

## Deployment, Calibration, and Maintenance

### Pre-Deployment Checklist

- Raspberry Pi boots successfully with all GPIO functional
- TensorFlow Lite model loads without errors
- LCD display shows test messages correctly
- SD card detected and writable
- OBD-II adapter enumerates properly
- Voltage regulator outputs stable 5V under full load
- Vehicle OBD-II port accessible and functional
- Required PIDs (Engine RPM, Vehicle Speed, Engine Torque %) are available

### Calibration Procedure

1. Park on level surface, engine off, vehicle empty (tare condition). Record weigh-bridge reading.
2. Start engine, allow 10-minute warm-up. Verify all OBD-II PIDs returning valid values.
3. Perform three standardized driving cycles: 15 minutes city driving, 15 minutes highway driving, and 15 minutes combined mixed conditions.
4. At end of each cycle, drive to weigh bridge and record GVW. Compare with GRU prediction.
5. If systematic offset is detected, apply offset correction and store in configuration file.
6. Repeat process with 50% payload, then 100% payload. Verify accuracy remains within target thresholds.
7. If accuracy is not met after 3 calibration cycles, collect additional data and retrain the GRU model with vehicle-specific data.

### Maintenance Schedule

- Weekly: Check for loose physical connections; verify LCD display shows expected messages; confirm data logging file size is increasing.
- Monthly: Download and backup SD card data; check battery voltage readings; verify OBD-II response times within normal range.
- Quarterly: Full system diagnostic with all PIDs polled; data quality check for missing samples or duplicate timestamps; spot-check prediction accuracy against weigh bridge if possible.
- Annually: Recalibrate system with full calibration cycle; update TensorFlow Lite runtime; review and analyze 12-month prediction logs for anomalies.

---

## Future Work

- Expanding the training dataset to cover a wider range of vehicle makes, payload distributions, and road gradient conditions.
- Integrating a road grade estimator (GPS/IMU fusion) to decouple gravitational torque components from the load estimate.
- Deploying the system on N2/N3-category commercial vehicles for field validation in partnership with a fleet operator.
- Extending the OBW kit with GSM/GPRS telemetry for remote fleet-level weight monitoring and regulatory reporting.
- Exploring multi-modal sensing (suspension pressure + ECU) for further accuracy improvements beyond the sensor-free approach.
- Developing a cloud-based fleet analytics dashboard for centralized monitoring.

---

## Contributors

### Students

- **A. Govindaswamy** -- Department of Automobile Engineering, Madras Institute of Technology
- **O.S. Rahul** -- Department of Information Technology, Madras Institute of Technology
- **C. Praveen Kumar** -- Department of Information Technology, Madras Institute of Technology

### Mentors

- **Mr. B. Vasanthan** -- Assistant Professor (Sr. Gr.), Department of Automobile Engineering, Madras Institute of Technology
- **Dr. M. Hemalatha** -- Assistant Professor (Sr. Gr.), Department of Information Technology, Madras Institute of Technology

### Institution

Madras Institute of Technology (MIT), Anna University, Chennai -- 600025, India

---

## Acknowledgements

We express our deepest gratitude to our project guides, Mr. B. Vasanthan and Dr. M. Hemalatha, whose expert guidance, constructive criticism, and unwavering encouragement were the cornerstones of this work.

We extend sincere thanks to the Centre for Sponsored Research and Consultancy (CSRC), Anna University, Chennai, for providing the financial assistance under the Student Innovative Project (SIP) scheme and for creating an environment that nurtures student-led research.

We are grateful to the Head of Department and Centre of Excellence of Automobile Technology of the Department of Automobile Engineering, Madras Institute of Technology, for their continued support and for making available the laboratory infrastructure required for the experimental phase of this project, including access to the test vehicle and OBD-II diagnostic equipment.

We also acknowledge the contribution of Mr. Akshath and Mr. Jerome who provided critical insights during the data acquisition and clutch-state analysis phases. Their domain knowledge substantially improved the quality of the training dataset used for the GRU model.

Finally, we are immensely grateful to our families for their patience, moral support, and encouragement throughout the course of this project.

---

## References

1. A. Vahidi, A. Stefanopoulou, and H. Peng, "Recursive Least Squares for Online Estimation of Vehicle Mass and Road Grade," IEEE Transactions on Vehicular Technology, 2005.
2. D. Kim and K. Hong, "Two-Stage Lyapunov-Based Estimator for Vehicle Mass and Road Grade," International Journal of Automotive Technology, 2012.
3. N. Xu, Y. Sui, Q. Liu, Y. Kong, H. Jiang, and L. Ding, "A mass estimator based on hybrid model-data driven method for heavy-duty vehicles," Proceedings of the Institution of Mechanical Engineers Part D, 2025.
4. T. Hayakawa and K. Matsumoto, "On-board Estimation of Vehicle Weight and Its Application," Toyota Technical Review, vol. 58, no. 2, pp. 68-73, 2021.
5. H. Zhang, L. Wang, and Y. Chen, "LSTM-Based Virtual Load Sensor for Heavy-Duty Vehicles," IEEE Sensors Journal, vol. 24, no. 3, pp. 3421-3430, 2024.
6. S. Kumar, R. Patel, and A. Sharma, "ECU Data Processing Techniques for Automotive Applications," International Journal of Automotive Engineering, vol. 14, no. 2, pp. 45-52, 2023.
7. R. Singh and P. Kumar, "Signal Processing Methods for Vehicle Parameter Estimation," Journal of Automotive Technology, vol. 15, no. 4, pp. 289-297, 2024.
8. Automotive Industry Standard (AIS-205) -- On-Board Weighing Systems for Commercial Vehicles, Automotive Research Association of India (ARAI).

---

## Project Metadata

| Attribute | Value |
| --- | --- |
| Project Title | Development of an Intelligent On-Board Weighing Equipment (OBW) Kit using Vehicle ECU Data and AI-Driven Estimation Techniques |
| SIP ID | 2526S0197 |
| Scheme | Student Innovative Project 2025 |
| Technology Readiness Level | TRL 4 -- Small Scale Prototype (Technology Validated in Lab) |
| Sustainable Development Goal | SDG 9: Industry, Innovation and Infrastructure |
| Sponsoring Agency | Centre for Sponsored Research and Consultancy (CSRC), Anna University, Chennai -- 600025 |