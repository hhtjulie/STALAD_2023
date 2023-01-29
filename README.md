# STALAD2023
Implementation of STALAD in [CCL20]

reference: [CCL20] Chen, C. Y., Chang, S. C., & Liao, D. Y. (2020). Equipment anomaly detection for semiconductor manufacturing by exploiting unsupervised learning from sensory data. Sensors, 20(19), 5650.

<!-- ABOUT THE PROJECT -->
## About The Project (Excerpt from the abstract in [CCL20])
In-line anomaly detection (AD) not only identifies the needs for semiconductor equipment maintenance but also indicates potential line yield problems. Prompt AD based on available equipment sensory data (ESD) facilitates proactive yield and operations management. However, ESD items are highly diversified and drastically scale up along with the increased use of sensors. Even veteran engineers lack knowledge about ESD items for automated AD. This paper presents a novel Spectral and Time Autoencoder Learning for Anomaly Detection (STALAD) framework. The design consists of four innovations: 
1. identification of cycle series and spectral transformation (CSST) from ESD, 
2. unsupervised learning from CSST of ESD by exploiting Stacked AutoEncoders, 
3. hypothesis test for AD based on the difference between the learned normal data and the tested sample data, 
4. dynamic procedure control enabling periodic and parallel learning and testing. 

Applications to ESD of an HDP-CVD tool demonstrate that STALAD learns normality without engineers’ prior knowledge, is tolerant to some abnormal data in training input, performs correct AD, and is efficient and adaptive for fab applications. Complementary to the current practice of using control wafer monitoring for AD, STALAD may facilitate early detection of equipment anomaly and assessment of impacts to process quality.
