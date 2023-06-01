# Todo List
- [x] Download raw data to ./data/raw/
   - Raw data was saved to /home/quandou/Projects/CMRxRecon/data/raw/ (on the deep server deep.bme.virginia.edu)
- [x] Check the data format
   - Raw data was stored in the Version 7.3 MAT files, which use the HDF5 based format
   - There is no need to convert the data into other format, Python package [h5py](https://docs.h5py.org/en/stable/index.html) or [mat73](https://github.com/skjerns/mat7.3) can be directly used to read the data
- [x] Run the standard code to check the reconstruction workflow
   - The standard code provides examples to reconstruct the image with zero-filling, SENSE (ESPIRiT), or GRAPPA in MATLAB
   - The SENSE (ESPIRiT) reconstruction result can be reproduced in Python using the [sigpy](https://sigpy.readthedocs.io/en/latest/) package
- [ ] Find the current state-of-the-art CINE reconstruction technique
   1. [A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction](https://ieeexplore.ieee.org/document/8067520). Jo Schlemper, Jose Caballero, Joseph V. Hajnal, Anthony N. Price, Daniel Rueckert. 2018.
   2. [Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction](https://ieeexplore.ieee.org/document/8425639). Chen Qin, Jo Schlemper, Jose Caballero, Anthony N. Price, Joseph V. Hajnal, Daniel Rueckert. 2018.
   3. [CINENet: deep learning-based 3D cardiac CINE MRI reconstruction with multi-coil complex-valued 4D spatio-temporal convolutions](https://www.nature.com/articles/s41598-020-70551-8). Thomas Küstner, Niccolo Fuin, Kerstin Hammernik, Aurelien Bustin, Haikun Qi, Reza Hajhosseiny, Pier Giorgio Masci, Radhouene Neji, Daniel Rueckert, René M. Botnar, Claudia Prieto. 2020.
   4. [Accelerating cardiac cine MRI using a deep learning-based ESPIRiT reconstruction](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28420). Christopher M. Sandino, Peng Lai, Shreyas S. Vasanawala, Joseph Y. Cheng. 2020.
   5. ... 
- [ ] Write code for building the network
   - It seems that all of the models have the following the structure:
   ```
   input -> CNN -> Data Consistency -> CNN -> Data Consistency -> ... -> output
   ```
   - The models are different in:
      1. 2D or 2D+t (3D) or 3D+t (4D)
      2. complex-valued or two-channel real-valued
      3. UNet CNN or ResNet CNN
- [ ] Write training code
- [ ] Run experiments
- [ ] Compare the performance of different models
- [ ] Write deployment code
- [ ] Build Docker image