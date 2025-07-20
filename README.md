# 🚗 SafeSpikr: Edge-SNN for Driver Risk Prediction with Verilog-Backed Personalization

SafeSpikr is an innovative, real-time Edge AI system designed for low-latency and energy-efficient prediction of driver behavioral risk (e.g., drowsiness, stress, distraction). It leverages Spiking Neural Networks (SNNs) and Verilog-accelerated personalization modules, running entirely in simulation to provide powerful in-cabin monitoring without requiring costly FPGA hardware.

---

## 🧠 Problem Statement

This project addresses **Edge AI Models for Driver Behavior Prediction & Autonomous Driving Systems**, a critical area for enhancing road safety and the reliability of autonomous vehicles.

---

## 🎯 Objective

Our goal is to build a real-time driver behavior prediction pipeline utilizing:

* **Spiking Neural Networks (SNNs):** For event-driven, low-power inference.
* **Verilog Modules:** For hardware-accelerated Multiply-Accumulate (MAC) and learning blocks.
* **Driver-specific personalization:** Achieved through an online hardware optimizer.
* **Simulated input:** Combining images and Photoplethysmography (PPG) data with hardware logic using Verilator and GTKWave.

---

## 🌟 Key Features

| Feature                   | Description                                                                 |
| :------------------------ | :-------------------------------------------------------------------------- |
| ✅ Neuromorphic Inference   | SNNs are ideal for low-latency, low-energy processing on the edge.          |
| ✅ Custom Verilog Hardware  | Spiking MACs and Optimizers are custom-implemented in Verilog.              |
| ✅ Personalized Learning    | An online learning block adjusts SNN weights specifically for each driver.  |
| ✅ Fully Free Simulation    | Verilator and GTKWave enable comprehensive testing without an FPGA.         |
| ✅ Hybrid Input Simulation  | Webcam images and PPG traces are used to mimic real driving conditions.     |

---

## 🛠️ Tech Stack

### 🔧 AI/ML

* Python, NumPy, OpenCV
* [Brian2](https://brian2.readthedocs.io/) or [Nengo](https://www.nengo.ai/) for SNN modeling
* *Optional:* PyTorch for pretraining

### 💾 Verilog + HLS

* **Custom Modules:** `mac_unit.v`, `optimizer.v`, `neuron.v`
* **Simulation:** Verilator, GTKWave
* *Optional:* hls4ml / Xilinx Vitis HLS for hardware export

### 🧪 System Testing

* Python testbench (`sim_driver.py`)
* Realistic inputs (images + PPG CSVs)
* Waveform validation (via GTKWave)

### 🖥️ Frontend (Optional)

* Streamlit/Flask for a live alert dashboard

---

## 📁 Project Structure

```
SafeSpikr/
├── data/                # Input data (images, PPG, labels)
├── gui/                 # Optional frontend (e.g., app.py)
├── model/               # SNN model, training, quantization scripts
├── simulation/          # Verilog simulation drivers, VCD waveforms
├── utils/               # Preprocessing, metrics, synthetic data generation
├── verilog/             # Custom Verilog modules (mac_unit, optimizer, neuron, top_module)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE
```

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tejasp0008/SafeSpikr.git
   cd SafeSpikr
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Verilator and GTKWave:**
   - On Ubuntu:
     ```bash
     sudo apt-get install verilator gtkwave
     ```
   - Or follow instructions at [Verilator](https://verilator.org/) and [GTKWave](http://gtkwave.sourceforge.net/).

4. **Run a simulation:**
   - Prepare your data in the `data/` folder.
   - Launch the simulation driver:
     ```bash
     python simulation/sim_driver.py
     ```
   - View waveforms with GTKWave:
     ```bash
     gtkwave simulation/waveform.vcd
     ```

5. **(Optional) Launch the dashboard:**
   ```bash
   python gui/app.py
   ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Acknowledgments

* [Nengo](https://www.nengo.ai/) and [Brian2](https://brian2.readthedocs.io/) for SNN frameworks.
* [Verilator](https://verilator.org/) and [GTKWave](http://gtkwave.sourceforge.net/) for simulation tools.
* Open source contributors and the community for their support.

---
