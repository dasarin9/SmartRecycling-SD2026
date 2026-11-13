# SmartRecycling-SD2026
# Smart Recycling Assistant (SD2026)

An edge-AI smart recycling bin system designed to process visual and chemical data to provide real-time disposal instructions with an end-to-end latency under 5 seconds.

## Hardware Architecture
* **Raspberry Pi 5 CPU:** Executes the Vision CNN via ExecuTorch (XNNPACK) and handles the IR Sensor Random Forest classifier processing.
* **Raspberry Pi AI HAT+ 2 (Hailo-10H NPU):** Runs the 4-bit quantized Llama 3.2 1B model via the `hailo-ollama` server to generate disposal instructions rapidly without consuming CPU RAM.

## Subsystems
* `llm_integration/` - Llama 3.2 1B deployment, system prompting, and Hailo-Ollama API streaming 
* `vision_cnn/` - Vision classification models and ExecuTorch compilation 
* `sensor_data/` - IR sensor data collection and Random Forest classifier 
* `display_ui/` - Display module and user interface 
