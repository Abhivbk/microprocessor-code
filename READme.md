# Autonomous Car Project

Welcome to the autonomous car project. This repository is cleanly organized to enforce reliable setups across different development laptops. It provides decoupled architectures for both background (ML testing, stream logic) and foreground (control, simulator interfaces).

## Requirements
- Python 3.10+
- Cargo & Rust (To run `rust-python` image)
- Docker
- FSDS (Flight Simulator Drone Simulator) binary

## 1. Initial Setup
Clone the repository and set up your initial environments. Ensure you place the `FSDS.exe` simulator binary correctly.

1. Download `FSDS.exe` into `foreground/engine_binaries`.
2. Ensure you have moved `setting.json` to the same folder: `foreground/engine_binaries`.
3. Open your terminal in the repository root `microprocessor-code`.

## 2. Install Dependencies
Switch and run installations for both the foreground and background components:

### Foreground Dependencies
```powershell
# Open terminal in root
cd foreground
pip install -r requirements.txt
```

### Background Dependencies
```powershell
# Open a new terminal in root
cd background
pip install -r requirements.txt
```

## 3. Patch msgpack-rpc-python
We use an older `msgpack-rpc-python` library that crashes due to changes in Python 3+. We have provided an automated patch script to resolve this on new machines natively without needing manual source code patching!

Run the patch script from the root of the repository:
```powershell
python patch_msgpackrpc.py
```
*You should see a success message indicating `msgpackrpc` was successfully patched.*

## 4. Run the Container
We provide a Docker image to run your environment consistently. You can build and run using:

```powershell
docker build -t rust-python .

# Windows PowerShell:
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "${PWD}:/work" -w /work rust-python bash

# macOS/Linux:
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "$(pwd):/work" -w /work rust-python bash
```

*(If you are running the project natively without Docker, ensure your Python and Rust environments are correctly set up and skip directly to **Step 5**).*

## 5. Running the Application
To run the full suite, you need to execute both components concurrently. This requires opening two separate terminal windows.

### Terminal 1: Background (ML & Logic)
Navigate to the `background` directory and run the test script:
```powershell
cd background
python python/test.py
```

### Terminal 2: Foreground (Rust Simulator Engine)
Navigate to the `foreground` directory and run the engine via cargo:
```powershell
cd foreground
cargo run
```

## 6. Development Workflow
Create your feature branches as `(your name)_(the function your solving)`. Make sure you are in your branch.

If you are modifying the Machine Learning implementations in `background/`, ensure that any new dependencies are manually added to `background/requirements.txt` (Do not run `pip freeze > requirements.txt` directly as it pollutes the file with local environment paths!).

Please open a Pull Request for all changes to merge into the main branch.
