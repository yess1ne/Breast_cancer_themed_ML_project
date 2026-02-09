# Virtual Environment Setup Guide
this is minimal documentation regarding how to set up and use the software for now, plus the chatbot feature is not available for now, prone to change.
This is how to create a Python virtual environment, activate it, and install the project dependencies from `requirements.txt`.


---
## 1. Clone the repo:
   ```bash
   git clone https://github.com/yess1ne/Breast_cancer_themed_ML_project.git
   cd Breadt_cancer_themed_ML_project
   ```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

**Windows (PowerShell):**
```bash
venv\\Scripts\\Activate
```

**Windows (CMD):**
```bash
venv\\Scripts\\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

---

## 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Deactivate the Virtual Environment

```bash
deactivate
```

---

## 6. Optional: Upgrade pip

```bash
pip install --upgrade pip

