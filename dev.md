From Python's website https://www.python.org/downloads/ download 3.12

Ensure all installation tools are up to date
```
python3.12 -m pip install --upgrade pip setuptools wheel
```

Create virtual environment
```
python3.12 -m venv .env
```

Activate the new virtual environment
```
source .env/bin/activate
```

Install project packages
```
pip install -r requirements.txt
```