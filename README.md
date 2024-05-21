## Setup and Running Project with virtualenv

1. Clone the repository:
```bash
git clone https://github.com/Nirmal-Re/nlp-gp-cw.git
```
2. Navigate to the repository
```bash
cd nlp-gp-cw
```
Create a virutal env using 
```bash
virtualenv env
```

If you don't have virutalenv, install it using -

```bash
pip install virtualenv
```

3. Activate the virtual environment: On Windows, run:
```bash
env/Scripts/activate
```
On Linux/Unix

```bash
source env/bin/activate
```
4. Install the required packages:
```bash
pip install -r requirements.txt
```
6. Run the project
```bash
python app.py
```
