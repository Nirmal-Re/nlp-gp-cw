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

## Docker image pulling and running commands

To pull the docker image from docker hub run the following command:

**docker pull nirmalbhandari/lm-server** \
This will pull down the docker image from docker hub (bare in mind its quite large)

To run the docker image inside of a container run the following command:

**docker run -d -p 4000:5000 --name mycontainer nirmalbhandari/lm-server** \
This will run the image inside of the container **mycontainer** (can be changed) on host port 4000 and container port 5000. Feel free to change the host port from 4000 to any desired port, though the container port must stay the same for the application to function (unless you change the port in app.py).
