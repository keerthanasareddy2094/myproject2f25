## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).


---

## Running the Application

Follow these steps to get the application running on your local machine.

### Step 1: Remove the existing code directory completely

Because the local repository can't been updated correctly, need to remove the directory first.

```bash
rm -rf team2f25
```

### Step 2: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/DrAlzahrani2025Projects/team2f25.git
```

### Step 3: Navigate to the Repository

Change to the cloned repository directory:

```bash
cd team2f25
```

### Step 4: Pull the Latest Version

Update the repository to the latest version:

```bash
git pull origin main
```


### Step 5: Make Scripts Executable

Make the setup and cleanup scripts executable. This step only needs to be done once.

*Note: If you are on Windows, you must run this command in a `bash` terminal, such as the one provided by Git Bash or WSL.*

```bash
chmod +x startup.sh cleanup.sh
```


### Step 6: Run the startup script

This will automatically build, start and run the container

```bash
./startup.sh
```

### Step 7: Access the internship finder chatbot

For Streamlit:

- Once the container starts, Open browser at : http://localhost:5002/team2f25

  

---
### Step 8: Clean Up

When you are finished, run the cleanup script to stop and remove the Docker container and image.

```bash
./cleanup.sh
```

---

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csusb.edu/team2f25 

## Google Colab Notebook  
[Open in Colab](https://colab.research.google.com/drive/1icOiUzhhm0l7PkDoCxUdDMqpX1eua8ug?usp=sharing)
