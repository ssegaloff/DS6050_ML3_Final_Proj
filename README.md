# ML III Group Project

Welcome to the project! 

We are using `uv` to manage our Python environment and dependencies. This ensures that everyone on the team is using the exact same version of Python and the exact same package versions, preventing "it works on my machine" bugs.

## 🛠 1. First-Time Setup

Before you start, make sure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) on your machine.

1. **Clone the repository:**
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```


2. **Sync the environment:**
Run the following command in the project folder:
```bash
uv sync
```

**What this does:** `uv` will automatically read the `.python-version` file to download the correct Python version if you don't have it. It will then read `uv.lock` and `pyproject.toml` to create a virtual environment (a `.venv` folder) and install all the required packages perfectly matched to the rest of the team.

3. **Activate the environment:**
To work on the project, activate the virtual environment so your terminal uses the correct Python and packages:
- **Mac/Linux:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

## 🔄 2. Daily Git Workflow
When you sit down to work on the project, follow this cycle to keep your work in sync with the group:

1. **Always pull first:** Before making any changes, grab the latest updates from the team.

```bash
git pull origin main
```

*(Note: If someone added a new package to the project, run `uv sync` again after pulling to install it.)*

2. **Make your changes:**
Write your code, save your files, and test your work.

3. **Stage and Commit:**
Once your work is ready, stage the files and save them with a descriptive message.

```bash
git add .
git commit -m "Briefly describe what you changed or added"
```

4. **Push to share:**
Send your committed changes up to the shared repository.

```bash
git push origin main
```

## 📦 3. Adding New Packages
If you need to install a new Python package (like `pandas` or `requests`) for your code to work, **do not use** `pip install`.

Instead, use `uv`:

```bash
uv add <package_name>
```

This will install the package and automatically update the `pyproject.toml` and `uv.lock` files. **You must commit and push these two files** so that when the rest of the team runs `git pull` and `uv sync`, they get the new package too!