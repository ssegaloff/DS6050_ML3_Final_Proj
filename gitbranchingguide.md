# ML III Group Project: Team Toothless!

Welcome to the project! (This is a version of the readme that explains how to git branch).

We are using `uv` to manage our Python environment and dependencies. This ensures that everyone on the team is using the exact same version of Python and the exact same package versions, preventing "it works on my machine" bugs.

---

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

---

## 🔄 2. Daily Git Workflow

When you sit down to work, follow this cycle to keep your work in sync with the group.

### Step 1 — Sync with the team first
Before writing a single line of code, pull the latest version of `main` so you're starting from the most up-to-date codebase.

```bash
git checkout main
git pull origin main
```

If `uv.lock` or `pyproject.toml` changed since your last pull (i.e., a teammate added a new package), also run:
```bash
uv sync
```

### Step 2 — Create a branch for your work
**Never commit directly to `main`.** Instead, create a feature branch. Suggestion: name it with a short category prefix so it's easy to understand at a glance:

- `feat/` — new code, models, or pipeline stages (e.g., `feat/add-demographic-parser`)
- `fix/` — bug fixes (e.g., `fix/matrix-dimension-error`)
- `docs/` — README or notes updates (e.g., `docs/update-methodology`)
- `data/` — changes to ingestion, cleaning, or storage logic (e.g., `data/update-schema-logic`)

```bash
git checkout -b feat/your-branch-name
```

### Step 3 — Make your changes
Write your code, save your files, and test your work. Use `git status` at any point to see which files you've changed.

### Step 4 — Commit your work
Once your work is ready, stage and save it with a descriptive message. Clear commit messages make the project history much easier to read.

```bash
git add .
git commit -m "feat: add demographic parser for census data"
```

### Step 5 — Push your branch
The first time you push a new branch, use:
```bash
git push -u origin your-branch-name
```
After that, a plain `git push` is enough.

### Step 6 — Open a Pull Request (PR) on GitHub
When your feature is done and tested, go to GitHub. You'll see a prompt to **"Compare & pull request"** — click it, write a short description of what you did, and tag a teammate to review it. Once approved, click **Merge Pull Request** and delete your feature branch on GitHub.

> **Tip:** Merge early and often. Smaller PRs mean fewer conflicts and faster reviews.

---

## 🔀 3. Staying in Sync Mid-Work

If a teammate merges a big change into `main` while you're still working on your branch, pull their changes in so you don't end up with a massive conflict at the end.

```bash
# While on your feature branch:
git pull origin main
```

If `pyproject.toml` or `uv.lock` changed, run `uv sync` afterward too. If Git flags any conflicts, see the section below.

---

## 🧩 4. Resolving Merge Conflicts

If Git says "Automatic merge failed," don't panic — this just means two people edited the same part of a file. Open the conflicting file in your editor and you'll see markers like this:

```python
<<<<<<< HEAD
# This is your version of the code
df = clean_data(raw_data, strict=True)
=======
# This is the version currently on main
df = clean_data(raw_data, drop_nulls=False)
>>>>>>> main
```

To resolve it:
1. Manually edit the file to look exactly how it should — combine the logic or pick the correct version.
2. **Delete** the `<<<<`, `====`, and `>>>>` marker lines entirely.
3. Save the file.
4. Stage and commit the resolution:

```bash
git add <filename>
git commit -m "resolve: merge conflict in cleaning script"
```

> **Note:** Conflicts can also happen in `pyproject.toml` or `uv.lock` if two people added packages at the same time. They look scarier, but you resolve them exactly the same way. When in doubt, ask in the group chat before guessing.

---

## 📦 5. Adding New Packages

If you need to install a new Python package for your code to work, **do not use** `pip install`.

Instead, use `uv`:

```bash
uv add <package_name>
```

This installs the package and automatically updates `pyproject.toml` and `uv.lock`. **You must commit and push both of those files** so that when your teammates run `git pull` and `uv sync`, they get the new package too.