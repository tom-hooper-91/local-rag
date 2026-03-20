# Python Virtual Environments

Virtual environments isolate Python dependencies per project, preventing conflicts between packages.

## Creating a Virtual Environment

```bash
python -m venv .venv
```

This creates a `.venv` directory containing a standalone Python installation.

## Activating

- **Windows:** `.venv\Scripts\activate`
- **Linux/macOS:** `source .venv/bin/activate`

When activated, `pip install` installs packages into the virtual environment instead of the system Python.

## Requirements Files

Track dependencies with `pip freeze > requirements.txt` and install them with `pip install -r requirements.txt`. Pin versions for reproducibility.

## Best Practices

- Always use a virtual environment for project work
- Add `.venv/` to `.gitignore`
- Pin dependency versions in `requirements.txt`
- Use separate environments for different projects
