# Resume Matcher

A Flask web app that analyzes your resume (PDF or DOCX) and matches it against job listings fetched from the Adzuna API. Uses AI-powered similarity scoring and ATS-style heuristics to rank job matches.

## Features

- Upload resume (PDF/DOCX) and get instant job matches
- AI-powered resume-to-job similarity scoring
- ATS (Applicant Tracking System) heuristic analysis
- Skill extraction and coverage reporting
- Filter by job category, country, and location
- Track applied jobs

## Quick Start

Choose one of the options below based on your system:

### Option 1: Run Locally with Python (Windows, macOS, Linux)

**Prerequisites:** Python 3.8+ installed

#### Windows (PowerShell)

```powershell
# Clone the repository (if not already)
git clone https://github.com/hardeepsinghdot/job-hawk.git
cd job-hawk

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

Visit `http://localhost:5000` in your browser.

#### macOS / Linux (Bash/Zsh)

```bash
# Clone the repository (if not already)
git clone https://github.com/hardeepsinghdot/job-hawk.git
cd job-hawk

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

Visit `http://localhost:5000` in your browser.

**Troubleshooting local install:**
- If `pip install -r requirements.txt` fails on `torch`, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to install the correct wheel for your OS/GPU.
- On macOS, you may need `xcode-select --install` to get build tools.
- On Linux, you may need `sudo apt-get install build-essential python3-dev` (Ubuntu/Debian) or equivalent.

### Option 2: Run with Docker (Any System)

**Prerequisites:** Docker Desktop installed ([download here](https://www.docker.com/products/docker-desktop))

#### Build and run:

```powershell
# Build the Docker image
docker build -t resume-matcher:latest .

# Run the container
docker run -e ADZUNA_APP_ID=00f602bf -e ADZUNA_API_KEY=a190d32c263caa6056651b729cb47fbb -p 8000:8000 resume-matcher:latest
```

Or, easier with `docker-compose`:

```powershell
# Run with docker-compose (handles env vars and port mapping)
docker-compose up --build
```

Visit `http://localhost:8000` in your browser.

**Docker benefits:**
- Same environment on Windows, macOS, and Linux—no OS-specific issues.
- No need to install Python or heavy dependencies locally.
- Ideal for sharing with friends or deploying to cloud.

### Option 3: Run with GitHub Pages + Cloud Backend (Public Link for Everyone)

This option gives you a **public URL** that anyone can use without installing anything.

1. **Deploy the Flask backend** to a cloud host (Render, Heroku, Railway, Google Cloud Run):
   - Go to [render.com](https://render.com) and sign up.
   - Click "New +" → "Web Service" → Connect your GitHub repo → select branch `main`.
   - Build Command: (leave empty—Render auto-installs from `requirements.txt`)
   - Start Command: `gunicorn main:app`
   - Add environment variables:
     - `ADZUNA_APP_ID`: `00f602bf`
     - `ADZUNA_API_KEY`: `a190d32c263caa6056651b729cb47fbb`
   - Click Deploy and note your public URL (e.g., `https://your-app.onrender.com`).

2. **Update the static frontend** with your backend URL:
   - Edit `docs/index.html` and replace `https://YOUR_BACKEND_URL_HERE` with your Render URL.
   - Commit and push:
     ```powershell
     git add docs/index.html
     git commit -m "Set backend URL"
     git push origin main
     ```

3. **Enable GitHub Pages** for this repo:
   - Go to GitHub → Repo Settings → Pages.
   - Source: Branch `main` → Folder `/docs`.
   - GitHub will publish at `https://<your-username>.github.io/<repo>/`.
   - **That's your public link!** Share it with anyone.

## API Endpoints

- `GET /` — Web form to upload resume and select filters.
- `POST /` — Submit resume and filters; returns HTML results page.
- `POST /api/analyze` — JSON API; submit resume and filters; returns JSON with job matches and stats.
- `GET /applied-jobs` — View applied jobs tracker.
- `POST /applied-jobs` — Add a job to the tracker.

## Environment Variables

Optional (defaults work out-of-the-box):
- `ADZUNA_APP_ID` — Adzuna API app ID (default: `00f602bf`)
- `ADZUNA_API_KEY` — Adzuna API key (default: `a190d32c263caa6056651b729cb47fbb`)

Set them in your shell, `.env` file, or Docker/cloud host dashboard.

## Project Structure

```
.
├── main.py               # Flask app and business logic
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Local Docker Compose setup
├── Procfile              # For cloud deployment (Render, Heroku)
├── templates/
│   ├── index.html        # Web form
│   ├── results.html      # Results page (web form flow)
│   └── applied_jobs.html # Applied jobs tracker
├── docs/
│   └── index.html        # Static frontend for GitHub Pages (calls /api/analyze)
├── uploads/              # Uploaded resume storage (temp)
├── applied_jobs.json     # Tracker data
└── README.md             # This file
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
- Install PyTorch for your OS: https://pytorch.org/get-started/locally/
- Or use Docker (sidesteps all of this).

### "CORS error when using GitHub Pages + backend"
- Ensure the backend `/api/analyze` endpoint has CORS enabled (already configured in `main.py`).
- Check that `docs/index.html` points to the correct backend URL (with no trailing slash).

### "Model downloading slowly"
- First request downloads the sentence-transformer model (~100MB). Be patient—subsequent requests are fast.
- Consider upgrading to a bigger cloud instance if deploying.

### "Port 8000/5000 already in use"
- Change the port. Locally: `python main.py` uses 5000 by default (change in code if needed).
- With Docker: `docker run -p 9000:8000 ...` maps port 9000 on your machine to 8000 in the container.

## Contributing

Have questions or want to improve the app? Feel free to fork and submit a PR!

## License

MIT