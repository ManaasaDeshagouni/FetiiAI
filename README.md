## FetiiAI - Deployment

### Local (Docker)

1. Build image:
```
docker build -t fetiiai .
```
2. Run:
```
docker run -p 8501:8501 -e OPENAI_API_KEY=YOUR_KEY fetiiai
```

Visit http://localhost:8501

### Gradio Demo (local)

```
pip install -r requirements.txt
python gradio_app.py
```
Then open the printed local/public URL.

### HuggingFace Spaces (Gradio)

1. Create a new Space (Gradio template)
2. Upload repo files; set `gradio_app.py` as the entrypoint
3. Add secret `OPENAI_API_KEY`
4. Deploy

### GitHub Actions (CI)

Pushing to `main` builds the Docker image using `.github/workflows/ci.yml`.

### Deploy to Render

1. Connect this repo to Render.
2. Render will read `render.yaml` and create a web service.
3. Add env var `OPENAI_API_KEY` in the Render dashboard.
4. Deploy.


