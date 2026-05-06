# DS-3002 Assignment 4

Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks on Heart Disease and MNIST.

## Project Contents
- `i23-2605_B_A4.ipynb`: main notebook covering Preprocessing and Parts A-E
- `processed.cleveland.data`: UCI Cleveland dataset used in the notebook
- `app.py`: local Streamlit front-end for heart disease risk prediction
- `heart_rf_pipeline.pkl`: serialized trained heart model pipeline
- `sample_patient.pkl`: default patient values for quick app demo
- `top_features.pkl`: precomputed top feature drivers for app display
- `requirements.txt`: Python dependencies

## Run Notebook
1. Open `i23-2605_B_A4.ipynb`.
2. Select the project Python environment.
3. Run cells from top to bottom.

## Run App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The notebook includes offline-friendly handling for MNIST loading.
- Fixed random seeds are used for reproducibility across experiments.
