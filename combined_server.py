"""Serve the React-facing API and legacy Gradio UI from one HF Space."""
from fastapi import FastAPI
import gradio as gr

from api import app as api_app
from app import demo

app: FastAPI = gr.mount_gradio_app(api_app, demo, path="/")
