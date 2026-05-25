#!/usr/bin/env python3
import connexion
from dataclasses import dataclass
from typing import List
from requests_toolbelt import MultipartEncoder
from flask import Flask, request, Response, send_from_directory
import flask

@dataclass
class Model:
    type: str
    version: str
    description: str
    name: str
    labels: List[str]



def get_models():
    return [Model("segmentation", "1", "NSU test model", "brain_segm", ["brain", "MRI", "brain tumor"])]

def inference(model, datapoint):
    print("Model: ", model)
    print("mriImage: ", datapoint)

    m = MultipartEncoder(fields={'params': '{"points": [], "extreme_points": []}', 'file': ("somename.dat", datapoint, "application/octet-stream")})
    return Response(m.to_string(), mimetype=m.content_type)

def go_home():
    """
    GET /
    Serves the home page
    """

    return flask.send_from_directory('templates', 'index.html')


app = connexion.FlaskApp(__name__, specification_dir='swagger/')
app.add_api('server.yaml')
app.run(port=5000)

