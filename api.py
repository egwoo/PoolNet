from flask import Flask, jsonify, request
from api_solver import Solver
from scipy import ndimage
from api_config import Config
from uuid import uuid4
from pathlib import Path
import os

app = Flask(__name__)

# Is this fine?
solver = Solver(Config())


@app.route("/", methods=["POST"])
def score():
    data = request.get_data()
    image_path = Path("/tmp") / (str(uuid4()))
    with open(image_path, "wb") as image_result:
        image_result.write(data)

    pred = solver.score(str(image_path)).tolist()
    os.remove(image_path)
    return jsonify(pred)

    # y, x = ndimage.measurements.center_of_mass(multi_fuse)

    # import cv2

    # multi_fuse = 255 * pred
    # cv2.imwrite(
    #     "./results/results.png",
    #     multi_fuse,
    # )
