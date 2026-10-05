import os
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify, request

from data_ingestion import DEFAULT_DATA_DIR
from logger import DEFAULT_LOG_PATH, RuntimeLogger
from model import DEFAULT_MODEL_DIR, model_load, model_predict, model_train


def validate_predict_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    country = payload.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ValueError("'country' is required (a country name, or 'all')")

    raw_date = payload.get("target_date")
    if not isinstance(raw_date, str) or not raw_date.strip():
        raise ValueError("'target_date' is required, format YYYY-MM-DD")
    try:
        datetime.strptime(raw_date.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"'target_date' must look like YYYY-MM-DD, got {raw_date!r}") from None
    return country.strip(), raw_date.strip()


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        DATA_DIR=DEFAULT_DATA_DIR,
        MODEL_DIR=DEFAULT_MODEL_DIR,
        LOG_PATH=DEFAULT_LOG_PATH,
        TOP_N=10,
        QUICK_TRAIN=os.environ.get("AAVAIL_QUICK_TRAIN") == "1",
    )
    if config:
        app.config.update(config)

    logger = RuntimeLogger(app.config["LOG_PATH"])
    state = {"bundles": model_load(app.config["MODEL_DIR"])}
    train_lock = threading.Lock()

    def fail(endpoint, inputs, message, code, started):
        logger.log(endpoint, inputs=inputs, runtime=time.perf_counter() - started,
                   status="error", note=message)
        return jsonify(error=message), code

    @app.route("/train", methods=["POST"])
    def train():
        started = time.perf_counter()
        if not train_lock.acquire(blocking=False):
            return fail("train", None, "a training run is already in progress", 409, started)
        try:
            summary = model_train(app.config["DATA_DIR"], app.config["MODEL_DIR"],
                                  top_n=app.config["TOP_N"], quick=app.config["QUICK_TRAIN"])
            state["bundles"] = model_load(app.config["MODEL_DIR"])
        except FileNotFoundError as exc:
            return fail("train", None, str(exc), 404, started)
        except ValueError as exc:
            return fail("train", None, str(exc), 422, started)
        finally:
            train_lock.release()

        logger.log("train", prediction={"models": [m["country"] for m in summary["models"]]},
                   runtime=time.perf_counter() - started, model_version=summary["model_version"])
        return jsonify(status="trained", **summary)

    @app.route("/predict", methods=["POST"])
    def predict():
        started = time.perf_counter()
        payload = request.get_json(silent=True)
        try:
            country, target_date = validate_predict_payload(payload)
        except ValueError as exc:
            return fail("predict", payload if isinstance(payload, dict) else None,
                        str(exc), 400, started)

        inputs = {"country": country, "target_date": target_date}
        if not state["bundles"]:
            return fail("predict", inputs, "no trained models found, call /train first", 503, started)
        try:
            result = model_predict(state["bundles"], country, target_date)
        except ValueError as exc:
            return fail("predict", inputs, str(exc), 400, started)

        request_id = logger.log("predict", inputs=inputs, prediction=result,
                                runtime=time.perf_counter() - started,
                                model_version=result["model_version"],
                                note="out-of-distribution input" if result["out_of_distribution"] else None)
        return jsonify(request_id=request_id, **result)

    @app.route("/logfile", methods=["GET"])
    def logfile():
        try:
            n = int(request.args.get("n", 100))
        except ValueError:
            return jsonify(error="'n' must be an integer"), 400
        if not 1 <= n <= 10000:
            return jsonify(error="'n' must be between 1 and 10000"), 400
        return Response(logger.tail_text(n), mimetype="text/plain")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(405)
    def wrong_method(_):
        return jsonify(error="method not allowed"), 405

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
