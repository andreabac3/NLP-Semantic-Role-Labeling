from flask import Flask, request, jsonify

from stud.implementation import build_model_34, build_model_234, build_model_1234

app = Flask(__name__)
model_34 = build_model_34('cpu')

try:
    model_234 = build_model_234('cpu')
except:
    model_234 = None

try:
    model_1234 = build_model_1234('cpu')
except:
    model_1234 = None


def prepare_data(data):
    data_34 = data
    data_234 = {
        'words': data['words'],
        'lemmas': data['lemmas'],
        'pos_tags': data['pos_tags'],
        'dependency_heads': data['dependency_heads'],
        'dependency_relations': data['dependency_relations'],
        'predicates': [1 if p != '_' else 0 for p in data['predicates']],
    }
    data_1234 = {
        'words': data['words'],
        'lemmas': data['lemmas'],
        'pos_tags': data['pos_tags'],
        'dependency_heads': data['dependency_heads'],
        'dependency_relations': data['dependency_relations'],
    }

    return data_34, data_234, data_1234


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):

    try:

        json_body = request.json
        data = json_body['data']
        data_34, data_234, data_1234 = prepare_data(data)

        predictions_34 = model_34.predict(data_34)

        if model_234:
            predictions_234 = model_234.predict(data_234)
        else:
            predictions_234 = None

        if model_1234:
            predictions_1234 = model_1234.predict(data_1234)
        else:
            predictions_1234 = None


    except Exception as e:

        app.logger.error(e, exc_info=True)
        return (
            {
                'error': 'Bad request',
                'message': 'There was an error processing the request. Please check logs/server.stderr'
            },
            400
        )

    return jsonify(
        data=data,
        predictions_34=predictions_34,
        predictions_234=predictions_234,
        predictions_1234=predictions_1234)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)