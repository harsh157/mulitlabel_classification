# -*- coding: utf-8 -*-

import os
import argparse
from flask import Flask, request
from intent_classifier import IntentClassifier

app = Flask(__name__)
model = IntentClassifier()

def parse_request(request):
    """Function to parse request object
    Parameters
    ----------
    request: flask request object.

    Returns
    -------
    Tuple of 
    invalid_request: bool
        whether request is invalid
    text: str
        parsed text from request body
    output: dict
        message to return when invalid request
    error_code: int
        error code when invalid request
    """


    data = request.get_json()
    message = ""
    invalid_request = True
    error_code = 0
    label = ""
    text = ""
    if not request.is_json:
        label = "BODY_MISSING"
        message = "Request doesn't have a body."
        error_code = 400
    elif 'text' not in data:
        label = "TEXT_MISSING"
        message = "\"text\" missing from request body."
        error_code = 400
    elif type(data['text']) != str:
        label = "INVALID_TYPE"
        message = "\"text\" is not a string."
        error_code = 400
    elif data['text'] == "":
        label = "TEXT_EMPTY"
        message = "\"text\" is empty."
        error_code = 400
    else:
        text = data['text']
        invalid_request = False
    output = {'label': label, 'message': message}
    return invalid_request, text, output, error_code

        

@app.route('/ready')
def ready():
    if model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent():
    # Implement this function according to the given API documentation
    invalid_request, text, output, error_code = parse_request(request)
    if not invalid_request:
        try:
            output = model.predict(text)
            error_code = 200
        except Exception as err:
            output = { 'label': "INTERNAL_ERROR", 
                       'message': str(err)
                    }
            error_code = 500
    return output, error_code


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    model.load(args.model)
    app.run(port=args.port)


if __name__ == '__main__':
    main()
