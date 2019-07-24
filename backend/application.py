#!flask/bin/python
from flask import Flask, jsonify
from flask import abort
from flask import request
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()
from azure.storage.blob import BlockBlobService
import json
import tempfile

app = Flask(__name__)

accountName = 'podcaststorage'
accountKey = 'IEgR3VJ9te4c3TwMCSqMijIJrMEoJJYsMikQ4LDWjHnRRTbEVmy+M3AG1MC5XwIDDVN/PnYFCYLPq3O1foXOKw=='
getCallContainerName = 'whoismasterpreds'
postCallContainerName = 'dummydata'

toServe = set()
served = set()
written = set()
blob_count = 0

def ends_with_json(s):
    if s.endswith('.json'):
        return True
    return False

def get_blob_listings():
    block_blob_service = BlockBlobService(account_name=accountName,
                                          account_key=accountKey)
    generator = block_blob_service.list_blobs(getCallContainerName)
    tmp_list = [x.name for x in generator]

    global toServe
    toServe = set(filter(ends_with_json, tmp_list))

def get_blob(file_name):
    block_blob_service = BlockBlobService(account_name=accountName,
                                          account_key=accountKey)
    json_data = block_blob_service.get_blob_to_text(getCallContainerName, file_name)
    return json_data.content

def clear_globals():
    global toServe
    toServe = set()
    global served
    served = set()
    global written
    served = written()
    global blob_count
    blob_count = 0

def get_blob_data():
    #Get listings from blob
    #Pick one that has not been served
    global toServe
    global served
    global written
    global blob_count
    if len(written) == blob_count and blob_count != 0:
        #There could be new data so clear the dictionary and call blob_listing again
        raise Exception

    if len(toServe) == 0 and blob_count == 0:
        get_blob_listings()

    if len(toServe) == 0 and len(written) != blob_count:
        toServe = toServe.union(served)
        served.clear()

    chosen_file = list(toServe - served)[0]

    return get_blob(chosen_file), chosen_file

def write_blob_data(file_name, content):
    try:
        block_blob_service = BlockBlobService(account_name=accountName,
                                              account_key=accountKey)
        block_blob_service.create_blob_from_text(postCallContainerName, file_name, content)
        tmp = tempfile.NamedTemporaryFile()
        block_blob_service.get_blob_to_stream(postCallContainerName, file_name, tmp)
        return 201
    except:
        return 400

@app.route('/get/raw', methods=['GET'])
def get_raw():
    try:
        content, fn = get_blob_data()
        json_content = json.loads(content)
        content = json.dumps(json_content)
        global served
        global toServe
        served.add(fn)
        toServe.remove(fn)
        return content, 201
    except:
        abort(400)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/post/candidates', methods=['POST'])
def write_candidate():
    if not request.json:
        abort(400)

    uri = request.json.get('uri')
    val = uri.split('/')[-1].split('.')[0]
    fname = val + '.json'
    status = write_blob_data(fname, str(request.json))

    if status >= 300:
        abort(500)
    else:
        global written
        global served
        written.add(fname)
        served.remove(fname)
        return jsonify({'task': fname}), 201

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
