import io
import pickle
import requests
import time
import torch


def _parse_model(ser):
    data = pickle.loads(ser)
    model_bytes = data['model_bytes']
    model_metadata = data['model_metadata']
    model_bytes_io = io.BytesIO(model_bytes)
    model = torch.load(model_bytes_io)
    return model, model_metadata


def _get_model_metadata(server):
    url = f'http://{server}/api/get_model_metadata'
    r = requests.get(url)
    assert r.ok, r.text
    return pickle.loads(r.content)['model_metadata']


def wait_for_next_round(server, last_trained_round=None, join_late_by_max=10):
    while True:
        metadata = _get_model_metadata(server)
        if metadata['round_age'] < join_late_by_max:
            if last_trained_round != metadata['round']:
                return
        time.sleep(1)


def get_model_and_notify_client_started(server, client_id):
    url = f'http://{server}/api/get_model_and_notify_client_started'
    r = requests.post(url, data={'client_id': client_id})
    assert r.ok, r.text
    return _parse_model(r.content)


def upload_updated_model(server, client_id, model, model_metadata):
    model_bytes_io = io.BytesIO()
    torch.save(model, model_bytes_io)
    model_bytes_io.seek(0)
    model_bytes = model_bytes_io.read()
    data = {'model_bytes': model_bytes, 'model_metadata': model_metadata, 'client_id': client_id}
    ser = pickle.dumps(data)

    url = f'http://{server}/api/upload_updated_model'
    r = requests.post(url, data=ser)
    if not r.ok:
        print('ERROR', r.text)
