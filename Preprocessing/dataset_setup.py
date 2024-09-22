import requests
from converter import progress_bar

urls = [
    ('eeg_ill.zip', 'http://brain.bio.msu.ru/eeg_data/schizophrenia/sch.zip'),
    ('eeg_health.zip', 'http://brain.bio.msu.ru/eeg_data/schizophrenia/norm.zip'),
    ('edf_data.zip','https://repod.icm.edu.pl/api/datasets/251/versions/59/files/download?format=original'),
    ('csv_data_one.zip','https://storage.googleapis.com/kaggle-data-sets/4369/271524/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240922%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240922T210309Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=68f3954b288a5b31f921eb6d807e9cae0a7f49841070bdc7cb563c92bbbbeb0c9a6368a46bf34310b86f5cf2f52b4c1ee95782eb12582e81395249c9305b679cf7a4470a1c3b2e83811e3c798c09268b4da9319752ce55a8d0a7623c96e29ded30d07840349b16eb9e1bf078896f0d4832984217920aa6f9b4a935167238dbb97365bc259ef1a8d890e543e1e65125ecacb099f697b7239a58cf2dca45990fb1a417702c84e973907a21cbaff68346ed7ee386a7c86e5f1ffbc5efa703ac5d8d60ca6df45e21ff9bccd3a565a0dfec95c68c12241015c7b47cc4223a8d0bd8cec5b174b4a35ceddd724ee0ccd559d6f0159af9c8cbb8c7dbccc627a6d9cff454'),
    ('csv_data_two.zip','https://storage.googleapis.com/kaggle-data-sets/112501/271525/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240922%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240922T210524Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=23139c39021a7bcab7fe7b263f14fce86d586c969d6708455ea661008e808c1f1b76510a8c221f5c7210d37e9522b48b93bb9f4c81d1f8769550d8eb1ddfed869ca26c91affeea33aaaa910b3e03e0a4eed950ef4706b0f36f47fcbd436f1547ea05c10e9ab42eea874e8f20841b07d617d033925001f08dd66976b54a7b1d8e3092e3ec3c6bc22f64b0ffb53a8526414785848c2dddd6be3134ae0b25274ef12f0f615943f382d9fbacbc407931b2b17020bd946382a5fb005e41e6b216254b57f4a4ef0b27888ee100fe9366dbbe0f884683ab44a28d40a228c1fc03ce6ad967f178962df118bc91f3458d36323de7ca5ae5cea4939da09238c0be019f7c4a')
]


def print_progress(currently_saved: int, file_size: int):
    print("\r{:.1f}/100.0%".format(min((currently_saved/file_size) * 100, 100)), end='\r')

def print_downloading(step, step_size = 2000):
    if step % step_size == 0:
        print(" " * len("Downloading..."), end='\r')
    dots = (step // step_size % 3) + 1
    print(f"Downloading{'.' * dots}", end='\r')

for archive_name, url in urls:
    print('Downloading ', archive_name)
    response = requests.get(url, stream=True)
    try:
        content_length = int(response.headers['Content-Length'])
    except KeyError:
        # hard coded value for edf value (request is chunked, no information for content length
        content_length = 0

    with open(archive_name, 'wb') as file:
        chunk_size = 10 * 1000 * 1024
        if content_length == 0:
            print_downloading(0)
        else:
            print_progress(0, content_length)

        for index, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
            file.write(chunk)
            if content_length != 0:
                print_progress(chunk_size * (index + 1), content_length)
            else:
                print_downloading(index)

