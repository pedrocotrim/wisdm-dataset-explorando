import requests
import zipfile
import io


def main() -> None:
    url = 'https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip'

    with io.BytesIO(requests.get(url).content) as result:
        with zipfile.ZipFile(result) as outer_zip:
            with zipfile.ZipFile(outer_zip.open('wisdm-dataset.zip')) as inner_zip:
                inner_zip.extractall()

if __name__ == "__main__":
    main()