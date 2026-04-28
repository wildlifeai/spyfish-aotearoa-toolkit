# Source: https://github.com/biigle/community-resources/blob/master/biigle/biigle.py
# Commit: df7eddbcaa5bde01538fc3e4c60d11655109a99b (2021-04-30)


import copy

import requests
from requests.auth import HTTPBasicAuth

from sftk.common import BIIGLE_API_EMAIL, BIIGLE_API_TOKEN


class Api(object):
    def __init__(
        self, email="", token="", base_url="https://biigle.de/api/v1", headers={}
    ):
        """Create a new instance.

        Args:
           email (str): The email address of the user.
           token (str): The API token of the user.

        Kwargs:
           base_url (str): Base URL to use for the API URL. Default: `'https://biigle.de/api/v1'`.
           headers (dict): Default headers to use for each request. Default: `{'Accept': 'application/json'}`.
        """
        email = email if email else BIIGLE_API_EMAIL
        token = token if token else BIIGLE_API_TOKEN
        if email is None or token is None:
            raise ValueError(
                "BIIGLE API credentials are required. Provide email and token or set environment variables."
            )
        self.auth = HTTPBasicAuth(email, token)
        self.base_url = base_url
        self.headers = {"Accept": "application/json"}
        self.headers.update(headers)

    def call(self, method, url, raise_for_status=True, *args, **kwargs):
        """Perform an API call

        In addition to the method and URL, any args or kwargs of the requests method are
        accepted.

        Args:
           method: The requests method to use for the api call.
           url: The API endpoint to call.
           raise_for_status: Raise an exception if the response code is not ok.
        """
        if "headers" in kwargs:
            headers = copy.deepcopy(self.headers)
            headers.update(kwargs["headers"])
        else:
            headers = self.headers
        kwargs["headers"] = headers
        kwargs["auth"] = self.auth

        response = method("{}/{}".format(self.base_url, url), *args, **kwargs)

        if raise_for_status:
            if response.status_code == 422:
                body = response.json()
                raise Exception(body["message"], body["errors"])
            else:
                response.raise_for_status()

        return response

    def get(self, url, *args, **kwargs):
        """Perform a GET request to the API

        See the `call` method for available arguments.
        """
        return self.call(requests.get, url, *args, **kwargs)

    def post(self, url, *args, **kwargs):
        """Perform a POST request to the API

        See the `call` method for available arguments.
        """
        return self.call(requests.post, url, *args, **kwargs)

    def put(self, url, *args, **kwargs):
        """Perform a PUT request to the API

        See the `call` method for available arguments.
        """
        return self.call(requests.put, url, *args, **kwargs)

    def delete(self, url, *args, **kwargs):
        """Perform a DELETE request to the API

        See the `call` method for available arguments.
        """
        return self.call(requests.delete, url, *args, **kwargs)
