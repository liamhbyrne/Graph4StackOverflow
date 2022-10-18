from stackapi import StackAPI, StackAPIError
import logging
logging.basicConfig(level=logging.INFO)


class StackExchangeAPI:
    def __init__(self):
        try:
            self._SITE = StackAPI('stackoverflow')
        except StackAPIError as e:
            logging.error("   Error URL: {}".format(e.url))
            logging.error("   Error Code: {}".format(e.code))
            logging.error("   Error Error: {}".format(e.error))
            logging.error("   Error Message: {}".format(e.message))



if __name__ == '__main__':
    a = StackExchangeAPI()
