from stackapi import StackAPI, StackAPIError
import logging
from typing import List

logging.basicConfig(level=logging.INFO)


class StackExchangeAPI:
    def __init__(self, max_pages: int, page_size: int):
        try:
            self._SITE = StackAPI('stackoverflow')
            self._SITE.max_pages = max_pages
            self._SITE.page_size = page_size
        except StackAPIError as e:
            logging.error("   Error URL: {}".format(e.url))
            logging.error("   Error Code: {}".format(e.code))
            logging.error("   Error Error: {}".format(e.error))
            logging.error("   Error Message: {}".format(e.message))

    """
    The following methods will be used for data collection in Iteration 1:
    User-expertise graph
    """

    def get_user_ids(self) -> List[int]:
        """
        Request a set of users and return their ids.
        :return: List of ids
        """
        response = self._SITE.fetch(
            'users', min=1000
        )
        user_ids = [user['account_id'] for user in response['items']]
        return user_ids

    def get_user_activity(self, account_ids: List[int]):
        """
        Get questions, answers, comments asked by the users
        :param account_ids: List of user ids
        :return:
        """
        if len(account_ids) > 100:
            logging.error("get_user_activity: account_ids should not have more than 100 ids.")
            return None
        response_answers = self._SITE.fetch(
            'users/{ids}/answers', ids=account_ids, body=True
        )
        print(response_answers)

        response_questions = self._SITE.fetch(
            'users/{ids}/questions', ids=account_ids, body=True
        )

        print(response_questions)

        response_comments = self._SITE.fetch(
            'users/{ids}/comments', ids=account_ids
        )

        print(response_comments)


if __name__ == '__main__':
    a = StackExchangeAPI(1, 100)
    ids = a.get_user_ids()
    print(len(ids))
    a.get_user_activity(ids)
