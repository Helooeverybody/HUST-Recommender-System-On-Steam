import json


class Mapping:
    """Mapping object used for CF.
    ### Attributes:
    - user_to_index: map each user_id to an user index
    - index_to_user: inverse mapping of user_to_index
    - item_to_index: map each app_id to an item index
    - index_to_item: inverse mapping of item_to_index
    - item_to_name: map each app_id to its name
    """

    def __init__(self, fname: str) -> None:
        """Create a mapping from json file"""
        with open(fname, "r") as data:
            json_obj = json.load(data)
        self.user_to_index = json_obj["user_to_index"]
        self.index_to_user = json_obj["index_to_user"]
        self.item_to_index = json_obj["item_to_index"]
        self.index_to_item = json_obj["user_to_index"]
        self.item_to_name = json_obj["user_to_index"]


if __name__ == "__main__":
    M = Mapping("data/steam_60/mapping.json")
    print("Users:", len(M.user_to_index))
    print("Items:", len(M.item_to_index))
