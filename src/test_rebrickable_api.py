import rebrick
import json
from dotenv import load_dotenv
import os

load_dotenv()


rb_api_key = os.getenv("REBRICKABLE_API_KEY")
rebrick.init(rb_api_key)


def get_set_theme_id(set_num: int) -> int:
    """
    Given a LEGO set number, return the theme ID it belongs to using the Rebrickable API.
    Args:
        set_num (int): The LEGO set number (e.g., 8121)
    Returns:
        int: The theme ID associated with the given set number.
    """
    set_info = rebrick.lego.get_set(set_num)
    set_info_data = json.loads(set_info.read())

    return set_info_data["theme_id"]


def get_theme_name(theme_id: int) -> str:
    """
    Given a theme ID, return the theme name using the Rebrickable API.
    Args:
        theme_id (int): The theme ID.
    Returns:
        str: The name of the theme.
    """
    theme_info = rebrick.lego.get_theme(theme_id)
    return json.loads(theme_info.read())["name"]


def main():
    set_id = 8121
    theme_id = get_set_theme_id(set_id)
    theme_name = get_theme_name(theme_id)
    print(f"Set {set_id} belongs to theme '{theme_name}' (ID: {theme_id})")


if __name__ == "__main__":
    main()