from config import Config



def calculate_bin_id(real_position: float, offset: int) -> int:
    """Calculate the bin ID for a given real position and offset."""
    bin_id = int((real_position - Config.MIN_POSITION) / Config.PRECISION) + offset
    return bin_id


if __name__ == "__main__":
    # Example usage
    position = 150.5
    offset = 628
    bin_id = calculate_bin_id(position, offset)
    print(f"Bin ID for position {position} with offset {offset} is: {bin_id}")