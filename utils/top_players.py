from scraping.fpl_api import *
from processing.parsers import *

def main():
    data = get_data()
    parse_top_players(data, 'data/2022-23')

if __name__ == '__main__':
    main()
