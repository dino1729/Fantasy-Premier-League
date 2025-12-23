from processing.parsers import *
from processing.cleaners import *
from scraping.fpl_api import *
from processing.collector import collect_gw, merge_gw
from scraping.understat import parse_epl_data
from utils.config import SEASON
import csv

def parse_data():
    """ Parse and store all the data
    """
    season = SEASON
    base_filename = 'data/' + season + '/'
    print("Getting data")
    data = get_data()
    print("Parsing summary data")
    parse_players(data["elements"], base_filename)
    xPoints = []
    for e in data["elements"]:
        xPoint = {}
        xPoint['id'] = e['id']
        xPoint['xP'] = e['ep_this']
        xPoints += [xPoint]
    gw_num = 0
    events = data["events"]
    for event in events:
        if event["is_current"] == True:
            gw_num = event["id"]
    print("Cleaning summary data")
    clean_players(base_filename + 'players_raw.csv', base_filename)
    print("Getting fixtures data")
    fixtures(base_filename)
    print("Getting teams data")
    parse_team_data(data["teams"], base_filename)
    print("Extracting player ids")
    id_players(base_filename + 'players_raw.csv', base_filename)
    player_ids = get_player_ids(base_filename)
    num_players = len(data["elements"])
    player_base_filename = base_filename + 'players/'
    gw_base_filename = base_filename + 'gws/'
    print("Extracting player specific data")
    for i,name in player_ids.items():
        player_data = get_individual_player_data(i)
        parse_player_history(player_data["history_past"], player_base_filename, name, i)
        parse_player_gw_history(player_data["history"], player_base_filename, name, i)
    if gw_num > 0:
        print("Writing expected points")
        with open(os.path.join(gw_base_filename, 'xP' + str(gw_num) + '.csv'), 'w+') as outf:
            w = csv.DictWriter(outf, ['id', 'xP'])
            w.writeheader()
            for xp in xPoints:
                w.writerow(xp)
        
        # Clear merged_gw.csv before collecting all GWs (to avoid duplicates)
        merged_gw_path = os.path.join(gw_base_filename, 'merged_gw.csv')
        if os.path.exists(merged_gw_path):
            os.remove(merged_gw_path)
        
        print(f"Collecting gw scores for GW1-{gw_num}")
        # Collect ALL gameweeks from 1 to current, not just the current one
        for gw in range(1, gw_num + 1):
            collect_gw(gw, player_base_filename, gw_base_filename, base_filename)
        
        print(f"Merging gw scores for GW1-{gw_num}")
        # Merge ALL gameweeks into merged_gw.csv
        for gw in range(1, gw_num + 1):
            merge_gw(gw, gw_base_filename)
    #understat_filename = base_filename + 'understat'
    #parse_epl_data(understat_filename)

def fixtures(base_filename):
    data = get_fixtures_data()
    parse_fixtures(data, base_filename)

def main():
    parse_data()

if __name__ == "__main__":
    main()
