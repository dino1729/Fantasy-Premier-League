import requests
import json
import time

def get_data():
    """ Retrieve the fpl player data from the hard-coded url
    """
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    responseStr = response.text
    data = json.loads(responseStr)
    return data

def get_individual_player_data(player_id):
    """ Retrieve the player-specific detailed data

    Args:
        player_id (int): ID of the player whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/element-summary/"
    full_url = base_url + str(player_id) + "/"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_entry_data(entry_id):
    """ Retrieve the summary/history data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/history/"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_entry_personal_data(entry_id):
    """ Retrieve the summary/history data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_entry_gws_data(entry_id,num_gws,start_gw=1):
    """ Retrieve the gw-by-gw data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    gw_data = []
    for i in range(start_gw, num_gws+1):
        full_url = base_url + str(entry_id) + "/event/" + str(i) + "/picks/"
        response = ''
        while response == '':
            try:
                response = requests.get(full_url)
            except:
                time.sleep(5)
        if response.status_code != 200:
            raise Exception("Response was code " + str(response.status_code))
        data = json.loads(response.text)
        gw_data += [data]
    return gw_data

def get_entry_transfers_data(entry_id):
    """ Retrieve the transfer data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/transfers/"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_fixtures_data():
    """ Retrieve the fixtures data for the season
    """
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = ''
    while response == '':
        try:
            response = requests.get(url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_classic_league_standings(league_id, page=1):
    """ Retrieve classic league standings with pagination support

    Args:
        league_id (int): ID of the classic league
        page (int): Page number (1-indexed), each page returns up to 50 entries

    Returns:
        dict: League info and standings data including:
            - league: {id, name, ...}
            - standings: {has_next, page, results: [{entry, player_name, entry_name, rank, total, ...}]}
    """
    base_url = "https://fantasy.premierleague.com/api/leagues-classic/"
    full_url = f"{base_url}{league_id}/standings/?page_standings={page}"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def get_entry_picks_for_gw(entry_id, gw):
    """ Retrieve picks for a specific entry and gameweek

    Args:
        entry_id (int): ID of the team
        gw (int): Gameweek number

    Returns:
        dict: Picks data including picks list, entry_history, active_chip
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = f"{base_url}{entry_id}/event/{gw}/picks/"
    response = ''
    while response == '':
        try:
            response = requests.get(full_url)
        except:
            time.sleep(5)
    if response.status_code != 200:
        raise Exception("Response was code " + str(response.status_code))
    data = json.loads(response.text)
    return data

def main():
    data = get_data()
    with open('raw.json', 'w') as outf:
        json.dump(data, outf)

if __name__ == '__main__':
    main()
