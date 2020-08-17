import glob
import json


def _load_json(file):
    """Loads a JSON file as a dict."""
    with open(file) as f:
        return json.load(f)


def _scrape_result(datum):
    """Scrapes the results from one datum.

    A datum here is a municipality (or a district) encoded as a dict. The keys
    are in German, so we return a dict with keys that make sense...
    """
    num_yes = datum['resultat']['jaStimmenAbsolut']
    num_valid = datum['resultat']['gueltigeStimmen']
    num_total = datum['resultat']['eingelegteStimmzettel']
    num_eligible = datum['resultat']['anzahlStimmberechtigte']
    yes_percent = (num_yes / num_valid
                   if num_yes is not None and num_valid is not None else None)
    turnout = (num_total / num_eligible
               if num_total is not None and num_eligible is not None else None)
    return {
        'num_yes': num_yes,
        'num_no': datum['resultat']['neinStimmenAbsolut'],
        'num_valid': num_valid,
        'num_total': num_total,
        'num_eligible': num_eligible,
        'yes_percent': yes_percent,
        'turnout': turnout
    }


def scrape_referendum(file):
    """Scrapes Swiss referendum results from a JSON file.

    Returns a list of dicts. Each dict contains data about a municipal result,
    such as the number of "yes", the timestamp of the result, and to which
    municipality the result comes from.
    """
    data = _load_json(file)
    results = list()
    timestamp = data['timestamp']
    for datum in data['schweiz']['vorlagen']:
        for canton in datum['kantone']:
            # Count the Kreise as municipality if they exist.
            kreise = canton.get('zaehlkreise', [])
            for muni in canton['gemeinden'] + kreise:
                ogd = int(muni['geoLevelnummer'])
                # Initialize municipal result.
                result = {
                    'vote': datum['vorlagenId'],
                    'municipality': ogd,
                    'timestamp': timestamp
                }
                # Scrape municipal result.
                result.update(_scrape_result(muni))
                results.append(result)
    return results


def scrape_referenda(data_dir):
    """Scrapes all the referendum results from a directory of JSON files.

    Each JSON file is one snapshot of the progress of the vote, sampled every
    two minutes. This function returns a list of list of results. See
    `scrape_referendum(file)` for the structure of the inner list.
    """
    data = list()
    files = glob.glob(f'{data_dir}/*.json')
    for file in files:
        # Scrape data.
        data.append(scrape_referendum(file))
    return data
