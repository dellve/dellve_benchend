import csv
import os

def csv_get_problem_set(name):
    result = []
    name = os.path.join(os.path.dirname(__file__), '../problemsets/' + name)
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(reader, None)  # skip header
        for row in reader:
            result.append([int(s.strip(',')) for s in filter(None, row)])

    return result
        
