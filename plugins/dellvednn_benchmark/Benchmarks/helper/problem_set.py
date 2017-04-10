import csv
import os
 
def csv_get_header(name):
    name = os.path.join(os.path.dirname(__file__), '../problemsets/' + name)
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|', skipinitialspace=True)
        return next(reader)
         
def csv_get_problem_set(name):
    result = []
    name = os.path.join(os.path.dirname(__file__), '../problemsets/' + name)

    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|', skipinitialspace=True)
        next(reader, None) # Skip header 
        for row in reader:
            result.append([int(s) for s in row])

    return result
        
