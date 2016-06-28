files = ['01_01_Introduction.txt',
		 '02_01_The_Relational_Model.txt',
		 '02_02_Querying_Relational_Databases.txt',
		 ]
file_terms = []

with open('Introduction_to_Databases_Captions/01_01_Introduction.txt') as f:
	for line in f:
		if "|" not in line:
			continue
		sentence, term = line.split("|")
		term = term.strip().replace("*", "")
		if len(term) > 2 and term not in file_terms:
			file_terms.append(term)

for ft in file_terms:
	print ft