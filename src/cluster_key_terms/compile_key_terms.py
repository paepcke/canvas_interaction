terms_set = set()

with open('key-terms.txt', 'r') as f:
	for line in f:
		if ',' in line:
			terms = line.split(',')
			for t in terms:
				terms_set.add(t.strip())
		else:
			terms_set.add(line.strip())

with open('final-key-terms.txt', 'w') as fo:
	for ts in terms_set:
		fo.write(ts+'\n')