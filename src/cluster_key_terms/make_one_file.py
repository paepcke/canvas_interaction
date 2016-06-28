filenames = ['01_01_Introduction.txt', '02_01_The_Relational_Model.txt', '02_02_Querying_Relational_Databases.txt', ]
with open('path/to/output/file', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)